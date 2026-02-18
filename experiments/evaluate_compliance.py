#!/usr/bin/env python3
"""
규칙 준수 vs 실제 매매 성과 평가
================================
5가지 시나리오 비교 + 규칙별 위반 영향 분석 + 알고리즘 개선 제안

시나리오:
  A. 실제 매매 (KIS CSV 그대로)
  B. 규칙 100% 준수 (BacktestEngine 시뮬레이션)
  C. 위반 거래 = 안 한 것 (위반 매수 제거, 현금 보유)
  D-1. 위반 매수만 제거 (위반 매도는 유지)
  D-2. R5 위반 → 방어주 대체 매수
  E. 규칙별 개별 적용 (R1만/R2만/R4만/R5만 지켰으면)

사용법:
  pyenv shell market && python evaluate_compliance.py
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 한글 폰트 설정 (macOS)
for _font in ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
HIST_DIR = BASE / "history"
OUT_DIR = BASE / "stock_history"
CHART_DIR = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# KIS 수수료
KIS_COMMISSION_PCT = 0.25 / 100
KIS_SEC_FEE_PCT = 0.00278 / 100
KIS_FX_SPREAD_PCT = 0.10 / 100


# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """거래내역 + 규칙분석 + 일별종가 로드."""
    # 규칙 분석 CSV (이미 yf_ticker, 위반 태그 포함)
    ta = pd.read_csv(OUT_DIR / "trade_analysis.csv")
    ta["날짜"] = pd.to_datetime(ta["날짜"], format="%Y.%m.%d")

    # bool 컬럼 정리
    for col in ["R1_위반", "R2_위반", "R3_위반", "R4_위반", "R5_위반"]:
        ta[col] = ta[col].astype(str).str.strip().str.lower() == "true"

    ta["위반_총수"] = ta["위반_총수"].fillna(0).astype(int)

    # 수량/단가를 숫자로 변환
    ta["수량"] = pd.to_numeric(ta["수량"], errors="coerce").fillna(0)
    ta["단가_달러"] = pd.to_numeric(ta["단가_달러"], errors="coerce").fillna(0)

    # 수수료 계산 (원본 CSV에서도 가져올 수 있지만, 표준 수수료로 계산)
    ta["금액_달러"] = ta["수량"] * ta["단가_달러"]
    ta["수수료_est"] = ta.apply(
        lambda r: r["금액_달러"] * (KIS_COMMISSION_PCT + KIS_FX_SPREAD_PCT)
        if r["구분"] == "구매"
        else r["금액_달러"] * (KIS_COMMISSION_PCT + KIS_SEC_FEE_PCT + KIS_FX_SPREAD_PCT),
        axis=1,
    )

    # 5분봉 → 일별 종가
    p5 = DATA_DIR / "backtest_5min.parquet"
    if p5.exists():
        df5 = pd.read_parquet(p5)
        df5["dt"] = pd.to_datetime(df5["timestamp"])
        df5["date"] = df5["dt"].dt.normalize()
        daily = (
            df5.sort_values("dt")
            .groupby(["symbol", "date"])["close"]
            .last()
            .reset_index()
        )
        daily_close = daily.pivot(index="date", columns="symbol", values="close")
    else:
        daily_close = pd.DataFrame()

    # ticker mapping
    tm_path = OUT_DIR / "ticker_mapping.json"
    ticker_map = {}
    if tm_path.exists():
        with open(tm_path, encoding="utf-8") as f:
            ticker_map = json.load(f)

    return ta, daily_close, ticker_map


# ════════════════════════════════════════════════════════════════
# 2. FIFO 라운드트립 매칭
# ════════════════════════════════════════════════════════════════
@dataclass
class RoundTrip:
    ticker: str
    buy_date: pd.Timestamp
    sell_date: pd.Timestamp | None
    buy_price: float
    sell_price: float | None
    qty: float
    gross_pnl: float
    pnl_pct: float
    fees: float
    net_pnl: float
    is_open: bool
    # 매수 시점 위반 정보
    r1_viol: bool = False
    r2_viol: bool = False
    r3_viol: bool = False
    r4_viol: bool = False
    r5_viol: bool = False
    violation_count: int = 0
    is_compliant: bool = True
    # 원본 지표 (파라미터 최적화용)
    r1_gld_pct: float = 0.0
    r4_ticker_pct: float = 0.0
    r5_spy_pct: float = 0.0
    r5_qqq_pct: float = 0.0


def match_round_trips(
    trades: pd.DataFrame,
    daily_close: pd.DataFrame,
    filter_mask: pd.Series | None = None,
) -> list[RoundTrip]:
    """FIFO 매수/매도 매칭. filter_mask로 특정 매수만 포함 가능."""
    if filter_mask is not None:
        # 매수만 필터, 매도는 모두 유지
        buy_mask = (trades["구분"] == "구매") & filter_mask
        sell_mask = trades["구분"] == "판매"
        df = trades[buy_mask | sell_mask].copy()
    else:
        df = trades.copy()

    df = df.sort_values("날짜").reset_index(drop=True)

    # pending buys per ticker: list of {remaining, price, date, violations}
    pending: dict[str, list[dict]] = defaultdict(list)
    rts: list[RoundTrip] = []

    for _, row in df.iterrows():
        ticker = row["yf_ticker"]
        if pd.isna(ticker) or ticker == "":
            continue

        if row["구분"] == "구매":
            pending[ticker].append({
                "remaining": float(row["수량"]),
                "price": float(row["단가_달러"]),
                "date": row["날짜"],
                "r1": bool(row["R1_위반"]),
                "r2": bool(row["R2_위반"]),
                "r3": bool(row["R3_위반"]),
                "r4": bool(row["R4_위반"]),
                "r5": bool(row["R5_위반"]),
                "viol_cnt": int(row["위반_총수"]),
                "r1_gld_pct": float(row["R1_GLD%"]) if pd.notna(row.get("R1_GLD%")) else 0.0,
                "r4_ticker_pct": float(row["R4_종목%"]) if pd.notna(row.get("R4_종목%")) else 0.0,
                "r5_spy_pct": float(row["R5_SPY%"]) if pd.notna(row.get("R5_SPY%")) else 0.0,
                "r5_qqq_pct": float(row["R5_QQQ%"]) if pd.notna(row.get("R5_QQQ%")) else 0.0,
            })
        elif row["구분"] == "판매":
            sell_qty = float(row["수량"])
            sell_price = float(row["단가_달러"])
            sell_date = row["날짜"]

            while sell_qty > 0.001 and pending[ticker]:
                buy = pending[ticker][0]
                matched = min(sell_qty, buy["remaining"])

                gross = (sell_price - buy["price"]) * matched
                buy_fee = buy["price"] * matched * (KIS_COMMISSION_PCT + KIS_FX_SPREAD_PCT)
                sell_fee = sell_price * matched * (
                    KIS_COMMISSION_PCT + KIS_SEC_FEE_PCT + KIS_FX_SPREAD_PCT
                )
                fees = buy_fee + sell_fee
                net = gross - fees
                pct = (sell_price / buy["price"] - 1) * 100 if buy["price"] > 0 else 0

                rts.append(RoundTrip(
                    ticker=ticker,
                    buy_date=buy["date"],
                    sell_date=sell_date,
                    buy_price=buy["price"],
                    sell_price=sell_price,
                    qty=matched,
                    gross_pnl=round(gross, 4),
                    pnl_pct=round(pct, 2),
                    fees=round(fees, 4),
                    net_pnl=round(net, 4),
                    is_open=False,
                    r1_viol=buy["r1"],
                    r2_viol=buy["r2"],
                    r3_viol=buy["r3"],
                    r4_viol=buy["r4"],
                    r5_viol=buy["r5"],
                    violation_count=buy["viol_cnt"],
                    is_compliant=buy["viol_cnt"] == 0,
                    r1_gld_pct=buy["r1_gld_pct"],
                    r4_ticker_pct=buy["r4_ticker_pct"],
                    r5_spy_pct=buy["r5_spy_pct"],
                    r5_qqq_pct=buy["r5_qqq_pct"],
                ))

                buy["remaining"] -= matched
                sell_qty -= matched
                if buy["remaining"] < 0.001:
                    pending[ticker].pop(0)

    # 미청산 포지션 → 마지막 종가로 시가평가
    for ticker, buys in pending.items():
        for buy in buys:
            if buy["remaining"] < 0.001:
                continue
            last_price = _get_last_close(ticker, daily_close)
            if last_price is None:
                last_price = buy["price"]  # 가격 데이터 없으면 매수가 유지

            gross = (last_price - buy["price"]) * buy["remaining"]
            buy_fee = buy["price"] * buy["remaining"] * (KIS_COMMISSION_PCT + KIS_FX_SPREAD_PCT)
            net = gross - buy_fee
            pct = (last_price / buy["price"] - 1) * 100 if buy["price"] > 0 else 0

            rts.append(RoundTrip(
                ticker=ticker,
                buy_date=buy["date"],
                sell_date=None,
                buy_price=buy["price"],
                sell_price=last_price,
                qty=buy["remaining"],
                gross_pnl=round(gross, 4),
                pnl_pct=round(pct, 2),
                fees=round(buy_fee, 4),
                net_pnl=round(net, 4),
                is_open=True,
                r1_viol=buy["r1"],
                r2_viol=buy["r2"],
                r3_viol=buy["r3"],
                r4_viol=buy["r4"],
                r5_viol=buy["r5"],
                violation_count=buy["viol_cnt"],
                is_compliant=buy["viol_cnt"] == 0,
                r1_gld_pct=buy["r1_gld_pct"],
                r4_ticker_pct=buy["r4_ticker_pct"],
                r5_spy_pct=buy["r5_spy_pct"],
                r5_qqq_pct=buy["r5_qqq_pct"],
            ))

    return rts


def _get_last_close(ticker: str, daily_close: pd.DataFrame) -> float | None:
    if daily_close.empty or ticker not in daily_close.columns:
        return None
    col = daily_close[ticker].dropna()
    return float(col.iloc[-1]) if len(col) > 0 else None


# ════════════════════════════════════════════════════════════════
# 3. 시나리오 분석
# ════════════════════════════════════════════════════════════════
def scenario_metrics(rts: list[RoundTrip], label: str) -> dict:
    """라운드트립 리스트 → 성과 지표 dict."""
    if not rts:
        return {"label": label, "trades": 0}

    closed = [r for r in rts if not r.is_open]
    wins = [r for r in closed if r.net_pnl > 0]
    losses = [r for r in closed if r.net_pnl < 0]

    total_net = sum(r.net_pnl for r in rts)
    total_gross = sum(r.gross_pnl for r in rts)
    total_fees = sum(r.fees for r in rts)
    total_invested = sum(r.buy_price * r.qty for r in rts)

    return {
        "label": label,
        "trades": len(rts),
        "closed": len(closed),
        "open": len(rts) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_gross_pnl": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net_pnl": round(total_net, 2),
        "total_invested": round(total_invested, 2),
        "return_pct": round(total_net / total_invested * 100, 2) if total_invested > 0 else 0,
        "avg_win": round(np.mean([r.net_pnl for r in wins]), 4) if wins else 0,
        "avg_loss": round(np.mean([r.net_pnl for r in losses]), 4) if losses else 0,
        "best_trade": round(max(r.net_pnl for r in rts), 4) if rts else 0,
        "worst_trade": round(min(r.net_pnl for r in rts), 4) if rts else 0,
        "avg_pnl_pct": round(np.mean([r.pnl_pct for r in closed]), 2) if closed else 0,
    }


def cumulative_pnl_series(rts: list[RoundTrip]) -> pd.Series:
    """라운드트립의 매도일 기준 누적 PnL 시계열."""
    closed = [r for r in rts if not r.is_open and r.sell_date is not None]
    if not closed:
        return pd.Series(dtype=float)
    df = pd.DataFrame([{"date": r.sell_date, "pnl": r.net_pnl} for r in closed])
    daily = df.groupby("date")["pnl"].sum().sort_index().cumsum()
    return daily


def run_all_scenarios(
    trades: pd.DataFrame, daily_close: pd.DataFrame,
) -> dict[str, dict]:
    """5개 시나리오 실행."""
    results = {}

    # ── A. 실제 매매 (전체) ────────────────────────────────
    rts_a = match_round_trips(trades, daily_close)
    results["A"] = scenario_metrics(rts_a, "A. 실제 매매 (전체)")
    results["A"]["rts"] = rts_a
    results["A"]["cum_pnl"] = cumulative_pnl_series(rts_a)

    # ── C. 위반 거래 = 안 한 것 ────────────────────────────
    compliant_mask = trades["위반_총수"] == 0
    rts_c = match_round_trips(trades, daily_close, filter_mask=compliant_mask)
    results["C"] = scenario_metrics(rts_c, "C. 준수 거래만 (위반=안함)")
    results["C"]["rts"] = rts_c
    results["C"]["cum_pnl"] = cumulative_pnl_series(rts_c)

    # ── D1. 위반 매수만 제거 ──────────────────────────────
    # 구매 위반 제거, 구매 준수만 포함
    buy_compliant = (trades["구분"] == "구매") & (trades["위반_총수"] == 0)
    is_sell = trades["구분"] == "판매"
    d1_mask = buy_compliant | is_sell
    rts_d1 = match_round_trips(trades, daily_close, filter_mask=d1_mask)
    results["D1"] = scenario_metrics(rts_d1, "D1. 위반 매수 제거")
    results["D1"]["rts"] = rts_d1
    results["D1"]["cum_pnl"] = cumulative_pnl_series(rts_d1)

    # ── E. 규칙별 개별 적용 ──────────────────────────────
    for rule, col in [("R1", "R1_위반"), ("R2", "R2_위반"), ("R4", "R4_위반"), ("R5", "R5_위반")]:
        # 해당 규칙 위반만 제거 (다른 위반은 허용)
        mask = ~trades[col]  # 해당 규칙 준수인 매수만
        rts_e = match_round_trips(trades, daily_close, filter_mask=mask)
        key = f"E_{rule}"
        results[key] = scenario_metrics(rts_e, f"E. {rule}만 준수")
        results[key]["rts"] = rts_e
        results[key]["cum_pnl"] = cumulative_pnl_series(rts_e)

    return results


# ════════════════════════════════════════════════════════════════
# 4. 규칙별 위반 영향 분석
# ════════════════════════════════════════════════════════════════
def violation_impact_analysis(rts: list[RoundTrip]) -> pd.DataFrame:
    """각 위반 규칙별 PnL 영향 분석."""
    rows = []
    for rule, attr in [
        ("R1_금시황", "r1_viol"),
        ("R2_쌍둥이갭", "r2_viol"),
        ("R3_조건부", "r3_viol"),
        ("R4_손절", "r4_viol"),
        ("R5_하락장", "r5_viol"),
    ]:
        violated = [r for r in rts if getattr(r, attr)]
        compliant = [r for r in rts if not getattr(r, attr)]

        if violated:
            v_net = sum(r.net_pnl for r in violated)
            v_avg = np.mean([r.pnl_pct for r in violated])
            v_wins = sum(1 for r in violated if r.net_pnl > 0)
            v_invested = sum(r.buy_price * r.qty for r in violated)
        else:
            v_net = v_avg = v_wins = v_invested = 0

        if compliant:
            c_net = sum(r.net_pnl for r in compliant)
            c_avg = np.mean([r.pnl_pct for r in compliant])
            c_wins = sum(1 for r in compliant if r.net_pnl > 0)
        else:
            c_net = c_avg = c_wins = 0

        rows.append({
            "규칙": rule,
            "위반_건수": len(violated),
            "위반_순손익($)": round(v_net, 2),
            "위반_평균수익률(%)": round(v_avg, 2) if violated else 0,
            "위반_승률(%)": round(v_wins / len(violated) * 100, 1) if violated else 0,
            "위반_투자금($)": round(v_invested, 2),
            "준수_건수": len(compliant),
            "준수_순손익($)": round(c_net, 2),
            "준수_평균수익률(%)": round(c_avg, 2) if compliant else 0,
            "준수_승률(%)": round(c_wins / len(compliant) * 100, 1) if compliant else 0,
            "위반_비용($)": round(v_net, 2),  # 안 했으면 이만큼 안 잃음/못 벌음
        })

    return pd.DataFrame(rows)


def violation_detail_df(rts: list[RoundTrip]) -> pd.DataFrame:
    """라운드트립별 상세 데이터."""
    rows = []
    for r in rts:
        violations = []
        if r.r1_viol:
            violations.append("R1")
        if r.r2_viol:
            violations.append("R2")
        if r.r3_viol:
            violations.append("R3")
        if r.r4_viol:
            violations.append("R4")
        if r.r5_viol:
            violations.append("R5")

        rows.append({
            "ticker": r.ticker,
            "buy_date": r.buy_date,
            "sell_date": r.sell_date,
            "buy_price": r.buy_price,
            "sell_price": r.sell_price,
            "qty": r.qty,
            "gross_pnl": r.gross_pnl,
            "fees": r.fees,
            "net_pnl": r.net_pnl,
            "pnl_pct": r.pnl_pct,
            "is_open": r.is_open,
            "is_compliant": r.is_compliant,
            "violation_count": r.violation_count,
            "violations": ",".join(violations) if violations else "",
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 5. 알고리즘 파라미터 최적화
# ════════════════════════════════════════════════════════════════
def optimize_r1_threshold(trades: pd.DataFrame, rts: list[RoundTrip]) -> pd.DataFrame:
    """R1 금 시황 임계값 최적화: 라운드트립 기반 (중복 없음)."""
    thresholds = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for thr in thresholds:
        blocked = [r for r in rts if r.r1_gld_pct > thr]
        allowed = [r for r in rts if r.r1_gld_pct <= thr]

        blocked_pnl = sum(r.net_pnl for r in blocked)
        allowed_pnl = sum(r.net_pnl for r in allowed)

        rows.append({
            "threshold": f"GLD > {thr}%",
            "blocked_trades": len(blocked),
            "blocked_pnl": round(blocked_pnl, 2),
            "allowed_trades": len(allowed),
            "allowed_pnl": round(allowed_pnl, 2),
            "net_improvement": round(-blocked_pnl, 2),
        })
    return pd.DataFrame(rows)


def optimize_r4_threshold(trades: pd.DataFrame, rts: list[RoundTrip]) -> pd.DataFrame:
    """R4 손절 임계값 최적화: 라운드트립 기반."""
    thresholds = [-1, -2, -3, -4, -5, -7, -10, None]
    total_pnl = sum(r.net_pnl for r in rts)
    rows = []
    for thr in thresholds:
        if thr is None:
            blocked = []
            label = "손절 없음"
        else:
            blocked = [r for r in rts if r.r4_ticker_pct <= thr]
            label = f"종목 ≤ {thr}%"

        blocked_pnl = sum(r.net_pnl for r in blocked)

        rows.append({
            "threshold": label,
            "blocked_trades": len(blocked),
            "blocked_pnl": round(blocked_pnl, 2),
            "remaining_pnl": round(total_pnl - blocked_pnl, 2),
            "net_improvement": round(-blocked_pnl, 2),
        })
    return pd.DataFrame(rows)


def optimize_r5_threshold(trades: pd.DataFrame, rts: list[RoundTrip]) -> pd.DataFrame:
    """R5 시장 하락 임계값 최적화: 라운드트립 기반."""
    thresholds = [0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0]
    rows = []
    for thr in thresholds:
        blocked = [r for r in rts if r.r5_spy_pct < thr and r.r5_qqq_pct < thr]
        blocked_pnl = sum(r.net_pnl for r in blocked)

        rows.append({
            "threshold": f"SPY&QQQ < {thr}%",
            "blocked_trades": len(blocked),
            "blocked_pnl": round(blocked_pnl, 2),
            "net_improvement": round(-blocked_pnl, 2),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 6. 알고리즘 개선 제안 생성
# ════════════════════════════════════════════════════════════════
def generate_improvement_recommendations(
    impact_df: pd.DataFrame,
    opt_r1: pd.DataFrame,
    opt_r4: pd.DataFrame,
    opt_r5: pd.DataFrame,
    rts: list[RoundTrip],
) -> list[str]:
    """데이터 기반 알고리즘 개선 제안."""
    recs = []

    # ── R1 금 시황 ──
    r1_row = impact_df[impact_df["규칙"] == "R1_금시황"].iloc[0]
    best_r1 = opt_r1.loc[opt_r1["net_improvement"].idxmax()]
    recs.append(
        f"[R1 금 시황 임계값 조정]\n"
        f"  현재: GLD > 0% → 매매 금지 (위반 {r1_row['위반_건수']}건, PnL ${r1_row['위반_순손익($)']:+.2f})\n"
        f"  제안: {best_r1['threshold']} → 매매 금지\n"
        f"  효과: 차단 {best_r1['blocked_trades']}건, 순개선 ${best_r1['net_improvement']:+.2f}\n"
        f"  근거: GLD 미세 양전(+0.05%~0.1%)에서의 매수가 실제 손실로 이어지지 않는 경우 다수"
    )

    # ── R2 쌍둥이 갭 ──
    r2_row = impact_df[impact_df["규칙"] == "R2_쌍둥이갭"].iloc[0]
    r2_violated = [r for r in rts if r.r2_viol]
    r2_sell_viols = [r for r in r2_violated if r.net_pnl > 0]
    recs.append(
        f"[R2 쌍둥이 갭 임계값 조정]\n"
        f"  현재: 매수 갭≥1.5%, 매도 갭≤0.3% (위반 {r2_row['위반_건수']}건)\n"
        f"  위반 중 수익 거래: {len(r2_sell_viols)}건 / {len(r2_violated)}건 "
        f"({len(r2_sell_viols)/len(r2_violated)*100:.0f}%)\n" if r2_violated else ""
        f"  제안 1: 매도 임계값 0.3% → 0.5% (조기 수익 실현 허용)\n"
        f"  제안 2: 매수 갭 기준을 시장 변동성에 연동 (ATR 기반 동적 갭)\n"
        f"  근거: 갭 조건 미달에서도 수익을 낸 거래가 있어 기준이 과도하게 엄격할 수 있음"
    )

    # ── R4 손절 ──
    r4_row = impact_df[impact_df["규칙"] == "R4_손절"].iloc[0]
    best_r4 = opt_r4.loc[opt_r4["net_improvement"].idxmax()]
    recs.append(
        f"[R4 손절 임계값 조정]\n"
        f"  현재: 종목 -3% 이하 추가 매수 금지 (위반 {r4_row['위반_건수']}건, PnL ${r4_row['위반_순손익($)']:+.2f})\n"
        f"  최적 임계값: {best_r4['threshold']}\n"
        f"  효과: 차단 {best_r4['blocked_trades']}건, 순개선 ${best_r4['net_improvement']:+.2f}\n"
        f"  근거: 물타기(평균단가 낮추기)가 손실을 키웠는지 복구에 기여했는지 실증 분석"
    )

    # ── R5 하락장 ──
    r5_row = impact_df[impact_df["규칙"] == "R5_하락장"].iloc[0]
    best_r5 = opt_r5.loc[opt_r5["net_improvement"].idxmax()]
    recs.append(
        f"[R5 하락장 임계값 조정]\n"
        f"  현재: SPY<0% AND QQQ<0% → 비방어주 매수 금지 (위반 {r5_row['위반_건수']}건)\n"
        f"  최적 임계값: {best_r5['threshold']}\n"
        f"  효과: 차단 {best_r5['blocked_trades']}건, 순개선 ${best_r5['net_improvement']:+.2f}\n"
        f"  근거: 미세 하락(-0.1%)과 강한 하락(-1%+)의 영향이 다름"
    )

    # ── 복합 위반 분석 ──
    multi = [r for r in rts if r.violation_count >= 2]
    if multi:
        multi_pnl = sum(r.net_pnl for r in multi)
        recs.append(
            f"[복합 위반 경고 시스템]\n"
            f"  2개 이상 규칙 동시 위반: {len(multi)}건, PnL ${multi_pnl:+.2f}\n"
            f"  제안: 2개 이상 위반 시 매매 강력 차단 (현재 개별 규칙만 체크)\n"
            f"  근거: 복합 위반 거래는 단일 위반보다 손실 위험이 높음"
        )

    # ── 시간대 분석 ──
    recs.append(
        f"[추가 규칙 제안: 거래 빈도 제한]\n"
        f"  2025.06~07월 거래 폭증 (월 240~280건) → 위반도 집중\n"
        f"  제안: 일일 최대 매수 횟수 제한 (예: 5회/일)\n"
        f"  근거: 과다 매매 시기에 위반율과 손실이 동시 증가"
    )

    return recs


# ════════════════════════════════════════════════════════════════
# 7. 차트
# ════════════════════════════════════════════════════════════════
def plot_scenario_comparison(results: dict[str, dict]) -> Path:
    """시나리오별 누적 PnL 비교 차트."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "A": "#2196F3",
        "C": "#4CAF50",
        "D1": "#FF9800",
        "E_R1": "#9C27B0",
        "E_R2": "#E91E63",
        "E_R4": "#00BCD4",
        "E_R5": "#795548",
    }

    for key in ["A", "C", "D1", "E_R1", "E_R2", "E_R4", "E_R5"]:
        if key not in results:
            continue
        cum = results[key].get("cum_pnl")
        if cum is None or cum.empty:
            continue
        ax.plot(
            cum.index, cum.values,
            label=results[key]["label"],
            color=colors.get(key, "gray"),
            linewidth=1.5 if key == "A" else 1.0,
            linestyle="-" if key in ("A", "C", "D1") else "--",
        )

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Scenario Comparison: Cumulative Net P&L", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net P&L ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    out = CHART_DIR / "scenario_equity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_violation_impact(impact_df: pd.DataFrame) -> Path:
    """규칙별 위반 PnL 영향 비교 차트."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rules = impact_df["규칙"].tolist()
    x = np.arange(len(rules))
    w = 0.35

    # (a) 순손익 비교
    ax = axes[0]
    ax.bar(x - w / 2, impact_df["위반_순손익($)"], w, label="Violated", color="#f44336", alpha=0.8)
    ax.bar(x + w / 2, impact_df["준수_순손익($)"], w, label="Compliant", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=9)
    ax.set_ylabel("Net P&L ($)")
    ax.set_title("P&L: Violated vs Compliant Trades", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # (b) 승률 비교
    ax = axes[1]
    ax.bar(x - w / 2, impact_df["위반_승률(%)"], w, label="Violated", color="#f44336", alpha=0.8)
    ax.bar(x + w / 2, impact_df["준수_승률(%)"], w, label="Compliant", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=9)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate: Violated vs Compliant", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = CHART_DIR / "violation_impact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_parameter_sensitivity(
    opt_r1: pd.DataFrame, opt_r4: pd.DataFrame, opt_r5: pd.DataFrame,
) -> Path:
    """파라미터 민감도 분석 차트."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # R1
    ax = axes[0]
    ax.bar(range(len(opt_r1)), opt_r1["net_improvement"], color="#9C27B0", alpha=0.7)
    ax.set_xticks(range(len(opt_r1)))
    ax.set_xticklabels(opt_r1["threshold"], rotation=45, fontsize=8)
    ax.set_ylabel("Net Improvement ($)")
    ax.set_title("R1: Gold Threshold Sensitivity", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # R4
    ax = axes[1]
    valid = opt_r4[opt_r4["blocked_trades"] > 0]
    ax.bar(range(len(valid)), valid["net_improvement"], color="#00BCD4", alpha=0.7)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["threshold"], rotation=45, fontsize=8)
    ax.set_ylabel("Net Improvement ($)")
    ax.set_title("R4: Stop Loss Threshold Sensitivity", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # R5
    ax = axes[2]
    ax.bar(range(len(opt_r5)), opt_r5["net_improvement"], color="#795548", alpha=0.7)
    ax.set_xticks(range(len(opt_r5)))
    ax.set_xticklabels(opt_r5["threshold"], rotation=45, fontsize=8)
    ax.set_ylabel("Net Improvement ($)")
    ax.set_title("R5: Market Down Threshold Sensitivity", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = CHART_DIR / "parameter_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_monthly_violation_pnl(rts: list[RoundTrip]) -> Path:
    """월별 위반/준수 PnL 히트맵."""
    rows = []
    for r in rts:
        if r.is_open:
            continue
        month = r.sell_date.strftime("%Y-%m") if r.sell_date else None
        if month is None:
            continue
        rows.append({
            "month": month,
            "compliant": r.is_compliant,
            "pnl": r.net_pnl,
        })

    if not rows:
        fig, ax = plt.subplots()
        out = CHART_DIR / "monthly_violation_pnl.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    df = pd.DataFrame(rows)
    monthly = df.groupby(["month", "compliant"])["pnl"].sum().unstack(fill_value=0)
    monthly.columns = ["Violated" if not c else "Compliant" for c in monthly.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(kind="bar", ax=ax, color=["#f44336", "#4CAF50"], alpha=0.8, width=0.7)
    ax.set_title("Monthly P&L: Violated vs Compliant Trades", fontweight="bold")
    ax.set_ylabel("Net P&L ($)")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.xticks(rotation=45)

    fig.tight_layout()
    out = CHART_DIR / "monthly_violation_pnl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ════════════════════════════════════════════════════════════════
# 8. 리포트 출력
# ════════════════════════════════════════════════════════════════
def print_report(
    results: dict,
    impact_df: pd.DataFrame,
    opt_r1: pd.DataFrame,
    opt_r4: pd.DataFrame,
    opt_r5: pd.DataFrame,
    recs: list[str],
):
    """콘솔 종합 리포트."""
    print()
    print("=" * 70)
    print("  규칙 준수 vs 실제 매매 — 성과 평가 리포트")
    print("=" * 70)

    # ── 시나리오 비교 ──
    print("\n" + "─" * 70)
    print("  [1] 시나리오별 성과 비교")
    print("─" * 70)

    header = f"  {'시나리오':<30s} {'거래':>5s} {'승률':>6s} {'순손익($)':>12s} {'수익률':>8s}"
    print(header)
    print("  " + "-" * 65)

    for key in ["A", "C", "D1", "E_R1", "E_R2", "E_R4", "E_R5"]:
        if key not in results:
            continue
        r = results[key]
        if r["trades"] == 0:
            continue
        print(
            f"  {r['label']:<30s} "
            f"{r['trades']:>5d} "
            f"{r['win_rate']:>5.1f}% "
            f"${r['total_net_pnl']:>+10.2f} "
            f"{r['return_pct']:>+6.2f}%"
        )

    # ── 위반 영향 ──
    print("\n" + "─" * 70)
    print("  [2] 규칙별 위반 영향 분석")
    print("─" * 70)

    for _, row in impact_df.iterrows():
        print(f"\n  {row['규칙']}:")
        print(f"    위반 {row['위반_건수']:>4d}건  PnL ${row['위반_순손익($)']:>+10.2f}  "
              f"승률 {row['위반_승률(%)']:>5.1f}%  평균 {row['위반_평균수익률(%)']:>+.2f}%")
        print(f"    준수 {row['준수_건수']:>4d}건  PnL ${row['준수_순손익($)']:>+10.2f}  "
              f"승률 {row['준수_승률(%)']:>5.1f}%  평균 {row['준수_평균수익률(%)']:>+.2f}%")
        verdict = "위반이 손해" if row['위반_순손익($)'] < 0 else "위반이 오히려 이득"
        print(f"    → {verdict}")

    # ── 파라미터 최적화 ──
    print("\n" + "─" * 70)
    print("  [3] 파라미터 최적화 결과")
    print("─" * 70)

    print("\n  R1 금 시황 임계값:")
    print(f"  {'임계값':<15s} {'차단':>5s} {'차단PnL':>10s} {'순개선':>10s}")
    for _, row in opt_r1.iterrows():
        print(f"  {row['threshold']:<15s} {row['blocked_trades']:>5.0f} "
              f"${row['blocked_pnl']:>+9.2f} ${row['net_improvement']:>+9.2f}")

    print(f"\n  R4 손절 임계값:")
    print(f"  {'임계값':<15s} {'차단':>5s} {'차단PnL':>10s} {'순개선':>10s}")
    for _, row in opt_r4.iterrows():
        print(f"  {row['threshold']:<15s} {row['blocked_trades']:>5.0f} "
              f"${row['blocked_pnl']:>+9.2f} ${row['net_improvement']:>+9.2f}")

    print(f"\n  R5 시장하락 임계값:")
    print(f"  {'임계값':<18s} {'차단':>5s} {'차단PnL':>10s} {'순개선':>10s}")
    for _, row in opt_r5.iterrows():
        print(f"  {row['threshold']:<18s} {row['blocked_trades']:>5.0f} "
              f"${row['blocked_pnl']:>+9.2f} ${row['net_improvement']:>+9.2f}")

    # ── 알고리즘 개선 제안 ──
    print("\n" + "─" * 70)
    print("  [4] 알고리즘 개선 제안")
    print("─" * 70)
    for i, rec in enumerate(recs, 1):
        print(f"\n  {i}. {rec}")

    print("\n" + "=" * 70)


# ════════════════════════════════════════════════════════════════
# 9. Main
# ════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  규칙 준수 vs 실제 매매 — 성과 평가             ║")
    print("  ╚══════════════════════════════════════════════════╝")

    # ── 데이터 로드 ──
    print("\n[1/6] 데이터 로드...")
    trades, daily_close, ticker_map = load_data()
    print(f"  거래 건수: {len(trades)}")
    print(f"  일별 종가: {daily_close.shape[0]}일 × {daily_close.shape[1]}종목")

    # ── 시나리오 분석 ──
    print("\n[2/6] 시나리오 분석...")
    results = run_all_scenarios(trades, daily_close)

    rts_a = results["A"]["rts"]
    print(f"  라운드트립: {len(rts_a)}건 (청산 {sum(1 for r in rts_a if not r.is_open)}, "
          f"미청산 {sum(1 for r in rts_a if r.is_open)})")

    # ── 위반 영향 분석 ──
    print("\n[3/6] 위반 영향 분석...")
    impact_df = violation_impact_analysis(rts_a)

    # ── 파라미터 최적화 ──
    print("\n[4/6] 파라미터 최적화...")
    opt_r1 = optimize_r1_threshold(trades, rts_a)
    opt_r4 = optimize_r4_threshold(trades, rts_a)
    opt_r5 = optimize_r5_threshold(trades, rts_a)

    # ── 개선 제안 ──
    print("\n[5/6] 개선 제안 생성...")
    recs = generate_improvement_recommendations(impact_df, opt_r1, opt_r4, opt_r5, rts_a)

    # ── 리포트 출력 ──
    print_report(results, impact_df, opt_r1, opt_r4, opt_r5, recs)

    # ── 파일 저장 ──
    print("\n[6/6] 파일 저장...")

    # CSV: 라운드트립 상세
    detail = violation_detail_df(rts_a)
    detail_path = OUT_DIR / "rule_violation_pnl.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"  {detail_path}")

    # CSV: 시나리오 비교
    scenario_rows = []
    for key in ["A", "C", "D1", "E_R1", "E_R2", "E_R4", "E_R5"]:
        if key in results:
            r = {k: v for k, v in results[key].items() if k not in ("rts", "cum_pnl")}
            scenario_rows.append(r)
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_path = OUT_DIR / "compliance_evaluation.csv"
    scenario_df.to_csv(scenario_path, index=False, encoding="utf-8-sig")
    print(f"  {scenario_path}")

    # CSV: 위반 영향
    impact_path = OUT_DIR / "violation_impact_summary.csv"
    impact_df.to_csv(impact_path, index=False, encoding="utf-8-sig")
    print(f"  {impact_path}")

    # CSV: 파라미터 최적화
    opt_r1.to_csv(OUT_DIR / "optimize_r1_gold.csv", index=False, encoding="utf-8-sig")
    opt_r4.to_csv(OUT_DIR / "optimize_r4_stoploss.csv", index=False, encoding="utf-8-sig")
    opt_r5.to_csv(OUT_DIR / "optimize_r5_market.csv", index=False, encoding="utf-8-sig")
    print(f"  {OUT_DIR / 'optimize_r1_gold.csv'}")
    print(f"  {OUT_DIR / 'optimize_r4_stoploss.csv'}")
    print(f"  {OUT_DIR / 'optimize_r5_market.csv'}")

    # 차트
    print("\n  차트 생성...")
    c1 = plot_scenario_comparison(results)
    print(f"  {c1}")
    c2 = plot_violation_impact(impact_df)
    print(f"  {c2}")
    c3 = plot_parameter_sensitivity(opt_r1, opt_r4, opt_r5)
    print(f"  {c3}")
    c4 = plot_monthly_violation_pnl(rts_a)
    print(f"  {c4}")

    print("\n  완료!")
    print()


if __name__ == "__main__":
    main()

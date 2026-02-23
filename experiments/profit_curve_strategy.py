"""
profit_curve_strategy.py
=========================
수익_거래내역.csv 기반 전략 분석 & 백테스트

Flow:
  1. OHLCV fetch  — yfinance로 2023-09-01부터 전 종목 수집 → data/market/daily/profit_curve_ohlcv.parquet
  2. Equity curve — 실제 수익 거래 재현 → 누적 손익 곡선
  3. Entry analysis — 각 매수 시점의 가격 컨텍스트 (MA, RSI, 낙폭)
  4. Strategy code — 분석 기반 DCA 진입/청산 규칙 정의 & 백테스트

사용법:
    pyenv shell ptj_stock_lab && python experiments/profit_curve_strategy.py [--step 1|2|3|4|all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import warnings
from collections import deque
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
HISTORY_DIR   = ROOT / "history"
DATA_DIR      = ROOT / "data" / "market" / "daily"
OHLCV_PATH    = DATA_DIR / "profit_curve_ohlcv.parquet"
TICKER_MAP    = ROOT / "data" / "meta" / "ticker_mapping.json"
PROFIT_CSV    = HISTORY_DIR / "수익_거래내역.csv"
LOSS_CSV      = HISTORY_DIR / "손해_거래내역.csv"
ALL_TRADES    = HISTORY_DIR / "거래내역_20231006_20260212.csv"

# ── 분석 대상 ticker ──────────────────────────────────────────────────────────
# 수익 합계 기준 TOP 종목 (수익 CSV에서 확인된 순서)
STRATEGY_TICKERS = ["MSTU", "CONL", "ROBN", "PTIR", "NVDL", "AMDL", "IREN", "XXRP"]
FETCH_START  = "2023-09-01"   # MSTU/CONL 상장 이후

# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def load_ticker_map() -> dict[str, str]:
    """ISIN → yfinance ticker"""
    with open(TICKER_MAP) as f:
        raw = json.load(f)
    return {v["isin"]: v["yf"] for v in raw.values() if v.get("isin") and v.get("yf")}


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — OHLCV 수집 (yfinance)
# ═══════════════════════════════════════════════════════════════════════════════
def step1_fetch_ohlcv():
    """
    yfinance로 STRATEGY_TICKERS 전체를 FETCH_START부터 오늘까지 다운로드.
    결과를 profit_curve_ohlcv.parquet에 저장.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance 미설치: pip install yfinance")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbols_str = " ".join(STRATEGY_TICKERS)
    print(f"[STEP 1] yfinance 다운로드: {STRATEGY_TICKERS}")
    print(f"  기간: {FETCH_START} ~ today")

    raw = yf.download(symbols_str, start=FETCH_START, auto_adjust=True, progress=False)
    # MultiIndex → long format
    frames = []
    for ticker in STRATEGY_TICKERS:
        if ticker not in raw["Close"].columns:
            print(f"  ⚠  {ticker}: 데이터 없음 (상장일 확인 필요)")
            continue
        df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        df["ticker"] = ticker
        df = df.reset_index()
        frames.append(df)
        print(f"  ✓  {ticker}: {len(df)}일 ({df['Date'].min().date()} ~ {df['Date'].max().date()})")

    if not frames:
        print("  데이터 없음")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(OHLCV_PATH, index=False)
    print(f"\n[STEP 1] 저장 완료 → {OHLCV_PATH}  ({len(merged)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 실제 수익 곡선 재현 (OHLCV 불필요)
# ═══════════════════════════════════════════════════════════════════════════════
def step2_equity_curve():
    """
    수익_거래내역.csv + 손해_거래내역.csv로 시간순 누적 손익 곡선 출력.
    ticker별 기여도 포함.
    """
    print("[STEP 2] 실제 수익 곡선 재현")

    profit = pd.read_csv(PROFIT_CSV, encoding="utf-8-sig")
    loss   = pd.read_csv(LOSS_CSV,   encoding="utf-8-sig")
    isin2t = load_ticker_map()

    for df in [profit, loss]:
        df["판매일자"] = pd.to_datetime(df["판매일자"])
        df["최초매수일"] = pd.to_datetime(df["최초매수일"])
        df["보유기간_일"] = (df["판매일자"] - df["최초매수일"]).dt.days
        df["ticker"] = df["종목코드"].map(isin2t)

    all_pnl = pd.concat([profit, loss], ignore_index=True)
    all_pnl = all_pnl.sort_values("판매일자").reset_index(drop=True)
    all_pnl["누적손익_원"] = all_pnl["실현손익_원"].cumsum()

    # ── 월별 집계 ──
    all_pnl["월"] = all_pnl["판매일자"].dt.to_period("M")
    monthly = (
        all_pnl.groupby("월")["실현손익_원"]
        .agg(건수="count", 손익합="sum")
        .assign(누적=lambda x: x["손익합"].cumsum())
    )
    print("\n[월별 실현 손익]")
    print(monthly.to_string())

    # ── ticker별 기여도 ──
    ticker_contrib = (
        all_pnl.groupby("ticker")["실현손익_원"]
        .agg(건수="count", 총손익="sum")
        .sort_values("총손익", ascending=False)
    )
    print("\n[ticker별 기여도]")
    print(ticker_contrib.to_string())

    # ── 핵심 통계 ──
    print(f"\n[핵심 통계]")
    print(f"  총 실현 손익  : {all_pnl['실현손익_원'].sum():>15,.0f} 원")
    print(f"  수익 건수     : {(all_pnl['실현손익_원'] > 0).sum()}")
    print(f"  손해 건수     : {(all_pnl['실현손익_원'] < 0).sum()}")
    print(f"  평균 보유기간  : {all_pnl['보유기간_일'].mean():.1f}일")
    print(f"  평균 손익률    : {profit['손익률_%'].mean():.1f}% (수익 거래)")
    print(f"  최대 단일 수익 : {all_pnl['실현손익_원'].max():>15,.0f} 원")
    print(f"  최대 단일 손해 : {all_pnl['실현손익_원'].min():>15,.0f} 원")

    return all_pnl


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 진입 조건 분석 (OHLCV 필요: step1 먼저 실행)
# ═══════════════════════════════════════════════════════════════════════════════
def step3_entry_analysis():
    """
    실제 매수 거래 vs OHLCV → 진입 당시 MA·RSI·낙폭 분포 분석.
    수익 판매로 이어진 매수 vs 손해 판매로 이어진 매수 비교.
    """
    print("[STEP 3] 진입 조건 분석")

    if not OHLCV_PATH.exists():
        print(f"  OHLCV 없음 → step1 먼저 실행: python {__file__} --step 1")
        return None

    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])

    # MA·RSI 계산
    records = []
    for ticker, grp in ohlcv.groupby("ticker"):
        g = grp.sort_values("Date").set_index("Date")
        g["ma5"]  = g["Close"].rolling(5).mean()
        g["ma20"] = g["Close"].rolling(20).mean()
        g["ma60"] = g["Close"].rolling(60).mean()
        g["rsi14"] = rsi(g["Close"], 14)
        g["pct_from_ma20"] = (g["Close"] - g["ma20"]) / g["ma20"] * 100
        g["pct_from_ma60"] = (g["Close"] - g["ma60"]) / g["ma60"] * 100
        # 20일 내 최고가 대비 낙폭
        g["roll_high20"] = g["High"].rolling(20).max()
        g["drawdown20"]  = (g["Close"] - g["roll_high20"]) / g["roll_high20"] * 100
        g["ticker"] = ticker
        g = g.reset_index()
        records.append(g)

    ohlcv_enriched = pd.concat(records, ignore_index=True)

    # 전체 매수 내역 로드
    all_trades = pd.read_csv(ALL_TRADES, encoding="utf-8-sig")
    all_trades["거래일자"] = pd.to_datetime(all_trades["거래일자"])
    isin2t = load_ticker_map()
    all_trades["ticker"] = all_trades["종목코드"].map(isin2t)
    buys = all_trades[all_trades["거래구분"] == "구매"].copy()

    # 수익 판매와 연결: 최초매수일 기준 매칭
    profit = pd.read_csv(PROFIT_CSV, encoding="utf-8-sig")
    loss   = pd.read_csv(LOSS_CSV,   encoding="utf-8-sig")
    profit["최초매수일"] = pd.to_datetime(profit["최초매수일"])
    loss["최초매수일"]   = pd.to_datetime(loss["최초매수일"])
    profit["ticker"]     = profit["종목코드"].map(isin2t)
    loss["ticker"]       = loss["종목코드"].map(isin2t)

    profit_keys = set(zip(profit["ticker"], profit["최초매수일"].dt.date))
    loss_keys   = set(zip(loss["ticker"],   loss["최초매수일"].dt.date))

    def classify(row):
        key = (row["ticker"], row["거래일자"].date())
        if key in profit_keys:
            return "profit"
        if key in loss_keys:
            return "loss"
        return "unknown"

    buys["outcome"] = buys.apply(classify, axis=1)

    # OHLCV와 조인
    merged = buys.merge(
        ohlcv_enriched[["Date", "ticker", "Close", "ma5", "ma20", "ma60",
                         "rsi14", "pct_from_ma20", "pct_from_ma60", "drawdown20"]],
        left_on=["거래일자", "ticker"],
        right_on=["Date", "ticker"],
        how="left",
    )

    has_ohlcv = merged[merged["Close"].notna()]
    print(f"  OHLCV 매칭된 매수: {len(has_ohlcv)}건 / 전체 매수 {len(buys)}건")
    if len(has_ohlcv) == 0:
        print("  매칭 없음 — OHLCV 기간이 매수 날짜를 커버하지 않음")
        return None

    print("\n[진입 시 조건별 평균값]")
    cols = ["rsi14", "pct_from_ma20", "pct_from_ma60", "drawdown20"]
    summary = has_ohlcv.groupby("outcome")[cols].mean().round(2)
    print(summary.to_string())

    print("\n[RSI 분포 (수익 vs 손해 첫 진입)]")
    for outcome in ["profit", "loss"]:
        sub = has_ohlcv[has_ohlcv["outcome"] == outcome]["rsi14"].dropna()
        if len(sub) > 0:
            print(f"  {outcome}: n={len(sub)}, "
                  f"median={sub.median():.1f}, "
                  f"25th={sub.quantile(0.25):.1f}, "
                  f"75th={sub.quantile(0.75):.1f}")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DCA 전략 백테스트 (OHLCV 필요)
# ═══════════════════════════════════════════════════════════════════════════════
class MomentumStrategy:
    """
    Step 3 분석 결과 기반 모멘텀 추세추종 전략.

    [핵심 인사이트]
    수익 진입 패턴:  RSI ≈ 67 (중앙값), 가격 MA20 대비 +18.7%  → 상승 추세 편승
    손해 진입 패턴:  RSI ≈ 20 (중앙값), 가격 MA20 대비 -13.3%  → 역발상 (하락 지속)

    [전략 규칙]
    진입  — MA20 위에 있고 (pct_from_ma20 >= entry_above_pct)
             RSI14 >= rsi_entry 이면 추세 확인 후 매수
    추가  — 추세 지속 중 pct 상승할 때마다 추가 매수 (피라미딩, max_add회)
    청산  — 평균단가 대비 target_pct% 이상 수익 or stop_pct% 이하 손실 or hold_days 경과
    """

    def __init__(
        self,
        ticker: str,
        entry_above_pct: float = 0.0,    # MA20 위 최소 % (0 = MA20 위면 진입)
        rsi_entry: float       = 55.0,   # 진입 최소 RSI
        add_pct: float         = 5.0,    # 피라미딩 간격 (직전 매수 대비 % 상승)
        max_add: int           = 2,      # 최대 추가 매수 횟수
        target_pct: float      = 15.0,   # 목표 수익률 (%)
        stop_pct: float        = -20.0,  # 손절 기준 (%)
        hold_days: int         = 45,     # 최대 보유일
        unit_usd: float        = 740.0,  # 1회 매수 금액 (USD, ≈ 1,000,000원 / 1350)
        fx: float              = 1350.0, # KRW/USD 환율 (고정)
    ):
        self.ticker          = ticker
        self.entry_above_pct = entry_above_pct
        self.rsi_entry       = rsi_entry
        self.add_pct         = add_pct
        self.max_add         = max_add
        self.target_pct      = target_pct
        self.stop_pct        = stop_pct
        self.hold_days       = hold_days
        self.unit_usd        = unit_usd
        self.fx              = fx

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        ohlcv: Date, Open, High, Low, Close 컬럼 포함 DataFrame (단일 ticker, USD 가격).
        returns: 거래 기록 DataFrame.
        """
        df = ohlcv.copy().sort_values("Date").reset_index(drop=True)
        df["ma20"]   = df["Close"].rolling(20).mean()
        df["rsi14"]  = rsi(df["Close"], 14)

        trades = []
        position = []   # [(entry_date, qty_shares, entry_price_usd)]
        add_count = 0
        last_buy_price = None

        for _, row in df.iterrows():
            if pd.isna(row["ma20"]) or pd.isna(row["rsi14"]):
                continue

            price = row["Close"]
            d     = row["Date"]
            pct_from_ma20 = (price - row["ma20"]) / row["ma20"] * 100
            r14           = row["rsi14"]
            has_position  = len(position) > 0

            # ── 청산 ──
            if has_position:
                total_qty  = sum(p[1] for p in position)
                total_cost = sum(p[1] * p[2] for p in position)
                avg_cost   = total_cost / total_qty
                pnl_pct    = (price - avg_cost) / avg_cost * 100
                held_days  = (d - position[0][0]).days

                should_exit = (
                    pnl_pct >= self.target_pct
                    or pnl_pct <= self.stop_pct
                    or held_days >= self.hold_days
                    or pct_from_ma20 < -15.0    # 추세 붕괴 (MA20 -15% 이탈)
                )
                if should_exit:
                    pnl_usd = total_qty * price - total_cost
                    trades.append({
                        "Date": d, "Action": "SELL", "ticker": self.ticker,
                        "Price_USD": price,
                        "Qty": total_qty,
                        "PnL_USD": pnl_usd,
                        "PnL_KRW": round(pnl_usd * self.fx),
                        "PnL_pct": round(pnl_pct, 2),
                        "HeldDays": held_days,
                        "ExitReason": (
                            "TARGET"  if pnl_pct >= self.target_pct   else
                            "TREND_BREAK" if pct_from_ma20 < -15.0    else
                            "STOP"    if pnl_pct <= self.stop_pct      else
                            "TIME"
                        ),
                    })
                    position, add_count, last_buy_price = [], 0, None
                    continue

            # ── 신규 진입 (추세 확인) ──
            if not has_position:
                if pct_from_ma20 >= self.entry_above_pct and r14 >= self.rsi_entry:
                    qty = self.unit_usd / price
                    position.append((d, qty, price))
                    last_buy_price = price
                    add_count = 0
                    trades.append({
                        "Date": d, "Action": "BUY", "ticker": self.ticker,
                        "Price_USD": price, "Qty": qty,
                        "PnL_USD": None, "PnL_KRW": None,
                        "PnL_pct": None, "HeldDays": None, "ExitReason": "ENTRY",
                    })

            # ── 피라미딩 추가 매수 (추세 지속 시) ──
            elif add_count < self.max_add and last_buy_price is not None:
                rise_from_last = (price - last_buy_price) / last_buy_price * 100
                if rise_from_last >= self.add_pct and r14 >= self.rsi_entry:
                    qty = self.unit_usd / price
                    position.append((d, qty, price))
                    last_buy_price = price
                    add_count += 1
                    trades.append({
                        "Date": d, "Action": "BUY", "ticker": self.ticker,
                        "Price_USD": price, "Qty": qty,
                        "PnL_USD": None, "PnL_KRW": None,
                        "PnL_pct": None, "HeldDays": None,
                        "ExitReason": f"ADD{add_count}",
                    })

        return pd.DataFrame(trades)


def step4_backtest():
    """
    DCAStrategy를 전체 STRATEGY_TICKERS에 적용, 합산 equity curve 출력.
    """
    print("[STEP 4] DCA 전략 백테스트")

    if not OHLCV_PATH.exists():
        print(f"  OHLCV 없음 → step1 먼저 실행: python {__file__} --step 1")
        return

    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])

    all_trades = []
    for ticker in STRATEGY_TICKERS:
        sub = ohlcv[ohlcv["ticker"] == ticker].copy()
        if len(sub) < 30:
            print(f"  {ticker}: 데이터 부족 ({len(sub)}일) — 스킵")
            continue

        strategy = MomentumStrategy(
            ticker          = ticker,
            entry_above_pct = 0.0,    # MA20 위이면 진입
            rsi_entry       = 55.0,
            add_pct         = 5.0,
            max_add         = 2,
            target_pct      = 15.0,
            stop_pct        = -20.0,
            hold_days       = 45,
            unit_usd        = 740.0,  # ≈ 1,000,000원
            fx              = 1350.0,
        )
        result = strategy.run(sub)
        if len(result) > 0:
            all_trades.append(result)
            sells = result[result["Action"] == "SELL"]
            if len(sells) > 0:
                print(f"  {ticker}: 매매 {len(result)}건 | "
                      f"수익 {(sells['PnL_KRW'] > 0).sum()}/{len(sells)} | "
                      f"총PnL {sells['PnL_KRW'].sum():,.0f} 원")

    if not all_trades:
        print("  백테스트 결과 없음")
        return

    merged = pd.concat(all_trades, ignore_index=True)
    sells  = merged[merged["Action"] == "SELL"].sort_values("Date")
    sells  = sells.reset_index(drop=True)
    sells["누적PnL_원"] = sells["PnL_KRW"].cumsum()

    print(f"\n[백테스트 요약]")
    print(f"  기간           : {sells['Date'].min().date()} ~ {sells['Date'].max().date()}")
    print(f"  총 매도 건수    : {len(sells)}")
    print(f"  수익 건수       : {(sells['PnL_KRW'] > 0).sum()}")
    print(f"  손해 건수       : {(sells['PnL_KRW'] < 0).sum()}")
    print(f"  총 실현 PnL     : {sells['PnL_KRW'].sum():>15,.0f} 원")
    print(f"  평균 보유기간    : {sells['HeldDays'].mean():.1f}일")
    print(f"  평균 손익률      : {sells['PnL_pct'].mean():.1f}%")

    print(f"\n[청산 사유 분포]")
    print(sells["ExitReason"].value_counts().to_string())

    return sells


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="profit_curve_strategy")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="실행할 단계 (기본: all)",
    )
    args = parser.parse_args()

    if args.step in ("1", "all"):
        step1_fetch_ohlcv()
        print()
    if args.step in ("2", "all"):
        step2_equity_curve()
        print()
    if args.step in ("3", "all"):
        step3_entry_analysis()
        print()
    if args.step in ("4", "all"):
        step4_backtest()


if __name__ == "__main__":
    main()

"""
Polymarket 기반 급락 감지 모델 (Crash Detection Model)
========================================================
훈련 기간: 2024-02-01 ~ 2025-12-31
타겟 기간: 2026-01-01 ~ 2026-02-17

신호 소스:
  - Polymarket: BTC daily Up/Down resolution, weekly reach/dip, NDX direction
  - Market Data: VIX, BTC return, QQQ/SOXL/TQQQ/MSTZ returns

전략:
  - 정상 (crash_score < 0.30): SOXL 100%   (3x 레버리지 롱)
  - 경계 (0.30 ~ 0.55):        TQQQ 100%   (1x~3x 나스닥)
  - 헤지 (0.55 ~ 0.75):        TQQQ 50% + MSTZ 50%
  - 급락 (>= 0.75):             MSTZ 100%   (인버스 공격 포지션)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
POLY_DIRS = [
    ROOT / "data/polymarket/2024",
    ROOT / "data/polymarket/2025",
    ROOT / "data/polymarket/2026",
]
MARKET_EXTRA = ROOT / "data/market/daily/extra_signals_daily.parquet"
MARKET_CRYPTO = ROOT / "data/market/daily/crypto_daily.parquet"
MARKET_VIX = ROOT / "data/market/daily/vix_daily.parquet"

TRAIN_START = "2024-02-01"
TRAIN_END = "2025-12-31"
TARGET_START = "2026-01-01"
TARGET_END = "2026-02-17"

# 전략 임계값
THRESHOLDS = {
    "normal": 0.30,   # < 0.30 → SOXL
    "hedge": 0.55,    # 0.30~0.55 → TQQQ
    "crash": 0.75,    # 0.55~0.75 → TQQQ 50%+MSTZ 50%
                      # >= 0.75 → MSTZ 100%
}


# ─────────────────────────────────────────────
# 1. Polymarket 신호 추출
# ─────────────────────────────────────────────

def load_poly_signals() -> pd.DataFrame:
    """모든 Polymarket JSON 파일에서 일별 신호 추출."""
    rows = []

    for poly_dir in POLY_DIRS:
        for fpath in sorted(poly_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)

            date_str = data.get("date", fpath.stem[:10])
            ind = data.get("indicators", {})
            row: dict = {"date": date_str}

            # BTC Up/Down resolution
            btc_fp = ind.get("btc_up_down", {}).get("final_prices", {})
            if btc_fp and "Up" in btc_fp:
                row["btc_up"] = float(btc_fp["Up"])

            # NDX Up/Down
            ndx_fp = ind.get("ndx_up_down", {}).get("final_prices", {})
            if ndx_fp and "Up" in ndx_fp:
                row["ndx_up"] = float(ndx_fp["Up"])

            # Weekly reach/dip (opening prob from time series)
            wkly_mkts = ind.get("btc_weekly", {}).get("markets", [])
            reach_probs, dip_probs = [], []
            for m in wkly_mkts:
                q = m.get("question", "").lower()
                for ok, ov in m.get("outcomes", {}).items():
                    if ov and ok.lower() == "yes":
                        p = ov[0]["p"]
                        if "reach" in q:
                            reach_probs.append(p)
                        elif "dip" in q:
                            dip_probs.append(p)

            if reach_probs or dip_probs:
                row["weekly_reach"] = max(reach_probs) if reach_probs else 0.0
                row["weekly_dip"] = max(dip_probs) if dip_probs else 0.0

            # Fed decision (cut → positive, hike → negative)
            fed_mkts = ind.get("fed_decision", {}).get("markets", [])
            fed_cut_prob = 0.0
            fed_hike_prob = 0.0
            for m in fed_mkts:
                q = m.get("question", "").lower()
                fp_fed = m.get("final_prices")
                if fp_fed:
                    yes_p = float(fp_fed.get("Yes", 0))
                    if "50" in q and ("cut" in q or "decreas" in q or "인하" in q):
                        fed_cut_prob = max(fed_cut_prob, yes_p)
                    elif "25" in q and ("cut" in q or "decreas" in q or "인하" in q):
                        fed_cut_prob = max(fed_cut_prob, yes_p * 0.5)
                    elif "increas" in q or "hike" in q or "인상" in q:
                        fed_hike_prob = max(fed_hike_prob, yes_p)
                else:
                    for ok, ov in m.get("outcomes", {}).items():
                        if ov:
                            p_val = ov[0]["p"]
                            lbl = ok.lower()
                            if "50" in lbl and ("cut" in q or "decreas" in q):
                                fed_cut_prob = max(fed_cut_prob, p_val)
                            elif "25" in lbl and ("cut" in q or "decreas" in q):
                                fed_cut_prob = max(fed_cut_prob, p_val * 0.5)
                            elif "hike" in lbl or "increas" in lbl:
                                fed_hike_prob = max(fed_hike_prob, p_val)

            row["fed_cut_prob"] = fed_cut_prob
            row["fed_hike_prob"] = fed_hike_prob

            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # 중복 날짜 제거 (마지막 값 우선)
    df = df[~df.index.duplicated(keep="last")]
    return df


# ─────────────────────────────────────────────
# 2. 시장 데이터 로드
# ─────────────────────────────────────────────

def load_market_data() -> pd.DataFrame:
    """BTC, VIX, SOXL, TQQQ, MSTZ, QQQ 일별 데이터 로드."""
    # Extra signals (SOXL, TQQQ, MSTZ, QQQ, VIX 포함)
    extra = pd.read_parquet(MARKET_EXTRA)
    extra["Date"] = pd.to_datetime(extra["Date"])

    tickers_needed = ["SOXL", "TQQQ", "MSTZ", "QQQ", "VIX"]
    dfs = {}
    for t in tickers_needed:
        sub = extra[extra["ticker"] == t][["Date", "Close"]].copy()
        sub = sub.set_index("Date").rename(columns={"Close": t})
        if not sub.empty:
            dfs[t] = sub

    # Crypto (BTC)
    crypto = pd.read_parquet(MARKET_CRYPTO)
    crypto["date"] = pd.to_datetime(crypto["date"])
    btc = crypto[crypto["symbol"] == "BTC"][["date", "close"]].copy()
    btc = btc.set_index("date").rename(columns={"close": "BTC"})
    dfs["BTC"] = btc

    # VIX 별도 파일 (extra에 없으면 사용)
    if "VIX" not in dfs or dfs["VIX"].empty:
        vix_df = pd.read_parquet(MARKET_VIX)
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix = vix_df[vix_df["symbol"] == "VIX"][["date", "close"]].copy()
        vix = vix.set_index("date").rename(columns={"close": "VIX"})
        dfs["VIX"] = vix

    # 합치기
    market = pd.concat(dfs.values(), axis=1).sort_index()

    # 수익률 계산
    for col in ["SOXL", "TQQQ", "MSTZ", "QQQ", "BTC"]:
        if col in market.columns:
            market[f"{col}_ret"] = market[col].pct_change()

    # VIX 정규화 (0~1) 및 변화율
    if "VIX" in market.columns:
        market["VIX_norm"] = (market["VIX"] - 12) / (80 - 12)
        market["VIX_norm"] = market["VIX_norm"].clip(0, 1)
        market["VIX_chg"] = market["VIX"].pct_change()

    return market


# ─────────────────────────────────────────────
# 3. Crash Score 계산
# ─────────────────────────────────────────────

def compute_crash_score(merged: pd.DataFrame) -> pd.DataFrame:
    """Polymarket + Market 신호를 결합하여 crash_score 계산.

    crash_score 구성 요소:
      - poly_btc_bear:   최근 3일 BTC down 비율         (가중치 0.30)
      - poly_weekly_dip: 주간 dip > reach 신호           (가중치 0.25)
      - vix_signal:      VIX 급등 정규화                  (가중치 0.25)
      - mkt_momentum:    QQQ 5일 수익률 음수              (가중치 0.20)
    """
    df = merged.copy()

    # ① Polymarket BTC 하락 신호 (3일 rolling)
    btc_down = (1 - df["btc_up"].fillna(0.5))  # 0=UP→0, 1=DOWN→1, NaN→0.5
    df["poly_btc_bear"] = btc_down.rolling(3, min_periods=1).mean()

    # ② 주간 dip > reach (weekly Polymarket)
    df["poly_weekly_dip_bias"] = 0.0
    mask = df["weekly_reach"].notna() & df["weekly_dip"].notna()
    df.loc[mask, "poly_weekly_dip_bias"] = (
        (df.loc[mask, "weekly_dip"] - df.loc[mask, "weekly_reach"]).clip(0, 1)
    )
    # Forward-fill (weekly → 해당 주 내내 유지)
    df["poly_weekly_dip_bias"] = df["poly_weekly_dip_bias"].replace(0, np.nan).ffill(limit=7).fillna(0)

    # ③ VIX 신호
    df["vix_signal"] = df.get("VIX_norm", pd.Series(0.3, index=df.index)).fillna(0.3)

    # ④ SOXL 5일 모멘텀 (음수 = 하락 압력) — 핵심 자산 직접 추적
    if "SOXL_ret" in df.columns:
        soxl_mom = -df["SOXL_ret"].rolling(5, min_periods=1).mean()
        df["mkt_momentum"] = soxl_mom.clip(0, 0.08) / 0.08
    elif "QQQ_ret" in df.columns:
        qqq_mom = -df["QQQ_ret"].rolling(5, min_periods=1).mean()
        df["mkt_momentum"] = qqq_mom.clip(0, 0.05) / 0.05
    else:
        df["mkt_momentum"] = 0.0

    # ⑤ VIX 당일 급등 (갑작스러운 변동성 폭발)
    if "VIX_chg" in df.columns:
        vix_spike = df["VIX_chg"].clip(0, 0.5) / 0.5
        df["vix_spike"] = vix_spike.fillna(0)
    else:
        df["vix_spike"] = 0.0

    # ⑥ BTC 실제 수익률 (Polymarket과 별도 — SOXL과의 공동 하락 감지)
    if "BTC_ret" in df.columns:
        btc_drop = -df["BTC_ret"].rolling(2, min_periods=1).mean()
        df["btc_ret_signal"] = btc_drop.clip(0, 0.05) / 0.05
    else:
        df["btc_ret_signal"] = 0.0

    # Crash Score 종합 (가중합)
    # poly_btc_bear: Polymarket 판단 (3일 BTC down)
    # vix_signal: VIX 레벨 (만성 위험)
    # mkt_momentum: SOXL 5일 모멘텀 하락
    # vix_spike: VIX 당일 급등 (즉각 반응)
    # btc_ret_signal: BTC 실제 하락 (SOXL과 co-move)
    # poly_weekly_dip_bias: 주간 편향
    df["crash_score"] = (
        df["poly_btc_bear"] * 0.20
        + df["vix_signal"] * 0.20
        + df["mkt_momentum"] * 0.25
        + df["vix_spike"] * 0.15
        + df["btc_ret_signal"] * 0.15
        + df["poly_weekly_dip_bias"] * 0.05
    )

    # Fed hawkish 패널티
    fed_hawk = df.get("fed_hike_prob", pd.Series(0.0, index=df.index)).fillna(0)
    df["crash_score"] = (df["crash_score"] + fed_hawk * 0.1).clip(0, 1)

    return df


# ─────────────────────────────────────────────
# 4. 포지션 결정 — 연속 사이즈 조절
# ─────────────────────────────────────────────
#
# crash_score 0.0 → SOXL 100%  (최대 공격 롱)
# crash_score 0.4 → SOXL 33% + TQQQ 67% (롱 축소)
# crash_score 0.6 → TQQQ 50% + MSTZ 50% (전환)
# crash_score 1.0 → MSTZ 100% (최대 공격 숏/인버스)
#
# 수식:
#   soxl_w  = clip(1 - score/0.5, 0, 1)     → score 0~0.5 구간에서 1→0
#   mstz_w  = clip((score-0.5)/0.5, 0, 1)   → score 0.5~1.0 구간에서 0→1
#   tqqq_w  = 1 - soxl_w - mstz_w           → 전환 구간 완충재

def assign_position(crash_score: float) -> dict:
    """crash_score → 연속 포지션 비중 결정 (사이즈 조절 방식).

    항상 공격적 — 현금 없음. crash_score에 따라
    SOXL(롱) ↔ TQQQ(중립) ↔ MSTZ(인버스) 연속 전환.

    Returns:
        dict: {ticker: weight}  합계 = 1.0
    """
    soxl_w = float(np.clip(1.0 - crash_score / 0.5, 0.0, 1.0))
    mstz_w = float(np.clip((crash_score - 0.5) / 0.5, 0.0, 1.0))
    tqqq_w = max(0.0, 1.0 - soxl_w - mstz_w)

    result = {}
    if soxl_w > 0.001:
        result["SOXL"] = round(soxl_w, 4)
    if tqqq_w > 0.001:
        result["TQQQ"] = round(tqqq_w, 4)
    if mstz_w > 0.001:
        result["MSTZ"] = round(mstz_w, 4)
    return result if result else {"SOXL": 1.0}


def mode_label(crash_score: float) -> str:
    if crash_score < 0.15:
        return "BULL"      # SOXL 100%
    elif crash_score < 0.40:
        return "LONG-"     # SOXL 축소
    elif crash_score < 0.60:
        return "NEUTRAL"   # TQQQ 중심
    elif crash_score < 0.80:
        return "SHORT+"    # MSTZ 진입
    else:
        return "CRASH"     # MSTZ 100%


# ─────────────────────────────────────────────
# 5. 백테스트
# ─────────────────────────────────────────────

def backtest(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """crash_score 기반 포지션 전략 백테스트.

    신호는 전날(t-1) crash_score → 오늘(t) 포지션 적용 (lookahead 방지).
    """
    period = df.loc[start:end].copy()
    ret_cols = {"SOXL": "SOXL_ret", "TQQQ": "TQQQ_ret", "MSTZ": "MSTZ_ret"}

    portfolio_ret = []
    positions = []
    modes = []

    for i, (dt, row) in enumerate(period.iterrows()):
        if i == 0:
            # 첫날은 신호 없음 → 기본 SOXL
            score = 0.0
        else:
            prev_dt = period.index[i - 1]
            raw = period.loc[prev_dt, "crash_score"]
            score = float(raw) if not hasattr(raw, "__len__") else float(raw.iloc[-1])

        pos = assign_position(score)
        mode = mode_label(score)

        # 포트폴리오 수익률
        p_ret = 0.0
        for ticker, weight in pos.items():
            col = ret_cols.get(ticker)
            if col and not pd.isna(row.get(col, np.nan)):
                p_ret += weight * row[col]
            else:
                p_ret += 0.0  # 데이터 없으면 수익률 0

        portfolio_ret.append(p_ret)
        positions.append(pos)
        modes.append(mode)

    period = period.copy()
    period["strategy_ret"] = portfolio_ret
    period["mode"] = modes
    period["position"] = [str(p) for p in positions]

    # 누적 수익
    period["strategy_cum"] = (1 + period["strategy_ret"]).cumprod()

    # 벤치마크: SOXL 단순 보유
    period["soxl_cum"] = (1 + period["SOXL_ret"].fillna(0)).cumprod()
    period["tqqq_cum"] = (1 + period["TQQQ_ret"].fillna(0)).cumprod()

    return period


# ─────────────────────────────────────────────
# 6. 성과 분석
# ─────────────────────────────────────────────

def calc_metrics(returns: pd.Series, label: str) -> dict:
    """주요 성과 지표 계산."""
    clean = returns.dropna()
    total_ret = (1 + clean).prod() - 1
    annual_ret = (1 + total_ret) ** (252 / len(clean)) - 1
    vol = clean.std() * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0

    # MDD
    cum = (1 + clean).cumprod()
    drawdown = cum / cum.cummax() - 1
    mdd = drawdown.min()

    # 승률
    win_rate = (clean > 0).mean()

    return {
        "label": label,
        "total_return": f"{total_ret:.1%}",
        "annual_return": f"{annual_ret:.1%}",
        "volatility": f"{vol:.1%}",
        "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{mdd:.1%}",
        "win_rate": f"{win_rate:.1%}",
        "n_days": len(clean),
    }


def print_report(train_result: pd.DataFrame, target_result: pd.DataFrame) -> None:
    """결과 리포트 출력."""
    print("\n" + "=" * 70)
    print("Polymarket 급락 감지 모델 — 성과 리포트")
    print("=" * 70)

    # 훈련 기간
    print(f"\n【훈련 기간: {TRAIN_START} ~ {TRAIN_END}】")
    train_metrics = [
        calc_metrics(train_result["strategy_ret"], "전략 (Crash Model)"),
        calc_metrics(train_result["SOXL_ret"].fillna(0), "SOXL 보유"),
        calc_metrics(train_result["TQQQ_ret"].fillna(0), "TQQQ 보유"),
    ]
    _print_metrics_table(train_metrics)

    # 모드 분포
    print("\n훈련 기간 포지션 분포:")
    mode_dist = train_result["mode"].value_counts().sort_index()
    for mode, cnt in mode_dist.items():
        pct = cnt / len(train_result) * 100
        print(f"  {mode:8s}: {cnt:3d}일 ({pct:.0f}%)")

    # 평균 비중
    soxl_avg = train_result.apply(lambda r: assign_position(r["crash_score"]).get("SOXL", 0), axis=1).mean()
    tqqq_avg = train_result.apply(lambda r: assign_position(r["crash_score"]).get("TQQQ", 0), axis=1).mean()
    mstz_avg = train_result.apply(lambda r: assign_position(r["crash_score"]).get("MSTZ", 0), axis=1).mean()
    print(f"\n  평균 비중 → SOXL: {soxl_avg:.1%}  TQQQ: {tqqq_avg:.1%}  MSTZ: {mstz_avg:.1%}")

    # 타겟 기간
    print(f"\n【타겟 기간: {TARGET_START} ~ {TARGET_END}】")
    tgt_metrics = [
        calc_metrics(target_result["strategy_ret"], "전략 (Crash Model)"),
        calc_metrics(target_result["SOXL_ret"].fillna(0), "SOXL 보유"),
        calc_metrics(target_result["TQQQ_ret"].fillna(0), "TQQQ 보유"),
    ]
    _print_metrics_table(tgt_metrics)

    # 타겟 기간 일별 시그널
    print(f"\n타겟 기간 일별 포지션 (연속 사이즈 조절):")
    print(f"{'날짜':12s} {'모드':8s} {'score':6s} {'SOXL':6s} {'TQQQ':6s} {'MSTZ':6s} {'전략':8s} {'SOXL':8s} {'누적':8s}")
    print("-" * 78)
    cum = 1.0
    for dt, row in target_result.iterrows():
        cum *= (1 + row["strategy_ret"])
        pos = assign_position(row["crash_score"])
        soxl_w = pos.get("SOXL", 0)
        tqqq_w = pos.get("TQQQ", 0)
        mstz_w = pos.get("MSTZ", 0)
        print(
            f"{str(dt.date()):12s} "
            f"{row['mode']:8s} "
            f"{row['crash_score']:5.2f} "
            f"{soxl_w:5.0%} "
            f"{tqqq_w:5.0%} "
            f"{mstz_w:5.0%} "
            f"{row['strategy_ret']:+7.2%} "
            f"{row.get('SOXL_ret', 0):+7.2%} "
            f"{cum - 1:+7.2%}"
        )

    # Crash Score 구성요소 분석
    print(f"\n【Crash Score 구성요소 (타겟 기간 평균)】")
    score_cols = ["poly_btc_bear", "poly_weekly_dip_bias", "vix_signal",
                  "mkt_momentum", "vix_spike", "btc_ret_signal"]
    for col in score_cols:
        if col in target_result.columns:
            print(f"  {col:25s}: {target_result[col].mean():.3f}")

    # 급락 미감지 분석
    print(f"\n【급락 구간 분석 (SOXL < -5%)】")
    crash_days = target_result[target_result["SOXL_ret"] < -0.05]
    print(f"{'날짜':12s} {'SOXL수익':10s} {'전날score':10s} {'모드':6s}")
    print("-" * 45)
    for dt, row in crash_days.iterrows():
        idx = list(target_result.index).index(dt)
        prev_score = target_result.iloc[idx - 1]["crash_score"] if idx > 0 else 0
        prev_mode = target_result.iloc[idx - 1]["mode"] if idx > 0 else "-"
        print(f"{str(dt.date()):12s} {row['SOXL_ret']:+9.2%}  {prev_score:.3f}     {prev_mode}")


def _print_metrics_table(metrics_list: list) -> None:
    """성과 테이블 출력."""
    header = f"{'':28s} {'총수익':8s} {'연수익':8s} {'변동성':8s} {'샤프':6s} {'MDD':8s} {'승률':6s}"
    print(header)
    print("-" * 75)
    for m in metrics_list:
        print(
            f"  {m['label']:26s} "
            f"{m['total_return']:>8s} "
            f"{m['annual_return']:>8s} "
            f"{m['volatility']:>8s} "
            f"{m['sharpe']:>6s} "
            f"{m['max_drawdown']:>8s} "
            f"{m['win_rate']:>6s}"
        )


# ─────────────────────────────────────────────
# 7. 결과 저장
# ─────────────────────────────────────────────

def save_results(train_result: pd.DataFrame, target_result: pd.DataFrame) -> None:
    """결과 CSV 저장."""
    out_dir = ROOT / "data/results/backtests"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_to_save = [
        "crash_score", "mode", "strategy_ret", "strategy_cum",
        "soxl_cum", "tqqq_cum", "SOXL_ret", "TQQQ_ret", "MSTZ_ret",
        "poly_btc_bear", "poly_weekly_dip_bias", "vix_signal", "mkt_momentum",
        "btc_up", "VIX",
    ]

    train_save = train_result[[c for c in cols_to_save if c in train_result.columns]]
    target_save = target_result[[c for c in cols_to_save if c in target_result.columns]]

    train_save.to_csv(out_dir / "poly_crash_train.csv")
    target_save.to_csv(out_dir / "poly_crash_target.csv")
    print(f"\n결과 저장: {out_dir}/poly_crash_train.csv")
    print(f"결과 저장: {out_dir}/poly_crash_target.csv")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    print("Polymarket 데이터 로드 중...")
    poly_df = load_poly_signals()
    print(f"  Polymarket: {len(poly_df)}일, BTC={poly_df['btc_up'].notna().sum()}일")

    print("시장 데이터 로드 중...")
    market_df = load_market_data()
    print(f"  Market: {len(market_df)}일, {market_df.index[0].date()} ~ {market_df.index[-1].date()}")
    print(f"  Tickers: {[c for c in market_df.columns if not c.endswith('_ret') and c not in ['VIX_norm','VIX_chg']]}")

    # 병합
    merged = poly_df.join(market_df, how="left")
    print(f"  Merged: {len(merged)}일")

    # Crash Score 계산
    print("\nCrash Score 계산 중...")
    merged = compute_crash_score(merged)

    # 백테스트
    print("백테스트 실행 중...")
    train_result = backtest(merged, TRAIN_START, TRAIN_END)
    target_result = backtest(merged, TARGET_START, TARGET_END)

    # 리포트
    print_report(train_result, target_result)

    # 저장
    save_results(train_result, target_result)


if __name__ == "__main__":
    main()

"""
JUN 매매법 v2 — 기술적 지표 계산
===================================
일봉 지표: RSI·MA20·MA60·20일 낙폭·5일/10일 수익률
분봉 지표: 당일 위치·직전 30분 모멘텀
매크로 지표: BTC 레짐·VIX 레짐
"""
from __future__ import annotations

import math

import pandas as pd


# ── RSI (Wilder EMA 방식) ──────────────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI 시리즈 반환."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


# ── 일봉 지표 배치 계산 ───────────────────────────────────────────────────────
def compute_daily_indicators(daily: pd.DataFrame, params_ns) -> pd.DataFrame:
    """
    daily: date 인덱스, 컬럼 [open, high, low, close, volume]
    params_ns: JunTradeParams (rsi_period, ma_short, ma_long, rolling_high_period)
    반환: 원본 + 지표 컬럼 추가된 DataFrame
    """
    df = daily.copy().sort_index()
    c = df["close"]

    df["rsi"]              = compute_rsi(c, params_ns.rsi_period)
    df["ma20"]             = c.rolling(params_ns.ma_short).mean()
    df["ma60"]             = c.rolling(params_ns.ma_long).mean()
    df["pct_from_ma20"]    = (c / df["ma20"] - 1) * 100
    df["pct_from_ma60"]    = (c / df["ma60"] - 1) * 100
    df["rolling_high_20"]  = c.rolling(params_ns.rolling_high_period).max()
    df["drawdown_20d"]     = (c / df["rolling_high_20"] - 1) * 100
    df["return_5d"]        = (c / c.shift(5) - 1) * 100
    df["return_10d"]       = (c / c.shift(10) - 1) * 100

    return df


# ── BTC 레짐 ──────────────────────────────────────────────────────────────────
def btc_regime(btc_price: float, btc_ma20: float, btc_ma60: float,
               btc_rsi: float, btc_10d_ret: float) -> int:
    """
    0: BTC < MA60 (약세)
    1: BTC > MA60 (중립)
    2: BTC > MA20 AND RSI > 55 (강세)
    3: BTC > MA20 AND RSI > 65 AND 10d_ret > 15% (과열)
    """
    if btc_price > btc_ma20 and btc_rsi > 65 and btc_10d_ret > 15:
        return 3
    if btc_price > btc_ma20 and btc_rsi > 55:
        return 2
    if btc_price > btc_ma60:
        return 1
    return 0


# ── VIX 레짐 ─────────────────────────────────────────────────────────────────
VIX_REGIMES = {
    "extreme_greed": (0,    15),
    "greed":         (15,   20),
    "neutral":       (20,   25),   # 최적 구간 — 승률 91.1%
    "fear":          (25,   30),
    "extreme_fear":  (30, 9999),
}


def vix_regime(vix: float) -> str:
    for name, (lo, hi) in VIX_REGIMES.items():
        if lo <= vix < hi:
            return name
    return "extreme_fear"


def vix_position_size(vix: float, params) -> int:
    """VIX 레짐별 1회 매수 금액(KRW) 반환. fear 이상이면 0."""
    regime = vix_regime(vix)
    if regime in ("fear", "extreme_fear"):
        return 0
    if regime == "neutral":
        return params.size_vix_optimal
    if regime == "greed":
        return params.size_vix_normal
    # extreme_greed
    return params.size_vix_overheat


# ── 분봉 지표 (당일 intraday) ─────────────────────────────────────────────────
def intraday_day_position(price: float, day_low: float, day_high: float) -> float:
    """당일 저·고 대비 현재가 위치 (0=저점, 1=고점)."""
    rng = day_high - day_low
    if rng <= 0:
        return 0.5
    return (price - day_low) / rng


def intraday_mom_30min(bars_before: pd.DataFrame, price: float) -> float | None:
    """
    bars_before: 진입 timestamp 직전 봉들 (시간순 정렬)
    price: 현재 진입가
    반환: 직전 30분 수익률(%). 데이터 부족 시 None.
    """
    tail = bars_before.tail(30)
    if len(tail) < 5:
        return None
    base = float(tail["close"].iloc[0])
    if base == 0:
        return None
    return (price / base - 1) * 100


def mins_from_open(ts: pd.Timestamp) -> int:
    """ET 기준 장 시작(09:30) 후 경과 분."""
    return (ts.hour - 9) * 60 + ts.minute - 30


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None

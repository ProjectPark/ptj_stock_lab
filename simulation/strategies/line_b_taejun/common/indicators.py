"""공통 기술적 지표 계산 함수.

conditional_conl.py, soxl_independent.py 등에서 공유.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def compute_adx(df: "pd.DataFrame", period: int = 14) -> float:
    """ADX(14)를 계산한다."""
    import math

    import pandas as pd

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)

    tr_s = tr.rolling(period).mean()
    plus_s = plus_dm.rolling(period).mean()
    minus_s = minus_dm.rolling(period).mean()

    plus_di = 100 * plus_s / tr_s.replace(0, float("nan"))
    minus_di = 100 * minus_s / tr_s.replace(0, float("nan"))
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.rolling(period).mean()
    val = adx.iloc[-1]
    return float(val) if not math.isnan(val) else 0.0


def ema_slope_positive(df: "pd.DataFrame", span: int = 20) -> bool:
    """20 EMA 기울기가 양수인지 확인한다."""
    close = df["close"].astype(float)
    ema = close.ewm(span=span, adjust=False).mean()
    if len(ema) < 2:
        return False
    return ema.iloc[-1] > ema.iloc[-2]

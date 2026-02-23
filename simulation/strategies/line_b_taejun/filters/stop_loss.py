"""
ATR 기반 손절 계산기 (Stop Loss)
==========================================
v5 매매 규칙 5절 — ATR 손절 + 레버리지 배수별 고변동성 차등 손절.

5-1. ATR 기반 손절: 진입가 - 1.5 × ATR14
5-2. 완화 손절 (강세장): 진입가 - 2.5 × ATR14
5-4. 고변동성 손절 (레버리지 차등):
     1x: -4%, 2x: -6%, 3x: -8%
     (ATR 손절과 비교해 더 엄격한 것 적용)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.params import STOP_LOSS

if TYPE_CHECKING:
    import pandas as pd


def _compute_atr14(df: "pd.DataFrame", period: int = 14) -> float:
    """ATR(14)를 계산한다."""
    import pandas as pd

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    val = atr.iloc[-1]
    return float(val) if not __import__("math").isnan(val) else 0.0


class StopLossCalculator:
    """ATR 기반 손절가 계산 + 레버리지별 고변동성 차등 손절.

    Parameters
    ----------
    params : dict | None
        STOP_LOSS 파라미터. None이면 기본값.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or STOP_LOSS

    # ------------------------------------------------------------------
    # 레버리지 배수 조회
    # ------------------------------------------------------------------

    def get_leverage(self, ticker: str) -> int:
        """종목의 레버리지 배수를 반환한다. 미등록 종목은 1로 처리."""
        lev_map = self.params.get("leverage_map", {})
        return lev_map.get(ticker, 1)

    # ------------------------------------------------------------------
    # 고변동성 판정
    # ------------------------------------------------------------------

    def is_high_volatility(
        self,
        ticker: str,
        recent_changes: list[float],
    ) -> bool:
        """최근 N거래일 내 고변동성 조건 충족 여부.

        Parameters
        ----------
        ticker : str
        recent_changes : list[float]
            최근 거래일 등락률(%) 목록 (최신순 또는 과거순 무관).
            최대 `high_vol_lookback` 개만 사용.

        Returns
        -------
        bool
        """
        leverage = self.get_leverage(ticker)
        lookback = self.params.get("high_vol_lookback", 5)
        min_count = self.params.get("high_vol_min_count", 2)
        thresholds: dict = self.params.get("high_vol_threshold", {1: 10.0, 2: 15.0, 3: 20.0})
        threshold = thresholds.get(leverage, 10.0)

        window = recent_changes[:lookback]
        count = sum(1 for c in window if abs(c) >= threshold)
        return count >= min_count

    # ------------------------------------------------------------------
    # 손절가 계산
    # ------------------------------------------------------------------

    def calc_atr_stop(
        self,
        entry_price: float,
        atr14: float,
        bullish: bool = False,
    ) -> float:
        """ATR 기반 손절가를 반환한다.

        Parameters
        ----------
        entry_price : float
        atr14 : float
            진입 시점의 ATR14 값.
        bullish : bool
            강세장 모드(Polymarket NDX >= 70%) 여부.
            True이면 2.5×ATR, False이면 1.5×ATR.

        Returns
        -------
        float
            손절가 (USD).
        """
        if bullish:
            mult = self.params.get("atr_multiplier_bullish", 2.5)
        else:
            mult = self.params.get("atr_multiplier", 1.5)
        return entry_price - atr14 * mult

    def calc_high_vol_stop(
        self,
        entry_price: float,
        ticker: str,
    ) -> float:
        """고변동성 고정 손절가를 반환한다 (레버리지별 차등).

        Returns
        -------
        float
            손절가 (USD).
        """
        leverage = self.get_leverage(ticker)
        stop_pcts: dict = self.params.get("high_vol_stop_pct", {1: -4.0, 2: -6.0, 3: -8.0})
        stop_pct = stop_pcts.get(leverage, -4.0)
        return entry_price * (1 + stop_pct / 100)

    def get_effective_stop(
        self,
        entry_price: float,
        atr14: float,
        ticker: str,
        recent_changes: list[float] | None = None,
        bullish: bool = False,
    ) -> tuple[float, str]:
        """ATR 손절과 고변동성 손절 중 더 엄격한(높은) 손절가를 반환한다.

        Parameters
        ----------
        entry_price : float
        atr14 : float
        ticker : str
        recent_changes : list[float] | None
            최근 등락률 목록. None이면 고변동성 체크 스킵.
        bullish : bool

        Returns
        -------
        (stop_price, reason)
        """
        atr_stop = self.calc_atr_stop(entry_price, atr14, bullish)

        if recent_changes and self.is_high_volatility(ticker, recent_changes):
            hv_stop = self.calc_high_vol_stop(entry_price, ticker)
            # 더 높은(더 엄격한) 손절 적용
            if hv_stop > atr_stop:
                leverage = self.get_leverage(ticker)
                stop_pcts: dict = self.params.get(
                    "high_vol_stop_pct", {1: -4.0, 2: -6.0, 3: -8.0}
                )
                pct = stop_pcts.get(leverage, -4.0)
                return hv_stop, f"고변동성 손절 ({leverage}x: {pct:.0f}%)"
            return atr_stop, "ATR 손절 (고변동성 손절보다 엄격)"

        mult = (
            self.params.get("atr_multiplier_bullish", 2.5) if bullish
            else self.params.get("atr_multiplier", 1.5)
        )
        mode = "강세장 완화" if bullish else "일반"
        return atr_stop, f"ATR 손절 {mode} ({mult}×ATR)"

    # ------------------------------------------------------------------
    # OHLCV 기반 ATR14 계산 (DataFrame 직접 전달)
    # ------------------------------------------------------------------

    def calc_atr14_from_df(
        self,
        df: "pd.DataFrame",
        period: int = 14,
    ) -> float:
        """OHLCV DataFrame에서 ATR14를 계산한다."""
        return _compute_atr14(df, period)

    def get_effective_stop_from_df(
        self,
        entry_price: float,
        df: "pd.DataFrame",
        ticker: str,
        recent_changes: list[float] | None = None,
        bullish: bool = False,
    ) -> tuple[float, str]:
        """DataFrame에서 ATR14를 계산 후 유효 손절가를 반환한다."""
        atr14 = self.calc_atr14_from_df(df)
        return self.get_effective_stop(
            entry_price, atr14, ticker, recent_changes, bullish
        )

    # ------------------------------------------------------------------
    # 손절 조건 체크 (포지션 관리 통합용)
    # ------------------------------------------------------------------

    def should_stop(
        self,
        current_price: float,
        stop_price: float,
    ) -> bool:
        """현재가가 손절가 이하인지 확인한다."""
        return current_price <= stop_price

    def summary(
        self,
        entry_price: float,
        atr14: float,
        ticker: str,
        recent_changes: list[float] | None = None,
        bullish: bool = False,
    ) -> dict:
        """손절 파라미터 요약."""
        stop_price, reason = self.get_effective_stop(
            entry_price, atr14, ticker, recent_changes, bullish
        )
        leverage = self.get_leverage(ticker)
        is_hv = self.is_high_volatility(ticker, recent_changes or [])

        return {
            "ticker": ticker,
            "leverage": leverage,
            "entry_price": round(entry_price, 4),
            "atr14": round(atr14, 4),
            "stop_price": round(stop_price, 4),
            "stop_pct": round((stop_price / entry_price - 1) * 100, 2),
            "reason": reason,
            "is_high_vol": is_hv,
            "bullish_mode": bullish,
        }

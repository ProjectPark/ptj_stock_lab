"""
Polymarket 데이터 품질 필터 (Poly-Quality)
============================================
필터 규칙:
1. 0%, 1%, 100% 값 제외 (극단값/미활성)
2. 5시간 이상 변동 없는 값 제외
3. Q-11: NDX 데이터 미갱신시 관련 조건 정지

출처: kakaotalk_trading_notes_2026-02-19.csv — Q-11 NDX 정지 로직
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..common.params import POLY_QUALITY


class PolyQualityFilter:
    """Polymarket 데이터 품질 필터.

    전략 dispatch 전에 MarketData.poly를 필터링한다.
    유효하지 않은 키는 제거되어 전략에 전달되지 않는다.

    Parameters
    ----------
    params : dict | None
        POLY_QUALITY 파라미터. None이면 기본값.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or POLY_QUALITY

    def filter(
        self,
        poly: dict[str, float] | None,
        poly_timestamps: dict[str, datetime] | None = None,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """유효한 poly 값만 필터링하여 반환한다.

        Parameters
        ----------
        poly : dict | None
            원본 Polymarket 확률.
        poly_timestamps : dict | None
            각 키별 마지막 갱신 시각.
        now : datetime | None
            현재 시각. None이면 datetime.now().

        Returns
        -------
        dict[str, float]
            필터 통과한 poly 값.
        """
        if not poly:
            return {}

        now = now or datetime.now()
        min_prob = self.params.get("min_prob", 0.02)
        max_prob = self.params.get("max_prob", 0.99)
        stale_hours = self.params.get("min_volatility_hours", 5)

        filtered: dict[str, float] = {}
        for key, val in poly.items():
            # 극단값 제외
            if val < min_prob or val > max_prob:
                continue

            # 갱신 시각 체크
            if poly_timestamps and self.params.get("stale_pause", True):
                ts = poly_timestamps.get(key)
                if ts and (now - ts) > timedelta(hours=stale_hours):
                    continue

            filtered[key] = val

        return filtered

    def ndx_is_reliable(
        self,
        poly_timestamps: dict[str, datetime] | None = None,
        now: datetime | None = None,
    ) -> bool:
        """NDX Polymarket 데이터가 신뢰할 수 있는지 확인한다.

        미갱신시 NDX 의존 조건을 정지해야 한다.
        영향받는 전략: jab_soxl, jab_tsll, sp500_entry, bargain_buy block_rules

        Returns
        -------
        bool
            True이면 NDX 데이터 유효, False이면 정지 필요.
        """
        if not poly_timestamps:
            return True  # 타임스탬프 없으면 신뢰 가정

        now = now or datetime.now()
        stale_hours = self.params.get("min_volatility_hours", 5)

        ndx_ts = poly_timestamps.get("ndx_up")
        if ndx_ts is None:
            return False  # NDX 데이터 자체가 없음

        return (now - ndx_ts) <= timedelta(hours=stale_hours)

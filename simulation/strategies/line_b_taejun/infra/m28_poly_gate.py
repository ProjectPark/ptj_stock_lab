"""
M28 Polymarket 포지션 게이트
============================
BTC/NDX 확률 기반 LONG/SHORT/NEUTRAL 분기.

출처: MT_VNQ3.md §12 (M28)
"""
from __future__ import annotations

from typing import Literal


class M28PolyGate:
    """Polymarket BTC/NDX 확률 기반 포지션 게이트.

    BTC primary 선택:
    - volume 기반 A/B 선정
    - ±1% 이내면 직전 primary 유지 (없으면 A 고정)

    확률 게이트:
    - p >= 0.51 → LONG
    - p <= 0.49 → SHORT
    - else → NEUTRAL
    """

    def __init__(self):
        self._prev_btc_primary: str | None = None

    def select_btc_primary(self, volume_a: float, volume_b: float,
                           prev_primary: str | None = None) -> str:
        """BTC primary 계약 선택.

        Parameters
        ----------
        volume_a : float
            계약 A의 거래량.
        volume_b : float
            계약 B의 거래량.
        prev_primary : str | None
            직전 primary. None이면 self._prev_btc_primary 사용.

        Returns
        -------
        str
            "A" 또는 "B"
        """
        prev = prev_primary or self._prev_btc_primary

        if volume_a <= 0 and volume_b <= 0:
            result = prev or "A"
            self._prev_btc_primary = result
            return result

        total = volume_a + volume_b
        if total <= 0:
            result = prev or "A"
            self._prev_btc_primary = result
            return result

        ratio_a = volume_a / total
        ratio_b = volume_b / total

        # ±1% 이내면 직전 유지
        if abs(ratio_a - ratio_b) <= 0.01:
            result = prev or "A"
            self._prev_btc_primary = result
            return result

        result = "A" if volume_a >= volume_b else "B"
        self._prev_btc_primary = result
        return result

    @staticmethod
    def btc_gate(p: float) -> Literal["LONG", "SHORT", "NEUTRAL"]:
        """BTC 확률 기반 포지션 방향.

        Parameters
        ----------
        p : float
            BTC 상승 확률 (0~1).

        Returns
        -------
        Literal["LONG", "SHORT", "NEUTRAL"]
        """
        if p >= 0.51:
            return "LONG"
        if p <= 0.49:
            return "SHORT"
        return "NEUTRAL"

    @staticmethod
    def ndx_gate(ndx_p: float) -> Literal["LONG", "SHORT", "NEUTRAL"]:
        """NDX 확률 기반 포지션 방향.

        Parameters
        ----------
        ndx_p : float
            NDX 상승 확률 (0~1).

        Returns
        -------
        Literal["LONG", "SHORT", "NEUTRAL"]
        """
        if ndx_p >= 0.51:
            return "LONG"
        if ndx_p <= 0.49:
            return "SHORT"
        return "NEUTRAL"

    def evaluate(self, poly: dict[str, float] | None) -> dict:
        """종합 게이트 평가.

        Parameters
        ----------
        poly : dict[str, float] | None
            Polymarket 확률 dict. {"btc_up": 0.63, "ndx_up": 0.55, ...}

        Returns
        -------
        dict
            {"btc_direction": str, "ndx_direction": str,
             "btc_p": float, "ndx_p": float}
        """
        if poly is None:
            return {
                "btc_direction": "NEUTRAL",
                "ndx_direction": "NEUTRAL",
                "btc_p": 0.5,
                "ndx_p": 0.5,
            }

        btc_p = poly.get("btc_up", 0.5)
        ndx_p = poly.get("ndx_up", 0.5)

        return {
            "btc_direction": self.btc_gate(btc_p),
            "ndx_direction": self.ndx_gate(ndx_p),
            "btc_p": btc_p,
            "ndx_p": ndx_p,
        }

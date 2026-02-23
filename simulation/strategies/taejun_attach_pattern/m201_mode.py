"""
M201 -- 즉시모드 v1.0 (BTC 확률 급변/전환)
========================================
출처: MT_VNQ3.md S14
우선순위: M200(킬스위치) 다음, 리스크모드 이전
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class M201Action(Enum):
    NONE = "none"
    CLOSE_LONG = "close_long"    # 롱 즉시 청산
    CLOSE_SHORT = "close_short"  # 숏 즉시 청산
    ENTER_SHORT = "enter_short"  # 청산 후 숏 진입
    ENTER_LONG = "enter_long"    # 청산 후 롱 진입


@dataclass
class M201Signal:
    action: M201Action
    reason: str
    p: float
    delta_pp: float


class M201ImmediateMode:
    """BTC 확률 급변/전환 대응 즉시모드.

    입력: p (현재 BTC 상승 확률), p_prev (직전 확률)
    delta_pp = (p - p_prev) * 100

    LONG 보유 중:
      - p <= 0.45 또는 (delta_pp <= -20 and p <= 0.49) -> CLOSE_LONG
      - 청산 후: p <= 0.49 -> ENTER_SHORT / 0.49 < p < 0.50 -> 현금 유지

    SHORT 보유 중:
      - p >= 0.60 -> CLOSE_SHORT
      - 청산 후: p >= 0.51 -> ENTER_LONG / 0.49 < p < 0.50 -> 현금 유지
    """

    def check_long(self, p: float, p_prev: float) -> Optional[M201Signal]:
        """LONG 포지션 보유 중 즉시 청산 여부 판단."""
        delta_pp = (p - p_prev) * 100
        if p <= 0.45:
            return M201Signal(
                M201Action.CLOSE_LONG,
                f"p={p:.3f} <= 0.45",
                p, delta_pp,
            )
        if delta_pp <= -20 and p <= 0.49:
            return M201Signal(
                M201Action.CLOSE_LONG,
                f"delta_pp={delta_pp:.1f}pp <= -20 and p={p:.3f} <= 0.49",
                p, delta_pp,
            )
        return None

    def check_short(self, p: float, p_prev: float) -> Optional[M201Signal]:
        """SHORT 포지션 보유 중 즉시 청산 여부 판단."""
        delta_pp = (p - p_prev) * 100
        if p >= 0.60:
            return M201Signal(
                M201Action.CLOSE_SHORT,
                f"p={p:.3f} >= 0.60",
                p, delta_pp,
            )
        return None

    def post_close_action(self, p: float, closed_side: str) -> M201Action:
        """청산 직후 전환 여부 판단.

        Parameters
        ----------
        p : float
            현재 BTC 상승 확률.
        closed_side : str
            "long" or "short"
        """
        if closed_side == "long":
            if p <= 0.49:
                return M201Action.ENTER_SHORT
            return M201Action.NONE  # 0.49 < p -> 현금 유지
        else:  # short
            if p >= 0.51:
                return M201Action.ENTER_LONG
            return M201Action.NONE  # 0.49 < p < 0.50 -> 현금 유지

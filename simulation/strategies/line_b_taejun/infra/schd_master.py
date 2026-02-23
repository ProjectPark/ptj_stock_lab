"""
SCHD 장기 적립 + 매도 차단
===========================
30일 실현손익 양수 시 구간별 매수 + 모든 모드에서 매도 차단.

출처: MT_VNQ3.md §16 (SCHD)
"""
from __future__ import annotations


class SCHDMaster:
    """SCHD 장기 적립 전략.

    Tier별 매수 금액:
    - Tier 1: ₩100,000
    - Tier 2: ₩200,000
    - Tier 3: ₩300,000
    - Tier 4: ₩400,000
    - Tier 5: ₩500,000

    매도 차단:
    - SCHD는 M200/M201/이머전시/리스크 모드 관계없이 매도 금지.
    """

    TIERS: list[tuple[int, int]] = [
        (1, 100_000),
        (2, 200_000),
        (3, 300_000),
        (4, 400_000),
        (5, 500_000),
    ]

    def should_buy(self, realized_pnl_30d_pct: float) -> tuple[bool, int]:
        """30일 실현손익 기반 매수 여부 + 금액 결정.

        Parameters
        ----------
        realized_pnl_30d_pct : float
            최근 30일 실현 손익률 (%).

        Returns
        -------
        tuple[bool, int]
            (매수 여부, 매수 금액 KRW).
            손익 음수 시 (False, 0).
        """
        if realized_pnl_30d_pct <= 0:
            return False, 0

        # 수익률에 따른 Tier 선택
        # 1% 미만 → Tier 1, 2% 미만 → Tier 2, ...
        tier_idx = min(int(realized_pnl_30d_pct), len(self.TIERS) - 1)
        tier_num, amount = self.TIERS[tier_idx]
        return True, amount

    @staticmethod
    def is_sell_blocked(ticker: str) -> bool:
        """SCHD 매도 차단 여부.

        Parameters
        ----------
        ticker : str

        Returns
        -------
        bool
            ticker == "SCHD" → True (항상 차단).
        """
        return ticker.upper() == "SCHD"

    @staticmethod
    def should_exclude_from_sell(ticker: str, mode: str) -> bool:
        """특정 모드에서 SCHD 매도 제외 여부.

        Parameters
        ----------
        ticker : str
        mode : str
            "M200", "M201", "emergency", "risk", "normal" 등.

        Returns
        -------
        bool
            SCHD이면 모든 모드에서 True (매도 제외).
        """
        if ticker.upper() != "SCHD":
            return False
        # SCHD는 모든 모드에서 매도 차단
        return True


# M300 규칙 — 환전/신호 전용 티커 차단
FORBIDDEN_FX = True  # 환전 주문 타입 차단
SIGNAL_ONLY_TICKERS = ["VNQ", "SK리츠", "TIGER 리츠부동산인프라", "롯데리츠"]

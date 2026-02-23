"""
M5 비중 관리자 — T1~T4 비중 배분 + 동적 조정
=============================================
순차 배분: T1 55%, T2 잔여 40%, T3 잔여 33%, T4 잔여 전액.
T5+: 예약 대기 → 10초 후 조건 미부합 시 취소.

동적 조정: GLD/VIX/USD 변동에 따른 비중 조절.

출처: MT_VNQ3.md §1, §10
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Allocation:
    """배분 결과."""
    tier: int
    ticker: str
    weight: float  # 0.0 ~ 1.0
    amount: float  # USD
    reserved: bool = False  # T5+ 예약 여부


class M5WeightManager:
    """M5 비중 배분 매니저.

    T1~T4 순차 배분 + 동적 조정.
    """

    # 순차 배분 비율
    TIER_RATIOS = [0.55, 0.40, 0.33, 1.0]  # T1, T2, T3, T4(잔여 전액)

    def allocate(
        self,
        signals: list,
        cash: float,
        total_assets: float,
    ) -> list[Allocation]:
        """T1~T4 비중 일괄 배분.

        Parameters
        ----------
        signals : list[Signal]
            매수 시그널 리스트 (우선순위 순).
        cash : float
            가용 현금.
        total_assets : float
            총 자산.

        Returns
        -------
        list[Allocation]
        """
        allocations: list[Allocation] = []
        remaining = cash

        for i, sig in enumerate(signals):
            tier = i + 1
            if tier <= len(self.TIER_RATIOS):
                ratio = self.TIER_RATIOS[tier - 1]
            else:
                # T5+: 예약 대기
                allocations.append(Allocation(
                    tier=tier,
                    ticker=sig.ticker,
                    weight=0.0,
                    amount=0.0,
                    reserved=True,
                ))
                continue

            amount = remaining * ratio
            weight = amount / total_assets if total_assets > 0 else 0.0
            allocations.append(Allocation(
                tier=tier,
                ticker=sig.ticker,
                weight=weight,
                amount=amount,
            ))
            remaining -= amount

        return allocations

    @staticmethod
    def dynamic_adjust(
        base_weight: float,
        gld_pct: float = 0.0,
        vix_pct: float = 0.0,
        usd_change: float = 0.0,
    ) -> float:
        """GLD/VIX/USD 변동에 따른 비중 동적 조정.

        Parameters
        ----------
        base_weight : float
            기본 비중 (%).
        gld_pct : float
            GLD 일간 변동률 (%).
        vix_pct : float
            VIX 일간 변동률 (%).
        usd_change : float
            USD 환율 변동 (양수=상승, 음수=하락).

        Returns
        -------
        float
            조정된 비중 (%).

        Rules
        -----
        - GLD: +1%마다 -0.1%p / -1%마다 +0.1%p
        - VIX: +2%마다 -3%p / -2%마다 +1%p
        - USD: 상승 +0.1%p / 하락 -0.2%p
        - 동시 발생 → 중첩 합산
        """
        adj = 0.0

        # GLD 조정
        if gld_pct != 0:
            adj += -0.1 * gld_pct  # +1% → -0.1, -1% → +0.1

        # VIX 조정
        if vix_pct >= 2.0:
            adj += -3.0 * (vix_pct / 2.0)
        elif vix_pct <= -2.0:
            adj += 1.0 * (abs(vix_pct) / 2.0)

        # USD 조정
        if usd_change > 0:
            adj += 0.1
        elif usd_change < 0:
            adj -= 0.2

        return base_weight + adj

    @staticmethod
    def check_limits(weight: float) -> bool:
        """비중 한도 체크.

        Parameters
        ----------
        weight : float
            현재 비중 (%).

        Returns
        -------
        bool
            True if within limits (매수 가능), False if 매수 금지.

        Rules
        -----
        - -1% 이하 → 매수 금지
        - 100.1% 이상 → 매수 금지
        """
        if weight <= -1.0:
            return False
        if weight >= 100.1:
            return False
        return True

    @staticmethod
    def sequential_allocations(free_cash: float) -> list[float]:
        """순차 배분 금액 계산.

        T1=55%, T2=40% of remainder, T3=33% of remainder, T4=rest.

        Parameters
        ----------
        free_cash : float
            가용 현금.

        Returns
        -------
        list[float]
            [T1_amount, T2_amount, T3_amount, T4_amount]
        """
        t1 = free_cash * 0.55
        rem1 = free_cash - t1
        t2 = rem1 * 0.40
        rem2 = rem1 - t2
        t3 = rem2 * 0.33
        t4 = rem2 - t3
        return [t1, t2, t3, t4]

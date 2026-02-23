"""
수익금 분배기 (Profit Distributor)
===================================
매도 수익금을 SOXL → ROBN → GLD → CONL 순서로 분배.
부족 시 skip.

출처: MT_VNQ3.md §17 (P-1 잠정 처리)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DistributionResult:
    """분배 결과."""
    ticker: str
    amount: float  # USD
    reason: str
    skipped: bool = False


class ProfitDistributor:
    """매도 수익금 분배기.

    분배 순서: SOXL → ROBN → GLD → CONL (P-1 잠정)
    부족 시 해당 항목 skip → 다음으로.
    """

    DEFAULT_SEQUENCE = ["SOXL", "ROBN", "GLD", "CONL"]

    def __init__(self, sequence: list[str] | None = None,
                 distribution_ratios: dict[str, float] | None = None):
        """
        Parameters
        ----------
        sequence : list[str] | None
            분배 순서. None이면 DEFAULT_SEQUENCE.
        distribution_ratios : dict[str, float] | None
            종목별 분배 비율 (0~1). None이면 균등 배분.
        """
        self.sequence = sequence or self.DEFAULT_SEQUENCE
        self.distribution_ratios = distribution_ratios or {
            t: 1.0 / len(self.sequence) for t in self.sequence
        }

    def distribute(self, profit_usd: float,
                   available_tickers: set[str] | None = None,
                   ) -> list[DistributionResult]:
        """수익금 분배.

        Parameters
        ----------
        profit_usd : float
            분배할 수익금 (USD).
        available_tickers : set[str] | None
            매수 가능한 종목 집합. None이면 전체 가능.

        Returns
        -------
        list[DistributionResult]
        """
        if profit_usd <= 0:
            return []

        results: list[DistributionResult] = []
        remaining = profit_usd
        available = available_tickers or set(self.sequence)

        for ticker in self.sequence:
            if remaining <= 0:
                break

            ratio = self.distribution_ratios.get(ticker, 0)
            amount = profit_usd * ratio

            if ticker not in available:
                results.append(DistributionResult(
                    ticker=ticker,
                    amount=0,
                    reason=f"ticker {ticker} not available",
                    skipped=True,
                ))
                continue

            if amount > remaining:
                amount = remaining

            results.append(DistributionResult(
                ticker=ticker,
                amount=amount,
                reason=f"profit distribution: {ratio:.0%} of ${profit_usd:.2f}",
            ))
            remaining -= amount

        return results

    def distribute_sequential(self, profit_usd: float,
                              min_buy_usd: float = 50.0,
                              ) -> list[DistributionResult]:
        """순차 분배 (P-1 잠정).

        첫 번째 종목에 전액 시도, 불가하면 다음으로.

        Parameters
        ----------
        profit_usd : float
            분배할 수익금.
        min_buy_usd : float
            최소 매수 금액. 미만이면 skip.

        Returns
        -------
        list[DistributionResult]
        """
        if profit_usd <= 0:
            return []

        results: list[DistributionResult] = []

        for ticker in self.sequence:
            if profit_usd < min_buy_usd:
                results.append(DistributionResult(
                    ticker=ticker,
                    amount=0,
                    reason=f"remaining ${profit_usd:.2f} < min ${min_buy_usd:.2f}",
                    skipped=True,
                ))
                continue

            results.append(DistributionResult(
                ticker=ticker,
                amount=profit_usd,
                reason=f"sequential distribution: ${profit_usd:.2f} to {ticker}",
            ))
            # P-1 잠정: 순차 → 첫 번째에 전액 배분
            profit_usd = 0
            break

        return results

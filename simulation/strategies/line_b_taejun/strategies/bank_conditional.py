"""
조건부 은행주 전략 (Bank-Conditional)
======================================
대형 은행 5개 모두 상승 중 BAC만 하락 = 역전 기대.
300만원 투자, +0.5% (수수료 포함) 매도.

출처: kakaotalk_trading_notes_2026-02-19.csv
- 조건부 매매 은행주 제이피모건체이스 HSBC 웰스파고...
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import BANK_CONDITIONAL
from ..common.registry import register


@register
class BankConditional(BaseStrategy):
    """조건부 은행주 — 대형은행 상승 + BAC 하락시 역전 매수."""

    name = "bank_conditional"
    version = "1.0"
    description = "JPM/HSBC/WFC/RBC/C 상승 & BAC 하락시 매수, +0.5% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or BANK_CONDITIONAL)

    def check_entry(self, market: MarketData) -> bool:
        """진입 조건.

        1. watch_tickers (JPM, HSBC, WFC, RBC, C) 모두 양전 (> 0%)
        2. target_ticker (BAC) 마이너스 (< 0%)
        """
        watch = self.params.get("watch_tickers", [])
        target = self.params.get("target_ticker", "BAC")

        # 모든 감시 종목이 양전
        for ticker in watch:
            chg = market.changes.get(ticker, None)
            if chg is None or chg <= 0:
                return False

        # 타겟 종목이 마이너스
        target_chg = market.changes.get(target, None)
        if target_chg is None or target_chg >= 0:
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        current = market.prices.get(position.ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        target = self.params.get("target_ticker", "BAC")

        if position is not None:
            current = market.prices.get(target, 0)
            if current <= 0:
                return Signal(Action.HOLD, target, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=target, size=1.0, target_pct=0,
                    reason=f"bank_cond target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={"pnl_pct": pnl_pct, "reinvest": "cash"},
                )
            return Signal(Action.HOLD, target, 0, self.params["target_pct"],
                         f"holding BAC: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            return Signal(Action.SKIP, target, 0, 0, "bank conditions not met")

        watch_chgs = {t: market.changes.get(t, 0)
                      for t in self.params.get("watch_tickers", [])}
        return Signal(
            action=Action.BUY, ticker=target, size=1.0,
            target_pct=self.params["target_pct"],
            reason=(
                f"bank_cond entry: watch all positive, "
                f"BAC={market.changes.get(target, 0):+.2f}%"
            ),
            metadata={
                "amount_krw": self.params.get("amount_krw", 3_000_000),
                "watch_changes": watch_chgs,
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("watch_tickers"):
            errors.append("watch_tickers required")
        if not self.params.get("target_ticker"):
            errors.append("target_ticker required")
        return errors

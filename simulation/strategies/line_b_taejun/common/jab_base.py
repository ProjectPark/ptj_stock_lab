"""잽모드 전략 공통 베이스.

jab_soxl, jab_bitu, jab_tsll, jab_etq 공통:
- _is_in_window(): KST 시간 윈도우 체크
- check_exit(): target_pct 비교 익절
- _make_exit_signal(): 익절 시그널 생성
- _make_hold_signal(): 보유 HOLD 시그널 생성
"""
from __future__ import annotations

from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal


class JabBase(BaseStrategy):
    """잽모드 전략 공통 베이스."""

    def _is_in_window(self, market: MarketData) -> bool:
        """KST 시간 윈도우 체크."""
        start = self.params.get("entry_start_kst", (17, 30))
        h, m = market.time.hour, market.time.minute
        return (h, m) >= tuple(start)

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """target_pct 익절 체크."""
        ticker = self.params.get("ticker", "")
        current = market.prices.get(ticker, 0)
        pnl = position.pnl_pct(current)
        if pnl is None:
            return False
        return pnl >= self.params["target_pct"]

    def _make_exit_signal(self, market: MarketData, position: Position,
                          ticker: str) -> Signal:
        """공통 익절/홀드 시그널 생성."""
        current = market.prices.get(ticker, 0)
        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         "no price data")

        pnl = position.pnl_pct(current)
        if pnl is None:
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         "no price data")

        if pnl >= self.params["target_pct"]:
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,
                target_pct=0,
                reason=f"{self.name} target hit: {pnl:.2f}% >= {self.params['target_pct']}%",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={"pnl_pct": pnl},
            )

        return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                      f"holding {ticker}: pnl={pnl:.2f}%")

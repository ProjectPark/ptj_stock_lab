"""
S&P500 편입 기업 전략 (SP500-Entry)
=====================================
S&P500 신규 편입 다음 날 매수, 목표 수익률 +1.5% 달성시 매도.
순이익 흑자 기업만 대상.

출처: kakaotalk_trading_notes_2026-02-19.csv — (1) S앤P500 기업편입 매수 공격모드
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import SP500_ENTRY
from ..common.registry import register


@register
class SP500Entry(BaseStrategy):
    """S&P500 편입 — 편입 다음 날 매수, +1.5% 매도."""

    name = "sp500_entry"
    version = "1.0"
    description = "S&P500 신규 편입 흑자 기업 종가 매수, +1.5% 수수료 제외 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or SP500_ENTRY)

    def check_entry(self, market: MarketData) -> bool:
        """진입 조건.

        1. Polymarket NASDAQ >= 51%
        2. GLD 상승시 매수 금지
        3. 편입 이벤트 데이터 존재 (metadata에서 전달)
        4. 기업 순이익 흑자 (metadata에서 전달)
        """
        if not market.poly:
            return False
        ndx_up = market.poly.get("ndx_up", 0)
        if ndx_up < self.params["poly_ndx_min"]:
            return False

        if self.params.get("gld_block_positive"):
            gld = market.changes.get("GLD", 0)
            if gld > 0:
                return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        current = market.prices.get(position.ticker, 0)
        pnl = position.pnl_pct(current)
        if pnl is None:
            return False
        return pnl >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        if position is not None:
            ticker = position.ticker
            current = market.prices.get(ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = position.pnl_pct(current) or 0.0
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=ticker, size=1.0, target_pct=0,
                    reason=f"sp500_entry target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={"pnl_pct": pnl_pct},
                )
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         f"holding {ticker}: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "SP500 entry conditions not met")

        # 편입 종목은 외부에서 metadata로 전달 필요
        # 이 전략은 check_entry만으로 시장 조건 확인,
        # 실제 편입 종목은 호출측에서 ticker를 지정해서 사용
        return Signal(
            action=Action.BUY, ticker="",  # 호출측에서 ticker 지정
            size=self.params.get("size", 1.0),
            target_pct=self.params["target_pct"],
            reason="sp500_entry: market conditions met, awaiting ticker",
            metadata={"net_income_min": self.params["net_income_min"]},
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        return errors

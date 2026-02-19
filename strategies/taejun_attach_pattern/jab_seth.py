"""
숏 잽모드 SETH 전략 (Jab-SETH)
================================
이더리움 하락 기대가 높을 때 숏 ETF 매수.

출처: kakaotalk_trading_notes_2026-02-19.csv — (4) 숏 잽모드 ETH 매수 <공격모드>
"""
from __future__ import annotations

from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from .params import JAB_SETH
from .registry import register


@register
class JabSETH(BaseStrategy):
    """숏 잽모드 SETH — ETH 하락 기대 높을 때 숏 ETF 매수."""

    name = "jab_seth"
    version = "1.0"
    description = "Polymarket ETH 하락 12%+ & GLD 양전 & SETH 양전시 진입, +0.5% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or JAB_SETH)

    def check_entry(self, market: MarketData) -> bool:
        """진입 조건.

        1. Polymarket ETH 하락 기대 >= 12%
           (eth_up이 낮을수록 하락 기대 높음: 1 - eth_up >= 0.12)
        2. GLD >= +0.01%
        3. SETH >= 0.00% (양전)
        """
        if not market.poly:
            return False

        eth_up = market.poly.get("eth_up", 0.5)
        eth_down = 1.0 - eth_up
        if eth_down < self.params["poly_eth_down_min"]:
            return False

        gld = market.changes.get("GLD", 0)
        if gld < self.params["gld_min"]:
            return False

        seth = market.changes.get("SETH", 0)
        if seth < self.params["seth_min"]:
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        ticker = self.params.get("ticker", "SETH")
        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "SETH")

        if position is not None:
            current = market.prices.get(ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=ticker, size=1.0, target_pct=0,
                    reason=f"jab_seth target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={"pnl_pct": pnl_pct},
                )
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         f"holding SETH: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            return Signal(Action.SKIP, ticker, 0, 0, "conditions not met")

        eth_up = market.poly.get("eth_up", 0.5) if market.poly else 0.5
        return Signal(
            action=Action.BUY, ticker=ticker,
            size=self.params.get("size", 1.0),
            target_pct=self.params["target_pct"],
            reason=(
                f"jab_seth entry: eth_down={1 - eth_up:.0%}, "
                f"GLD={market.changes.get('GLD', 0):+.2f}%, "
                f"SETH={market.changes.get('SETH', 0):+.2f}%"
            ),
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        return errors

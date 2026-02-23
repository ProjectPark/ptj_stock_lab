"""
잽모드 TSLL 전략 (Jab-TSLL)
============================
프리마켓 시간대, 테슬라 본주 상승 + TSLL 과매도 괴리를 노리는 소액 단타.
200만원 이하 소액 한정.

출처: kakaotalk_trading_notes_2026-02-19.csv — (3) 잽 모드 TSLL 매수 <공격모드>
"""
from __future__ import annotations

from ..common.base import Action, ExitReason, MarketData, Position, Signal
from ..common.jab_base import JabBase
from ..common.params import JAB_TSLL
from ..common.registry import register


@register
class JabTSLL(JabBase):
    """잽모드 TSLL — TSLA 상승 + TSLL 과매도 역전 소액 단타."""

    name = "jab_tsll"
    version = "1.0"
    description = "Polymarket NDX 63%+ & TSLA 상승 & TSLL 과매도시 소액 진입, +1.0% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or JAB_TSLL)

    def check_entry(self, market: MarketData) -> bool:
        """모든 조건 ALL 충족.

        1. 시간 17:30 KST 이후
        2. Polymarket NASDAQ >= 63%
        3. GLD <= +0.3%
        4. TSLL <= -0.8%
        5. TSLA >= +0.5%
        6. QQQ >= +0.7%
        """
        if not self._is_in_window(market):
            return False

        if not market.poly:
            return False
        ndx_up = market.poly.get("ndx_up", 0)
        if ndx_up < self.params["poly_ndx_min"]:
            return False

        gld = market.changes.get("GLD", 0)
        if gld > self.params["gld_max"]:
            return False

        tsll = market.changes.get("TSLL", 0)
        if tsll > self.params["tsll_max"]:
            return False

        tsla = market.changes.get("TSLA", 0)
        if tsla < self.params["tsla_min"]:
            return False

        qqq = market.changes.get("QQQ", 0)
        if qqq < self.params["qqq_min"]:
            return False

        return True

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "TSLL")

        if position is not None:
            return self._make_exit_signal(market, position, ticker)

        if not self.check_entry(market):
            return Signal(Action.SKIP, ticker, 0, 0, "conditions not met")

        return Signal(
            action=Action.BUY, ticker=ticker,
            size=self.params.get("size", 1.0),
            target_pct=self.params["target_pct"],
            reason=(
                f"jab_tsll entry: poly_ndx={market.poly.get('ndx_up', 0):.0%}, "
                f"GLD={market.changes.get('GLD', 0):+.2f}%, "
                f"TSLL={market.changes.get('TSLL', 0):+.2f}%, "
                f"TSLA={market.changes.get('TSLA', 0):+.2f}%"
            ),
            metadata={
                "max_amount_krw": self.params.get("max_amount_krw", 2_000_000),
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        if self.params.get("max_amount_krw", 0) <= 0:
            errors.append("max_amount_krw must be positive")
        return errors

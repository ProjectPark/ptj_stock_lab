"""
VIX 급등 → GDXU 전략 (VIX-Gold)
==================================
공포 지수 급등시 금광 3배 레버리지로 안전자산 랠리 포착.
GDXU 목표 +10% 도달 후 매도 → 수익 전액 IAU 매수.

출처: kakaotalk_trading_notes_2026-02-19.csv — 3️⃣ VIX 급등 → GDXU 몰빵 전략
정정: GLD 대신 IAU 매수 (2026.2.18 18:57 정정)
"""
from __future__ import annotations

from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from .params import VIX_GOLD
from .registry import register


@register
class VixGold(BaseStrategy):
    """VIX 급등 → GDXU — 공포 지수 기반 금광 매수."""

    name = "vix_gold"
    version = "1.0"
    description = "VIX +10% & Polymarket 하락 30%시 GDXU 매수, +10% 매도 → IAU 재투자"

    def __init__(self, params: dict | None = None):
        super().__init__(params or VIX_GOLD)

    def check_entry(self, market: MarketData) -> bool:
        """진입 조건 ALL 충족.

        1. VIX 일간 변동 >= +10%
        2. Polymarket 전반적 하락 기대 >= 30%
        """
        vix_chg = market.changes.get("VIX", 0)
        if vix_chg < self.params["vix_spike_min"]:
            return False

        if not market.poly:
            return False

        # Polymarket 하락 기대: 주요 지표들의 하락 확률 평균
        btc_down = 1.0 - market.poly.get("btc_up", 0.5)
        ndx_down = 1.0 - market.poly.get("ndx_up", 0.5)
        avg_down = (btc_down + ndx_down) / 2
        if avg_down < self.params["poly_down_min"]:
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        ticker = self.params.get("ticker", "GDXU")
        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "GDXU")

        if position is not None:
            current = market.prices.get(ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=ticker, size=1.0, target_pct=0,
                    reason=f"vix_gold target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={
                        "pnl_pct": pnl_pct,
                        "reinvest_ticker": self.params["reinvest_ticker"],
                        "reinvest_action": "buy_all_profit_into_IAU",
                    },
                )
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         f"holding GDXU: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            return Signal(Action.SKIP, ticker, 0, 0, "VIX/Polymarket conditions not met")

        return Signal(
            action=Action.BUY, ticker=ticker, size=1.0,
            target_pct=self.params["target_pct"],
            reason=(
                f"vix_gold entry: VIX={market.changes.get('VIX', 0):+.1f}%, "
                f"poly_down={self._calc_poly_down(market):.0%}"
            ),
        )

    def _calc_poly_down(self, market: MarketData) -> float:
        if not market.poly:
            return 0
        btc_down = 1.0 - market.poly.get("btc_up", 0.5)
        ndx_down = 1.0 - market.poly.get("ndx_up", 0.5)
        return (btc_down + ndx_down) / 2

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("vix_spike_min", 0) <= 0:
            errors.append("vix_spike_min must be positive")
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        return errors

"""
숏포지션 전환 전략 (Short-Macro)
==================================
나스닥/S&P500 역대 최고가(ATH) 감지시 전면 숏 전환.
GDXU, IAU, GLD, 현금 제외 모든 롱 매도 → GDXU 100% 구축.

출처: kakaotalk_trading_notes_2026-02-19.csv — 4️⃣ 숏포지션
정정: 분할매도 하지않고 전액 매도 (2026.2.18 19:56 정정)
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import SHORT_MACRO
from ..common.registry import register


@register
class ShortMacro(BaseStrategy):
    """숏포지션 전환 — 지수 ATH시 전면 숏."""

    name = "short_macro"
    version = "1.1"
    description = "나스닥/S&P500 ATH시 롱 전체 매도, GDXU 100% 구축"

    def __init__(self, params: dict | None = None):
        super().__init__(params or SHORT_MACRO)

    def check_entry(self, market: MarketData) -> bool:
        """숏 전환 조건.

        확인 항목:
        1. 나스닥/S&P500 역대 최고가 (ATH)
        """
        if not market.history:
            return False

        macro = market.history.get("_macro", {})

        if not macro.get("index_ath", False):
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """GDXU +90% 도달시 청산."""
        if position.ticker == "GDXU":
            current = market.prices.get("GDXU", 0)
            if current <= 0 or position.avg_price <= 0:
                return False
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["action"]["gdxu_target_pct"]:
                return True

        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        # GDXU 포지션 보유중 → 청산 검토
        if position is not None and position.ticker == "GDXU":
            current = market.prices.get("GDXU", 0)
            if current <= 0:
                return Signal(Action.HOLD, "GDXU", 0, 0, "no price data")

            pnl_pct = (current - position.avg_price) / position.avg_price * 100

            if self.check_exit(market, position):
                return Signal(
                    action=Action.SELL, ticker="GDXU", size=1.0, target_pct=0,
                    reason=f"short_macro exit: target_hit (pnl={pnl_pct:.1f}%)",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={
                        "pnl_pct": pnl_pct,
                        "reinvest_ticker": self.params["action"]["reinvest_ticker"],
                        "exit_type": self.params["exit"]["exit_type"],
                    },
                )
            return Signal(Action.HOLD, "GDXU", 0,
                         self.params["action"]["gdxu_target_pct"],
                         f"holding GDXU: pnl={pnl_pct:.1f}%")

        # 숏 전환 조건 체크
        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "short macro conditions not met")

        keep = set(self.params["action"].get("sell_all_except", []))
        return Signal(
            action=Action.SELL, ticker="*",
            size=1.0, target_pct=0,
            reason="short_macro: sell all longs except GDXU/IAU/GLD/cash → build GDXU 100%",
            metadata={
                "keep_tickers": list(keep),
                "next_action": "buy_GDXU_100pct",
                "gdxu_target_pct": self.params["action"]["gdxu_target_pct"],
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("action", {}).get("gdxu_target_pct"):
            errors.append("gdxu_target_pct required")
        return errors

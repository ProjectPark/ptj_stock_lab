"""
하락장 방어 전략 (Bearish-Defense)
==================================
하락장 감지 시 BRKU 비중 고정.
Legacy: signals_v2.check_bearish_v2()
"""
from __future__ import annotations

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .registry import register


@register
class BearishDefenseStrategy(BaseStrategy):
    """하락장 감지 시 BRKU 포트폴리오 비중 고정."""

    name = "bearish_defense"
    version = "1.0"
    description = "하락장 모드에서 BRKU 비중 매수"

    def __init__(self, params: dict | None = None):
        defaults = {
            "brku_weight_pct": 10.0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def evaluate(self, mode: str) -> dict:
        """Legacy 호환 — 기존 check_bearish_v2()과 동일한 dict 반환.

        Returns
        -------
        dict
            mode, buy_brku, brku_weight_pct, message
        """
        brku_weight_pct = self.params.get("brku_weight_pct", 10.0)
        buy_brku = mode == "bearish"

        if buy_brku:
            message = f"하락장 감지 → BRKU {brku_weight_pct:.1f}% 매수, 나머지 현금 보유"
        else:
            message = f"시황 {mode} → BRKU 매수 불필요, 현금 보유"

        return {
            "mode": mode,
            "buy_brku": buy_brku,
            "brku_weight_pct": brku_weight_pct,
            "message": message,
        }

    def check_entry(self, market: MarketData) -> bool:
        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, "BRKU", 0, 0,
                      "use evaluate() for bearish defense signals")

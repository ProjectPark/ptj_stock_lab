"""
조건부 COIN 매수 전략 (Conditional-COIN)
========================================
ETHU/XXRP/SOLT 각각 +N% 이상이면 COIN 매수 시그널.
Legacy: signals_v2.check_conditional_coin_v2()
"""
from __future__ import annotations

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .registry import register


@register
class ConditionalCoinStrategy(BaseStrategy):
    """ETHU/XXRP/SOLT 트리거 기반 COIN 조건부 매수."""

    name = "conditional_coin"
    version = "1.0"
    description = "ETHU/XXRP/SOLT 각각 트리거 이상이면 COIN 매수"

    def __init__(self, params: dict | None = None):
        defaults = {
            "triggers": ["ETHU", "XXRP", "SOLT"],
            "target": "COIN",
            "trigger_pct": 3.0,
            "sell_profit_pct": 3.0,
            "sell_bearish_pct": 0.3,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def evaluate(self, changes: dict, mode: str = "normal") -> dict:
        """Legacy 호환 — 기존 check_conditional_coin_v2()과 동일한 dict 반환.

        Returns
        -------
        dict
            triggers, all_above_threshold, buy_signal, target, target_pct,
            trigger_avg_pct, sell_target_pct, message
        """
        triggers = self.params.get("triggers", ["ETHU", "XXRP", "SOLT"])
        target = self.params.get("target", "COIN")
        trigger_pct = self.params.get("trigger_pct", 3.0)
        sell_profit_pct = self.params.get("sell_profit_pct", 3.0)
        sell_bearish_pct = self.params.get("sell_bearish_pct", 0.3)

        trigger_info: dict[str, dict] = {}
        all_above = True
        pct_sum = 0.0

        for ticker in triggers:
            data = changes.get(ticker, {})
            pct = data.get("change_pct", 0.0)
            above = pct >= trigger_pct
            if not above:
                all_above = False
            pct_sum += pct
            trigger_info[ticker] = {"change_pct": pct, "above_threshold": above}

        trigger_avg = pct_sum / len(triggers) if triggers else 0.0

        target_data = changes.get(target, {})
        target_pct_val = target_data.get("change_pct", 0.0)

        sell_target = sell_bearish_pct if mode == "bearish" else sell_profit_pct

        if all_above:
            parts = ", ".join(
                f"{t} {trigger_info[t]['change_pct']:+.2f}%" for t in triggers
            )
            message = (
                f"{parts} 각각 ≥ +{trigger_pct:.1f}% → {target} 매수 "
                f"(매도 기준: 순수익 +{sell_target:.1f}%)"
            )
        else:
            below = [t for t in triggers if not trigger_info[t]["above_threshold"]]
            message = (
                f"{', '.join(below)} 미달 (기준 +{trigger_pct:.1f}%) "
                f"→ {target} 매수 보류"
            )

        return {
            "triggers": trigger_info,
            "all_above_threshold": all_above,
            "buy_signal": all_above,
            "target": target,
            "target_pct": target_pct_val,
            "trigger_avg_pct": round(trigger_avg, 4),
            "sell_target_pct": sell_target,
            "message": message,
        }

    def check_entry(self, market: MarketData) -> bool:
        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, self.params.get("target", "COIN"),
                      0, 0, "use evaluate() for conditional coin signals")

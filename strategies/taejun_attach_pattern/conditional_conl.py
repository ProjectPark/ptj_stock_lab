"""
조건부 CONL 매수 전략 (Conditional-CONL)
========================================
ETHU/XXRP/SOLT 각각 +N% 이상이면 CONL 매수 시그널.
Legacy: signals_v2.check_conditional_conl_v2()
"""
from __future__ import annotations

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .registry import register


@register
class ConditionalConlStrategy(BaseStrategy):
    """ETHU/XXRP/SOLT 트리거 기반 CONL 조건부 매수."""

    name = "conditional_conl"
    version = "1.0"
    description = "ETHU/XXRP/SOLT 각각 트리거 이상이면 CONL 매수"

    def __init__(self, params: dict | None = None):
        defaults = {
            "trigger_pct": 3.0,
            "sell_profit_pct": 2.8,
            "sell_avg_pct": 1.0,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def evaluate(self, changes: dict) -> dict:
        """Legacy 호환 — 기존 check_conditional_conl_v2()과 동일한 dict 반환.

        Returns
        -------
        dict
            triggers, all_above_threshold, trigger_avg_pct,
            sell_on_avg_drop, conl_pct, buy_signal, message
        """
        trigger_pct = self.params.get("trigger_pct", 3.0)
        sell_avg_pct = self.params.get("sell_avg_pct", 1.0)
        trigger_tickers = ["ETHU", "XXRP", "SOLT"]

        trigger_info: dict[str, dict] = {}
        all_above = True
        pct_sum = 0.0

        for ticker in trigger_tickers:
            data = changes.get(ticker, {})
            pct = data.get("change_pct", 0.0)
            above = pct >= trigger_pct
            if not above:
                all_above = False
            pct_sum += pct
            trigger_info[ticker] = {"change_pct": pct, "above_threshold": above}

        trigger_avg = pct_sum / len(trigger_tickers) if trigger_tickers else 0.0
        sell_on_avg_drop = trigger_avg < sell_avg_pct

        conl_data = changes.get("CONL", {})
        conl_pct = conl_data.get("change_pct", 0.0)

        if all_above:
            message = (
                f"ETHU/XXRP/SOLT 각각 ≥ +{trigger_pct:.1f}% → CONL 매수 "
                f"(평균 {trigger_avg:+.2f}%)"
            )
        else:
            below = [
                t for t in trigger_tickers
                if not trigger_info[t]["above_threshold"]
            ]
            message = (
                f"{', '.join(below)} 미달 (기준 +{trigger_pct:.1f}%) "
                f"→ CONL 매수 보류 (평균 {trigger_avg:+.2f}%)"
            )

        return {
            "triggers": trigger_info,
            "all_above_threshold": all_above,
            "trigger_avg_pct": round(trigger_avg, 4),
            "sell_on_avg_drop": sell_on_avg_drop,
            "conl_pct": conl_pct,
            "buy_signal": all_above,
            "message": message,
        }

    def check_entry(self, market: MarketData) -> bool:
        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, "CONL", 0, 0,
                      "use evaluate() for conditional conl signals")

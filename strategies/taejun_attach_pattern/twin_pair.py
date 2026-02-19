"""
쌍둥이 페어 갭 전략 (Twin-Pair)
================================
Legacy: signals_v2.check_twin_pairs_v2()
v3+: entry_threshold 2.2%로 변경 가능
"""
from __future__ import annotations

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .registry import register


@register
class TwinPairStrategy(BaseStrategy):
    """쌍둥이 페어 갭 분석 — multi-follow + 매도 기준."""

    name = "twin_pair"
    version = "1.0"
    description = "선행 종목과 후행 종목의 갭을 분석하여 진입/매도 시그널 생성"

    def __init__(self, params: dict | None = None):
        defaults = {
            "entry_threshold": 1.5,
            "sell_threshold": 0.9,
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def evaluate(self, changes: dict, pairs: dict) -> list[dict]:
        """Legacy 호환 — 기존 check_twin_pairs_vN()과 동일한 list[dict] 반환.

        Parameters
        ----------
        changes : dict
            {ticker: {"change_pct": float, ...}, ...}
        pairs : dict
            {key: {"lead": str, "follow": [str, ...], "label": str}, ...}

        Returns
        -------
        list[dict]
            각 follow별 시그널 (pair, lead, follow, lead_pct, follow_pct, gap, signal, message)
        """
        entry_threshold = self.params.get("entry_threshold", 1.5)
        sell_threshold = self.params.get("sell_threshold", 0.9)
        results: list[dict] = []

        for _key, pair_cfg in pairs.items():
            lead = pair_cfg["lead"]
            follows = pair_cfg["follow"]
            label = pair_cfg.get("label", _key)

            lead_data = changes.get(lead, {})
            lead_pct = lead_data.get("change_pct", 0.0)

            for follow_ticker in follows:
                follow_data = changes.get(follow_ticker, {})
                follow_pct = follow_data.get("change_pct", 0.0)
                gap = lead_pct - follow_pct

                if gap <= sell_threshold and follow_pct > 0:
                    signal = "SELL"
                    message = (
                        f"{label} | {follow_ticker} 갭 {gap:+.2f}% ≤ {sell_threshold}% "
                        f"& 양봉 → 매도"
                    )
                elif gap >= entry_threshold:
                    signal = "ENTRY"
                    message = (
                        f"{label} | {lead} +{lead_pct:.2f}% vs "
                        f"{follow_ticker} +{follow_pct:.2f}% (갭 {gap:+.2f}%) → 진입"
                    )
                else:
                    signal = "HOLD"
                    message = (
                        f"{label} | {lead} {lead_pct:+.2f}% vs "
                        f"{follow_ticker} {follow_pct:+.2f}% (갭 {gap:+.2f}%) → 대기"
                    )

                results.append({
                    "pair": label,
                    "lead": lead,
                    "follow": follow_ticker,
                    "lead_pct": lead_pct,
                    "follow_pct": follow_pct,
                    "gap": gap,
                    "signal": signal,
                    "message": message,
                })

        return results

    def check_entry(self, market: MarketData) -> bool:
        return False  # 단일 종목 진입 판단은 evaluate()에서 처리

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False  # evaluate()에서 SELL 시그널로 처리

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, "", 0, 0, "use evaluate() for twin pair signals")

"""
부동산 유동성 감소 이벤트 전략 (REIT-Risk)
=============================================
리츠 상승 + 탐욕지수 = 레버리지 매매 90일 중단.
GDXU만 예외.

출처: kakaotalk_trading_notes_2026-02-19.csv — 상승장 매도 계획 부동산 유동성 감소 이벤트
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .params import REIT_RISK
from .registry import register


@register
class ReitRisk(BaseStrategy):
    """부동산 유동성 감소 — 리츠 과열시 레버리지 매매 중단."""

    name = "reit_risk"
    version = "1.1"
    description = "리츠 7일 +0.1% & 탐욕지수 75+시 레버리지 90일 금지"

    def __init__(self, params: dict | None = None):
        super().__init__(params or REIT_RISK)
        self._ban_until: datetime | None = None

    @property
    def is_ban_active(self) -> bool:
        """레버리지 매매 금지가 활성 상태인지."""
        if self._ban_until is None:
            return False
        return datetime.now() < self._ban_until

    def check_entry(self, market: MarketData) -> bool:
        """리스크 이벤트 감지 조건.

        1. 리츠 3종목 7일 연속 평균 +0.1%
        2. 탐욕지수 >= 75
        """
        if not market.history:
            return False

        macro = market.history.get("_macro", {})
        conditions = self.params.get("conditions", {})

        # 리츠 평균 변동률 (당일 — 7일 연속은 외부에서 검증)
        reits = conditions.get("reits", [])
        reit_changes = [market.changes.get(r, 0) for r in reits]
        if not reit_changes:
            return False
        reit_avg = sum(reit_changes) / len(reit_changes)
        if reit_avg < conditions.get("reits_avg_min", 0.1):
            return False

        # 리츠 7일 연속 상승 여부 (외부 데이터)
        if not macro.get("reits_7d_positive", False):
            return False

        # 탐욕지수
        fear_greed = macro.get("fear_greed_index", 50)
        if fear_greed < conditions.get("fear_greed_min", 75):
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        if self.is_ban_active:
            days_left = (self._ban_until - market.time).days if self._ban_until else 0
            return Signal(
                action=Action.SKIP, ticker="", size=0, target_pct=0,
                reason=f"reit_risk: leverage ban active ({days_left}d remaining)",
                metadata={"ban_until": str(self._ban_until), "ban_except": ["GDXU"]},
            )

        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "reit risk conditions not met")

        ban_days = self.params.get("action", {}).get("ban_days", 90)
        self._ban_until = market.time + timedelta(days=ban_days)
        ban_except = self.params.get("action", {}).get("ban_except", ["GDXU"])

        return Signal(
            action=Action.SELL,
            ticker="*leveraged",
            size=1.0, target_pct=0,
            reason=f"reit_risk triggered: leverage ban for {ban_days}d (except {ban_except})",
            metadata={
                "ban_days": ban_days,
                "ban_until": str(self._ban_until),
                "ban_except": ban_except,
                "trigger": "reits_7d_up + fear_greed_75+",
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("conditions", {}).get("reits"):
            errors.append("reits list required")
        return errors

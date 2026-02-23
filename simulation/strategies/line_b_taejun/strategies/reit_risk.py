"""
부동산 유동성 감소 이벤트 전략 (REIT-Risk)
=============================================
리츠 상승 + 탐욕지수 = 레버리지 매매 90일 중단.
GDXU만 예외.

출처: kakaotalk_trading_notes_2026-02-19.csv — 상승장 매도 계획 부동산 유동성 감소 이벤트
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..common.base import Action, BaseStrategy, MarketData, Position, Signal
from ..common.params import REIT_RISK
from ..common.registry import register


@register
class ReitRisk(BaseStrategy):
    """부동산 유동성 감소 — 리츠 과열시 레버리지 매매 중단."""

    name = "reit_risk"
    version = "1.2"
    description = "REIT_MIX(VNQ+KR리츠) 7일 수익률 1%+ → 레버리지 90일 금지 + 조심모드"

    def __init__(self, params: dict | None = None):
        super().__init__(params or REIT_RISK)
        self._ban_until: datetime | None = None
        self._reit_mix: float = 0.0
        self._last_alarm: bool = False

    def is_ban_active_at(self, market_time: datetime) -> bool:
        """레버리지 매매 금지가 특정 시각에 활성 상태인지."""
        if self._ban_until is None:
            return False
        return market_time < self._ban_until

    @property
    def is_ban_active(self) -> bool:
        """레버리지 매매 금지가 활성 상태인지. (하위 호환용 — 백테스트에서는 is_ban_active_at 사용)"""
        if self._ban_until is None:
            return False
        return datetime.now() < self._ban_until

    def check_entry(self, market: MarketData) -> bool:
        """리스크 이벤트 감지 조건 (MT_VNQ3 §9 변경).

        REIT_MIX = VNQ + (available KR리츠) 평균 계산.
        전부 결측 → UNKNOWN → BUY_STOP (metadata.alarm=True)
        일부 결측 → 나머지로 평균 (P-16 잠정 처리)
        """
        if not market.history:
            return False

        conditions = self.params.get("conditions", {})
        reits = conditions.get("reits", [])  # Primary: ["VNQ"]
        kr_aux = conditions.get("reits_kr_aux", [])  # KR auxiliary
        min_return = conditions.get("reits_7d_return_min", 1.0)

        if not reits:
            return False

        # Collect all available 7d returns
        all_returns = []
        for reit in reits + kr_aux:
            reit_hist = market.history.get(reit, {})
            ret_7d = reit_hist.get("return_7d")
            if ret_7d is not None:
                all_returns.append(ret_7d)

        if not all_returns:
            # 전부 결측 → UNKNOWN → BUY_STOP
            self._last_alarm = True
            return True  # triggers signal generation which will handle alarm

        # REIT_MIX = average of available returns
        reit_mix = sum(all_returns) / len(all_returns)
        self._reit_mix = reit_mix
        return reit_mix >= min_return

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        if self.is_ban_active_at(market.time):
            days_left = (self._ban_until - market.time).days if self._ban_until else 0
            return Signal(
                action=Action.SKIP, ticker="", size=0, target_pct=0,
                reason=f"reit_risk: leverage ban active ({days_left}d remaining)",
                metadata={"ban_until": str(self._ban_until), "ban_except": ["GDXU"]},
            )

        # All data missing → alarm + BUY_STOP
        if getattr(self, '_last_alarm', False):
            self._last_alarm = False
            return Signal(
                action=Action.SKIP, ticker="", size=0, target_pct=0,
                reason="reit_risk: REIT_MIX all data missing → BUY_STOP",
                metadata={"alarm": True, "buy_stop": True},
            )

        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "reit risk conditions not met")

        reit_mix = getattr(self, '_reit_mix', 0.0)

        # Cautious mode check: REIT_MIX +1% → cautious_mode=True
        cautious = reit_mix >= 1.0

        ban_days = self.params.get("action", {}).get("ban_days", 90)
        self._ban_until = market.time + timedelta(days=ban_days)
        ban_except = self.params.get("action", {}).get("ban_except", ["GDXU"])
        cautious_cfg = self.params.get("cautious_mode", {})
        attack_leverage_pct = cautious_cfg.get("attack_leverage_pct", 50)

        return Signal(
            action=Action.SELL,
            ticker="*leveraged",
            size=1.0, target_pct=0,
            reason=f"reit_risk triggered: leverage ban for {ban_days}d (except {ban_except})",
            metadata={
                "ban_days": ban_days,
                "ban_until": str(self._ban_until),
                "ban_except": ban_except,
                "trigger": "reit_mix_7d_return",
                "reit_mix": reit_mix,
                "cautious_mode": cautious,
                "attack_leverage_pct": attack_leverage_pct,
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("conditions", {}).get("reits"):
            errors.append("reits list required")
        return errors

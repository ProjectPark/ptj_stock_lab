"""
이머전시 모드 (Emergency Mode)
================================
Polymarket 30pp 이상 급변시 발동.
- 기본: 수익중 포지션 즉시 매도
- 서브 모드: BTC 급등 → BITU, NASDAQ 급등 → SOXL, NASDAQ 급락 → SOXS
- 목표: 0.9% net (수수료 제외)

출처: kakaotalk_trading_notes_2026-02-19.csv — 이머전시 모드
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import EMERGENCY_MODE
from ..common.registry import register


@register
class EmergencyMode(BaseStrategy):
    """이머전시 모드 — Polymarket 30pp+ 급변 대응."""

    name = "emergency_mode"
    version = "1.0"
    description = "Polymarket 30pp+ 변동시 수익 포지션 매도 + 방향성 매수"

    def __init__(self, params: dict | None = None):
        super().__init__(params or EMERGENCY_MODE)
        self._triggered: bool = False
        self._trigger_key: str = ""     # 어떤 poly 키가 트리거했는지
        self._trigger_swing: float = 0  # 트리거 변동폭 (pp)

    # ------------------------------------------------------------------
    # Polymarket 급변 감지
    # ------------------------------------------------------------------

    def _detect_poly_swing(self, market: MarketData) -> tuple[str, float]:
        """Polymarket 이전값 대비 급변 감지.

        Returns (key, swing_pp) — 가장 큰 변동을 보인 키.
        swing_pp > 0이면 상승, < 0이면 하락.
        """
        if not market.poly or not market.poly_prev:
            return "", 0.0

        max_key = ""
        max_swing = 0.0

        for key, current in market.poly.items():
            prev = market.poly_prev.get(key, current)
            swing = (current - prev) * 100  # pp 단위
            if abs(swing) > abs(max_swing):
                max_swing = swing
                max_key = key

        return max_key, max_swing

    # ------------------------------------------------------------------
    # 서브 모드 매칭
    # ------------------------------------------------------------------

    def _match_sub_mode(self, trigger_key: str, swing: float
                        ) -> dict | None:
        """트리거된 poly 키와 변동 방향으로 서브 모드를 매칭한다."""
        sub_modes = {
            "btc_surge": self.params.get("btc_surge", {}),
            "ndx_bull": self.params.get("ndx_bull", {}),
            "ndx_bear": self.params.get("ndx_bear", {}),
        }

        for mode_name, cfg in sub_modes.items():
            if not cfg:
                continue
            poly_key = cfg.get("poly_key", "")
            direction = cfg.get("direction", "up")
            combined_min = cfg.get("combined_swing_min", 30.0)

            if trigger_key != poly_key:
                continue

            if direction == "up" and swing >= combined_min:
                return cfg
            elif direction == "down" and swing <= -combined_min:
                return cfg

        return None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """Polymarket 30pp 이상 변동 감지."""
        key, swing = self._detect_poly_swing(market)
        min_swing = self.params.get("poly_swing_min", 30.0)
        if abs(swing) >= min_swing:
            self._triggered = True
            self._trigger_key = key
            self._trigger_swing = swing
            return True
        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """이머전시 모드에서는 수익중 포지션을 즉시 매도."""
        current = market.prices.get(position.ticker, 0)
        pnl = position.pnl_pct(current)
        if pnl is None:
            return False
        return pnl > 0

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        # 기존 포지션 수익 매도
        if position is not None:
            current = market.prices.get(position.ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, position.ticker, 0, 0, "no price data")
            pnl_pct = position.pnl_pct(current) or 0.0
            if pnl_pct > 0:
                return Signal(
                    action=Action.SELL,
                    ticker=position.ticker,
                    size=1.0,
                    target_pct=0,
                    reason=f"emergency_sell: {position.ticker} pnl={pnl_pct:.2f}%",
                    exit_reason=ExitReason.CONDITION_BREAK,
                    metadata={"emergency_mode": True, "pnl_pct": pnl_pct},
                )
            return Signal(Action.HOLD, position.ticker, 0, 0,
                         f"emergency: holding {position.ticker} (loss={pnl_pct:.2f}%)")

        # 진입 검토
        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "emergency conditions not met")

        # 서브 모드 매수 시그널
        sub = self._match_sub_mode(self._trigger_key, self._trigger_swing)
        if sub:
            ticker = sub.get("ticker", "")
            target_net = self.params.get("target_net_pct", 0.9)
            return Signal(
                action=Action.BUY,
                ticker=ticker,
                size=1.0,
                target_pct=target_net + 0.5,  # gross ≈ net + 0.5% 수수료
                reason=f"emergency_buy: {ticker} ({self._trigger_key} "
                       f"swing={self._trigger_swing:+.1f}pp)",
                metadata={
                    "emergency_mode": True,
                    "trigger_key": self._trigger_key,
                    "trigger_swing": self._trigger_swing,
                    "target_net_pct": target_net,
                },
            )

        # 서브 모드 미매칭 — 기본 모드 (수익 매도만, 신규 매수 없음)
        return Signal(
            Action.SKIP, "", 0, 0,
            f"emergency_base: {self._trigger_key} swing={self._trigger_swing:+.1f}pp "
            f"(no sub-mode matched)",
            metadata={"emergency_mode": True, "base_only": True},
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("poly_swing_min", 0) <= 0:
            errors.append("poly_swing_min must be positive")
        return errors

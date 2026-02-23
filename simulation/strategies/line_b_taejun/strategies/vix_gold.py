"""
VIX 방어 전략 (VIX Defense — 13-7절)
======================================
Unix(VIX) +10% 이상 급등 시 IAU 40% + GDXU 30% 동시 매수.

매수 조건:
  VIX 일간 변동률 >= +10%

GDXU 전술 운용 (2-3 거래일):
  - 최소 2거래일 보유, 수익 구간에서 청산 가능
  - 3거래일 경과 시 강제 청산 (수익/손실 무관)
  - GDXU -12% 발생 → 강제 청산 + IAU 매수 금지 40거래일

IAU 운용:
  - 진입가 대비 -5% 손절
  - GDXU -12% 후 40거래일 매수 금지 쿨다운

generate_signals() 오버라이드:
  VIX 방어 발동 시 IAU BUY + GDXU BUY 두 개 시그널을 동시 반환.
"""
from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING, Any

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import VIX_GOLD
from ..common.registry import register

if TYPE_CHECKING:
    pass


@register
class VixGold(BaseStrategy):
    """VIX +10% → IAU 40% + GDXU 30% 동시 매수.

    v5 13-7절: 공포 지수 급등 시 IAU(금 ETF) + GDXU(금광 3x) 동시 방어.
    GDXU는 2-3 거래일 전술 운용, IAU는 손절 -5%.
    GDXU -12% 시 IAU 매수 40거래일 금지 (쿨다운).
    """

    name = "vix_gold"
    version = "2.0"
    description = "VIX +10% → IAU 40%+GDXU 30% 동시 방어, GDXU 2-3일 전술, IAU 쿨다운"

    def __init__(self, params: dict | None = None):
        super().__init__(params or VIX_GOLD)

        # GDXU 보유 기간 추적 (거래일 카운트)
        self._gdxu_entry_date: date | None = None
        self._gdxu_last_date: date | None = None
        self._gdxu_hold_days: int = 0           # 진입 이후 거래일 수

        # IAU 쿨다운 (GDXU -12% 발생 후 40거래일 금지)
        self._iau_cooldown_remaining: int = 0   # 잔여 금지 거래일
        self._iau_cooldown_last_date: date | None = None

        # 일일 카운터 통합 (하루 1회만 갱신)
        self._counter_last_date: date | None = None

    # ------------------------------------------------------------------
    # 내부: 일일 카운터 갱신 (generate_signals 시작 시 1회 호출)
    # ------------------------------------------------------------------

    def _advance_day_if_new(self, today: date) -> None:
        """새 거래일이면 GDXU 보유일과 IAU 쿨다운을 1일 진행한다."""
        if self._counter_last_date == today:
            return
        # 새 거래일
        if self._counter_last_date is not None:
            if self._gdxu_entry_date is not None:
                self._gdxu_hold_days += 1
            if self._iau_cooldown_remaining > 0:
                self._iau_cooldown_remaining -= 1
        self._counter_last_date = today

    # ------------------------------------------------------------------
    # 진입 조건
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """VIX 일간 변동률 >= +10%."""
        vix_chg = market.changes.get("VIX", 0.0)
        return vix_chg >= self.params.get("vix_spike_min", 10.0)

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """IAU/GDXU 개별 청산 조건 체크 (단일 포지션 호환용)."""
        if position.ticker == "GDXU":
            return self._should_exit_gdxu(market, position)
        if position.ticker == "IAU":
            return self._should_exit_iau(market, position)
        return False

    def _should_exit_gdxu(self, market: MarketData, position: Position) -> bool:
        current = market.prices.get("GDXU", 0.0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        gdxu_min = self.params.get("gdxu_min_days", 2)
        gdxu_max = self.params.get("gdxu_max_days", 3)
        cooldown_trigger = self.params.get("gdxu_cooldown_trigger", -12.0)
        return (
            pnl_pct <= cooldown_trigger
            or self._gdxu_hold_days >= gdxu_max
            or (self._gdxu_hold_days >= gdxu_min and pnl_pct > 0)
        )

    def _should_exit_iau(self, market: MarketData, position: Position) -> bool:
        current = market.prices.get("IAU", 0.0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct <= self.params.get("iau_stop_pct", -5.0)

    # ------------------------------------------------------------------
    # 복수 시그널 생성 (메인 인터페이스)
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        market: MarketData,
        positions: "dict[str, Position] | None" = None,
    ) -> list[Signal]:
        """IAU + GDXU 동시 진입/청산 시그널 반환.

        반환 순서: SELL 우선, 그 다음 BUY.
        """
        positions = positions or {}
        today = market.time.date()
        self._advance_day_if_new(today)

        gdxu_pos = positions.get("GDXU")
        iau_pos = positions.get("IAU")

        # strategy_name 필터
        gdxu_pos = gdxu_pos if (gdxu_pos and gdxu_pos.strategy_name == self.name) else None
        iau_pos = iau_pos if (iau_pos and iau_pos.strategy_name == self.name) else None

        signals: list[Signal] = []

        # SELL 시그널 먼저
        if gdxu_pos:
            sig = self._gdxu_exit_signal(market, gdxu_pos)
            if sig.action == Action.SELL:
                signals.append(sig)

        if iau_pos:
            sig = self._iau_exit_signal(market, iau_pos)
            if sig.action == Action.SELL:
                signals.append(sig)

        # BUY: 포지션 없을 때만 진입 검토
        if not gdxu_pos and not iau_pos and self.check_entry(market):
            signals.extend(self._entry_signals(market))

        return signals

    def generate_signal(
        self, market: MarketData, position: Position | None = None
    ) -> Signal:
        """단일 포지션 호환 래퍼. generate_signals() 사용 권장."""
        today = market.time.date()
        self._advance_day_if_new(today)

        if position:
            if position.ticker == "GDXU":
                return self._gdxu_exit_signal(market, position)
            return self._iau_exit_signal(market, position)

        if not self.check_entry(market):
            return Signal(Action.SKIP, "GDXU", 0, 0, "vix_gold: VIX conditions not met")

        entry = self._entry_signals(market)
        return entry[0] if entry else Signal(
            Action.SKIP, "GDXU", 0, 0, "vix_gold: no entry signal"
        )

    # ------------------------------------------------------------------
    # 내부: 진입 시그널 생성
    # ------------------------------------------------------------------

    def _entry_signals(self, market: MarketData) -> list[Signal]:
        """VIX 방어 발동 → IAU + GDXU BUY 시그널 동시 생성."""
        vix_chg = market.changes.get("VIX", 0.0)
        iau_pct = self.params.get("iau_pct", 0.40)
        gdxu_pct = self.params.get("gdxu_pct", 0.30)
        gdxu_min = self.params.get("gdxu_min_days", 2)
        gdxu_max = self.params.get("gdxu_max_days", 3)

        signals: list[Signal] = []

        # GDXU 30% 매수
        self._gdxu_entry_date = market.time.date()
        self._gdxu_last_date = market.time.date()
        self._gdxu_hold_days = 0
        signals.append(Signal(
            action=Action.BUY,
            ticker="GDXU",
            size=gdxu_pct,
            target_pct=0,
            reason=(
                f"vix_gold: VIX {vix_chg:+.1f}% → GDXU "
                f"{int(gdxu_pct * 100)}% 진입 ({gdxu_min}-{gdxu_max}거래일 전술)"
            ),
            metadata={
                "vix_defense": True,
                "vix_chg": vix_chg,
                "gdxu_min_days": gdxu_min,
                "gdxu_max_days": gdxu_max,
            },
        ))

        # IAU 40% 매수 (쿨다운 중이면 스킵)
        if self._iau_cooldown_remaining > 0:
            signals.append(Signal(
                Action.SKIP, "IAU", 0, 0,
                f"vix_gold: IAU 쿨다운 잔여 {self._iau_cooldown_remaining}거래일 — 매수 금지",
                metadata={"iau_cooldown_remaining": self._iau_cooldown_remaining},
            ))
        else:
            signals.append(Signal(
                action=Action.BUY,
                ticker="IAU",
                size=iau_pct,
                target_pct=0,
                reason=(
                    f"vix_gold: VIX {vix_chg:+.1f}% → IAU "
                    f"{int(iau_pct * 100)}% 진입"
                ),
                metadata={
                    "vix_defense": True,
                    "vix_chg": vix_chg,
                    "iau_stop_pct": self.params.get("iau_stop_pct", -5.0),
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # 내부: GDXU 청산 시그널
    # ------------------------------------------------------------------

    def _gdxu_exit_signal(self, market: MarketData, position: Position) -> Signal:
        current = market.prices.get("GDXU", 0.0)
        if current <= 0:
            return Signal(Action.HOLD, "GDXU", 0, 0, "vix_gold: no GDXU price")

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        gdxu_min = self.params.get("gdxu_min_days", 2)
        gdxu_max = self.params.get("gdxu_max_days", 3)
        cooldown_trigger = self.params.get("gdxu_cooldown_trigger", -12.0)
        cooldown_days = self.params.get("iau_cooldown_days", 40)

        # GDXU -12% → 강제 청산 + IAU 쿨다운 시작
        if pnl_pct <= cooldown_trigger:
            self._iau_cooldown_remaining = cooldown_days
            self._iau_cooldown_last_date = market.time.date()
            self._gdxu_entry_date = None
            self._gdxu_hold_days = 0
            return Signal(
                action=Action.SELL,
                ticker="GDXU",
                size=1.0,
                target_pct=0,
                reason=(
                    f"vix_gold: GDXU {pnl_pct:+.1f}% <= {cooldown_trigger}% → "
                    f"IAU 쿨다운 {cooldown_days}거래일 시작"
                ),
                exit_reason=ExitReason.STOP_LOSS,
                metadata={
                    "pnl_pct": pnl_pct,
                    "iau_cooldown_started": cooldown_days,
                },
            )

        # 3거래일 → 강제 청산
        if self._gdxu_hold_days >= gdxu_max:
            self._gdxu_entry_date = None
            self._gdxu_hold_days = 0
            return Signal(
                action=Action.SELL,
                ticker="GDXU",
                size=1.0,
                target_pct=0,
                reason=f"vix_gold: GDXU {gdxu_max}거래일 만기 강제 청산 pnl={pnl_pct:+.1f}%",
                exit_reason=ExitReason.TIME_LIMIT,
                metadata={"hold_days": self._gdxu_hold_days, "pnl_pct": pnl_pct},
            )

        # 2거래일 이상 + 수익 → 청산 가능
        if self._gdxu_hold_days >= gdxu_min and pnl_pct > 0:
            self._gdxu_entry_date = None
            self._gdxu_hold_days = 0
            return Signal(
                action=Action.SELL,
                ticker="GDXU",
                size=1.0,
                target_pct=0,
                reason=f"vix_gold: GDXU {self._gdxu_hold_days}거래일+ 수익 청산 {pnl_pct:+.1f}%",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={"hold_days": self._gdxu_hold_days, "pnl_pct": pnl_pct},
            )

        return Signal(
            Action.HOLD, "GDXU", 0, 0,
            f"vix_gold: GDXU {self._gdxu_hold_days}거래일 보유 pnl={pnl_pct:+.1f}%",
        )

    # ------------------------------------------------------------------
    # 내부: IAU 청산 시그널
    # ------------------------------------------------------------------

    def _iau_exit_signal(self, market: MarketData, position: Position) -> Signal:
        current = market.prices.get("IAU", 0.0)
        if current <= 0:
            return Signal(Action.HOLD, "IAU", 0, 0, "vix_gold: no IAU price")

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        stop_pct = self.params.get("iau_stop_pct", -5.0)

        if pnl_pct <= stop_pct:
            return Signal(
                action=Action.SELL,
                ticker="IAU",
                size=1.0,
                target_pct=0,
                reason=f"vix_gold: IAU 손절 {pnl_pct:+.1f}% <= {stop_pct}%",
                exit_reason=ExitReason.STOP_LOSS,
                metadata={"pnl_pct": pnl_pct},
            )

        return Signal(
            Action.HOLD, "IAU", 0, 0,
            f"vix_gold: IAU 보유 중 pnl={pnl_pct:+.1f}%",
        )

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def iau_cooldown_remaining(self) -> int:
        """IAU 매수 금지 잔여 거래일."""
        return self._iau_cooldown_remaining

    @property
    def gdxu_hold_days(self) -> int:
        """GDXU 현재 보유 거래일 수."""
        return self._gdxu_hold_days

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("vix_spike_min", 0) <= 0:
            errors.append("vix_spike_min must be positive")
        total = self.params.get("iau_pct", 0) + self.params.get("gdxu_pct", 0)
        if total > 1.0:
            errors.append(f"iau_pct + gdxu_pct must not exceed 1.0 (got {total:.2f})")
        return errors

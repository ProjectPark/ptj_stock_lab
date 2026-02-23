"""
급등 스윙 전략 (Swing Mode)
==========================================
v5 매매 규칙 13절 — 대상 종목 +15% 급등 시 장기 스윙 전환.

상태 머신:
  INACTIVE → PHASE1(3m, 90%) → PHASE2(5m, IAU 70%) → INACTIVE

13-1. 진입: 대상 종목 +15% 일간 상승
13-2. 전환 청산: 기존 보유분 전량 매도
13-3. 2단계 구조: 1단계(급등 90%, 3개월) → 2단계(IAU 70%, 5개월)
13-5. 금지: 스윙 중 일반 매매 전면 차단
13-6. 매수·매도 1회 한정
13-8. VIX 방어모드 우선 (VIX +10% > 스윙)
"""
from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Any

from ..common.base import Action, ExitReason, MarketData, Position, Signal


class SwingPhase(Enum):
    INACTIVE = "inactive"
    PHASE1 = "phase1"   # 급등 종목 90%, 3개월
    PHASE2 = "phase2"   # IAU 70%, 5개월


class SwingModeManager:
    """급등 스윙 모드 관리자 — 포트폴리오 레벨 상태 머신.

    Note: This is NOT a BaseStrategy subclass. It's a portfolio-level mode manager.
    Do NOT use @register decorator.
    """

    def __init__(self, params: dict | None = None):
        from ..common.params import SWING_MODE
        self.params = params or SWING_MODE

        # State
        self._phase: SwingPhase = SwingPhase.INACTIVE
        self._phase1_ticker: str = ""
        self._phase1_entry_date: date | None = None
        self._phase1_entry_price: float = 0.0
        self._phase2_entry_date: date | None = None
        self._phase2_entry_price: float = 0.0

        # 1회 매수/매도 추적
        self._phase1_bought: bool = False
        self._phase1_sold: bool = False
        self._phase2_bought: bool = False
        self._phase2_sold: bool = False

    @property
    def phase(self) -> SwingPhase:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._phase != SwingPhase.INACTIVE

    @property
    def is_trading_blocked(self) -> bool:
        """일반 매매 차단 여부. 스윙 모드 중 True."""
        return self.is_active

    def check_trigger(self, market: MarketData) -> str | None:
        """대상 종목 중 +15% 급등 종목을 감지한다.

        Returns ticker or None.

        복수 급등 시 선택 기준 (13-1):
        1. 급등률 최대
        2. 거래량 최대
        3. 시가총액 (fallback: ticker 순)
        """
        if self.is_active:
            return None  # 이미 스윙 중

        trigger_pct = self.params.get("trigger_pct", 15.0)
        tickers = self.params.get("tickers", [])

        candidates = []
        for ticker in tickers:
            chg = market.changes.get(ticker, 0.0)
            if chg >= trigger_pct:
                vol = (market.volumes or {}).get(ticker, 0.0)
                candidates.append((ticker, chg, vol))

        if not candidates:
            return None

        # Sort: highest change first, then volume, then alphabetical
        candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return candidates[0][0]

    def generate_signals(
        self, market: MarketData, positions: dict[str, Position] | None = None
    ) -> list[Signal]:
        """스윙 모드 시그널 생성.

        Returns: SELL signals first (clearing), then BUY signals.
        """
        positions = positions or {}
        signals: list[Signal] = []

        if self._phase == SwingPhase.INACTIVE:
            trigger_ticker = self.check_trigger(market)
            if trigger_ticker:
                signals = self._enter_swing(market, positions, trigger_ticker)
            return signals

        if self._phase == SwingPhase.PHASE1:
            return self._phase1_signals(market, positions)

        if self._phase == SwingPhase.PHASE2:
            return self._phase2_signals(market, positions)

        return signals

    def _enter_swing(
        self, market: MarketData, positions: dict[str, Position], trigger_ticker: str
    ) -> list[Signal]:
        """스윙 모드 진입 — 기존 포지션 전량 매도 + 스윙 종목 매수."""
        signals: list[Signal] = []
        chg = market.changes.get(trigger_ticker, 0.0)

        # 13-2: 기존 보유분 전량 매도
        for ticker, pos in positions.items():
            if pos.qty > 0:
                signals.append(Signal(
                    action=Action.SELL,
                    ticker=ticker,
                    size=1.0,
                    target_pct=0,
                    reason=f"swing_mode: 스윙 전환 청산 — {trigger_ticker} {chg:+.1f}% 급등",
                    exit_reason=ExitReason.CONDITION_BREAK,
                    metadata={"swing_clearing": True},
                ))

        # Phase 1 진입
        self._phase = SwingPhase.PHASE1
        self._phase1_ticker = trigger_ticker
        self._phase1_entry_date = market.time.date()
        self._phase1_entry_price = market.prices.get(trigger_ticker, 0.0)
        self._phase1_bought = True
        self._phase1_sold = False

        phase1_pct = self.params.get("phase1_pct", 0.90)
        signals.append(Signal(
            action=Action.BUY,
            ticker=trigger_ticker,
            size=phase1_pct,
            target_pct=0,
            reason=(
                f"swing_mode: {trigger_ticker} {chg:+.1f}% 급등 → "
                f"1단계 {int(phase1_pct*100)}% 진입 (3개월)"
            ),
            metadata={
                "swing_phase": "phase1",
                "trigger_chg": chg,
                "phase1_months": self.params.get("phase1_months", 3),
                "stop_pct": self.params.get("phase1_stop_pct", -15.0),
            },
        ))

        return signals

    def _phase1_signals(
        self, market: MarketData, positions: dict[str, Position]
    ) -> list[Signal]:
        """1단계: 급등 종목 보유 — 만기/손절 체크."""
        ticker = self._phase1_ticker
        pos = positions.get(ticker)

        if not pos or pos.qty <= 0:
            # Position already sold externally
            return self._transition_to_phase2(market)

        current = market.prices.get(ticker, 0.0)
        if current <= 0 or self._phase1_entry_price <= 0:
            return []

        pnl_pct = (current - self._phase1_entry_price) / self._phase1_entry_price * 100
        today = market.time.date()

        # 매도 조건 체크
        months = self.params.get("phase1_months", 3)
        stop_pct = self.params.get("phase1_stop_pct", -15.0)

        # 3개월 만기 (약 63 거래일 ≈ 90 calendar days)
        if self._phase1_entry_date:
            days_held = (today - self._phase1_entry_date).days
            if days_held >= months * 30:  # approximate
                return self._sell_phase1(market, pos, pnl_pct, "3개월 만기")

        # 진입가 대비 -15% 손절
        if pnl_pct <= stop_pct:
            return self._sell_phase1(market, pos, pnl_pct, f"손절 {pnl_pct:+.1f}%")

        # ATR 손절 (if OHLCV available)
        atr_mul = self.params.get("phase1_atr_multiplier", 1.5)
        ohlcv = market.ohlcv or {}
        ticker_df = ohlcv.get(ticker)
        if ticker_df is not None and len(ticker_df) >= 15:
            try:
                atr = self._calc_atr(ticker_df)
                atr_stop = self._phase1_entry_price - atr_mul * atr
                if current <= atr_stop:
                    return self._sell_phase1(
                        market, pos, pnl_pct,
                        f"ATR 손절 (price {current:.2f} <= {atr_stop:.2f})"
                    )
            except Exception:
                pass  # ATR calc failed, skip

        return []  # HOLD

    def _sell_phase1(
        self, market: MarketData, pos: Position, pnl_pct: float, reason: str
    ) -> list[Signal]:
        """1단계 매도 → 2단계 전환."""
        self._phase1_sold = True
        signals = [Signal(
            action=Action.SELL,
            ticker=self._phase1_ticker,
            size=1.0,
            target_pct=0,
            reason=f"swing_mode: 1단계 매도 — {reason}",
            exit_reason=ExitReason.TIME_LIMIT if "만기" in reason else ExitReason.STOP_LOSS,
            metadata={"swing_phase": "phase1", "pnl_pct": pnl_pct},
        )]
        signals.extend(self._transition_to_phase2(market))
        return signals

    def _transition_to_phase2(self, market: MarketData) -> list[Signal]:
        """2단계 전환: IAU 70% 매수."""
        self._phase = SwingPhase.PHASE2
        self._phase2_entry_date = market.time.date()

        phase2_ticker = self.params.get("phase2_ticker", "IAU")
        phase2_pct = self.params.get("phase2_pct", 0.70)

        self._phase2_entry_price = market.prices.get(phase2_ticker, 0.0)
        self._phase2_bought = True
        self._phase2_sold = False

        return [Signal(
            action=Action.BUY,
            ticker=phase2_ticker,
            size=phase2_pct,
            target_pct=0,
            reason=f"swing_mode: 2단계 IAU {int(phase2_pct*100)}% 진입 (5개월)",
            metadata={
                "swing_phase": "phase2",
                "phase2_months": self.params.get("phase2_months", 5),
                "stop_pct": self.params.get("phase2_stop_pct", -5.0),
            },
        )]

    def _phase2_signals(
        self, market: MarketData, positions: dict[str, Position]
    ) -> list[Signal]:
        """2단계: IAU 보유 — 만기/손절 체크."""
        phase2_ticker = self.params.get("phase2_ticker", "IAU")
        pos = positions.get(phase2_ticker)

        if not pos or pos.qty <= 0:
            self._reset()
            return []

        current = market.prices.get(phase2_ticker, 0.0)
        if current <= 0 or self._phase2_entry_price <= 0:
            return []

        pnl_pct = (current - self._phase2_entry_price) / self._phase2_entry_price * 100
        today = market.time.date()

        months = self.params.get("phase2_months", 5)
        stop_pct = self.params.get("phase2_stop_pct", -5.0)

        # 5개월 만기
        if self._phase2_entry_date:
            days_held = (today - self._phase2_entry_date).days
            if days_held >= months * 30:
                self._phase2_sold = True
                self._reset()
                return [Signal(
                    action=Action.SELL,
                    ticker=phase2_ticker,
                    size=1.0,
                    target_pct=0,
                    reason="swing_mode: 2단계 5개월 만기 → 스윙 종료",
                    exit_reason=ExitReason.TIME_LIMIT,
                    metadata={"swing_phase": "phase2", "pnl_pct": pnl_pct},
                )]

        # IAU -5% 손절
        if pnl_pct <= stop_pct:
            self._phase2_sold = True
            self._reset()
            return [Signal(
                action=Action.SELL,
                ticker=phase2_ticker,
                size=1.0,
                target_pct=0,
                reason=f"swing_mode: 2단계 IAU 손절 {pnl_pct:+.1f}%",
                exit_reason=ExitReason.STOP_LOSS,
                metadata={"swing_phase": "phase2", "pnl_pct": pnl_pct},
            )]

        return []  # HOLD

    def on_vix_defense_triggered(self, market: MarketData, positions: dict[str, Position]) -> list[Signal]:
        """VIX 방어모드 발동 시 스윙 전환 처리 (13-8절).

        - 1단계 중: 급등 보유분 매도 → INACTIVE (방어모드가 이어받음)
        - 2단계 중: IAU 유지, GDXU 30% 추가 매수는 방어모드가 처리
        """
        signals = []

        if self._phase == SwingPhase.PHASE1:
            # 1단계 중 VIX 방어 → 매도 후 방어모드 전환
            pos = positions.get(self._phase1_ticker)
            if pos and pos.qty > 0:
                signals.append(Signal(
                    action=Action.SELL,
                    ticker=self._phase1_ticker,
                    size=1.0,
                    target_pct=0,
                    reason="swing_mode: VIX 방어모드 우선 — 1단계 매도",
                    exit_reason=ExitReason.CONDITION_BREAK,
                    metadata={"swing_vix_override": True},
                ))
            self._reset()

        elif self._phase == SwingPhase.PHASE2:
            # 2단계(IAU 보유) 중 VIX → IAU 유지, GDXU는 방어모드가 처리
            pass  # No action needed, vix_gold handles GDXU

        return signals

    def _reset(self):
        """상태 초기화."""
        self._phase = SwingPhase.INACTIVE
        self._phase1_ticker = ""
        self._phase1_entry_date = None
        self._phase1_entry_price = 0.0
        self._phase2_entry_date = None
        self._phase2_entry_price = 0.0
        self._phase1_bought = False
        self._phase1_sold = False
        self._phase2_bought = False
        self._phase2_sold = False

    @staticmethod
    def _calc_atr(df, period: int = 14) -> float:
        """ATR(14) 계산."""
        import math

        import pandas as pd

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not math.isnan(atr) else 0.0

    def summary(self) -> dict:
        return {
            "phase": self._phase.value,
            "phase1_ticker": self._phase1_ticker,
            "phase1_entry_date": str(self._phase1_entry_date) if self._phase1_entry_date else None,
            "phase2_entry_date": str(self._phase2_entry_date) if self._phase2_entry_date else None,
            "trading_blocked": self.is_trading_blocked,
        }

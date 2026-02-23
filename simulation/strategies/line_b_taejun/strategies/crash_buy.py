"""
급락 역매수 전략 (Crash Buy)
==========================================
v5 매매 규칙 5-5절 — SOXL/CONL/IRE -30% 이상 급락 또는 LULD 3회 이상 발생 시
장마감 5분전(15:55 ET)에 총 투자금의 95%로 진입, 다음날 갭상승 시 전량 매도.

서킷 브레이커 예외 (1-4절):
- CB-1, CB-2, CB-3 발동 중에도 허용
- CB-5 발동 시 금지
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import CRASH_BUY
from ..common.registry import register


def _is_entry_time(time: datetime, entry_et_hour: int = 15, entry_et_min: int = 55) -> bool:
    """장마감 5분전 진입 시간대인지 확인한다.

    백테스트 단순화: 시간이 15:55 이상이면 true (실전에서는 ET 변환 필요).
    """
    return time.hour == entry_et_hour and time.minute >= entry_et_min


@register
class CrashBuyStrategy(BaseStrategy):
    """급락 역매수 — SOXL/CONL/IRE -30%+ 또는 LULD 3회+ → 95% 매수.

    v5 변경: -40% → -30% (현실적 트리거), LULD 3회 추가 조건.
    """

    name = "crash_buy"
    version = "1.0"
    description = "SOXL/CONL/IRE -30%+ 또는 LULD 3회+ → 장마감5분전 95% 매수, 갭상승시 전량매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or CRASH_BUY)
        # 포지션별 진입일 추적 (갭상승 매도용)
        self._crash_entry_dates: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # 트리거 판정
    # ------------------------------------------------------------------

    def _find_crash_ticker(self, market: MarketData) -> str | None:
        """급락 역매수 조건 충족 종목 중 우선순위 1개를 반환한다.

        우선순위: 1) 하락률 큰 종목, 2) 동률이면 거래량 큰 종목.
        """
        tickers = self.params.get("tickers", ["SOXL", "CONL", "IRE"])
        drop_trigger = self.params.get("drop_trigger", -30.0)
        luld_min = self.params.get("luld_count_min", 3)

        candidates: list[tuple[str, float]] = []

        for ticker in tickers:
            chg = market.changes.get(ticker, 0.0)
            luld = (market.luld_counts or {}).get(ticker, 0)

            triggered_by_drop = chg <= drop_trigger
            triggered_by_luld = luld >= luld_min

            if triggered_by_drop or triggered_by_luld:
                candidates.append((ticker, chg))

        if not candidates:
            return None

        # 우선순위 1: 하락률 절대값 큰 종목
        candidates.sort(key=lambda x: x[1])  # 가장 낮은(음수 큰) 종목 선택
        best_ticker, best_drop = candidates[0]

        # 복수 동률이면 거래량으로 결정
        same_drop = [t for t, d in candidates if d == best_drop]
        if len(same_drop) > 1 and market.volumes:
            same_drop.sort(
                key=lambda t: market.volumes.get(t, 0.0),
                reverse=True,
            )
            best_ticker = same_drop[0]

        return best_ticker

    # ------------------------------------------------------------------
    # 진입 / 청산 조건
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """급락 역매수 진입 조건:
        1. 대상 종목 중 하나가 -30% 이하 또는 LULD 3회+
        2. 장마감 5분전 (15:55 ET)
        3. 현재 crash_buy 포지션 미보유
        """
        tickers = self.params.get("tickers", ["SOXL", "CONL", "IRE"])
        # 이미 crash_buy 포지션 보유 중이면 재진입 금지
        for t in tickers:
            if t in self._crash_entry_dates:
                return False

        if not _is_entry_time(
            market.time,
            self.params.get("entry_et_hour", 15),
            self.params.get("entry_et_min", 55),
        ):
            return False

        return self._find_crash_ticker(market) is not None

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """갭상승(시가 > 전일 종가) 시 전량 매도."""
        ticker = position.ticker
        current = market.prices.get(ticker, 0.0)
        entry_price = position.avg_price
        if current <= 0 or entry_price <= 0:
            return False
        # 갭상승: 현재가가 진입가보다 높으면 (다음날 시초가 기준)
        return current > entry_price

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signal(
        self, market: MarketData, position: Position | None = None
    ) -> Signal:
        if position is not None:
            return self._generate_exit_signal(market, position)
        return self._generate_entry_signal(market)

    def _generate_entry_signal(self, market: MarketData) -> Signal:
        """진입 시그널 — 장마감 5분전 95% 매수."""
        if not self.check_entry(market):
            return Signal(Action.SKIP, "", 0, 0, "crash_buy: conditions not met")

        ticker = self._find_crash_ticker(market)
        if not ticker:
            return Signal(Action.SKIP, "", 0, 0, "crash_buy: no trigger ticker")

        buy_pct = self.params.get("buy_pct", 0.95)
        chg = market.changes.get(ticker, 0.0)
        luld = (market.luld_counts or {}).get(ticker, 0)

        # 진입 날짜 기록
        self._crash_entry_dates[ticker] = market.time

        return Signal(
            action=Action.BUY,
            ticker=ticker,
            size=buy_pct,
            target_pct=0,
            reason=(
                f"crash_buy: {ticker} {chg:+.1f}% (LULD {luld}회) "
                f"→ {int(buy_pct*100)}% 매수 @ 장마감5분전"
            ),
            metadata={
                "crash_buy": True,
                "trigger_chg": chg,
                "luld_count": luld,
                "entry_time": str(market.time),
                "cash_reserve_pct": 1.0 - buy_pct,  # 5% 현금 유지
            },
        )

    def _generate_exit_signal(
        self, market: MarketData, position: Position
    ) -> Signal:
        """매도 시그널.

        - 갭상승: 즉시 전량 매도
        - 보합 (시가 ≈ 진입가): HOLD (30분 관찰)
        - 갭하락: 일반 손절 규칙으로 전환 (HOLD + metadata)
        """
        ticker = position.ticker
        current = market.prices.get(ticker, 0.0)
        entry = position.avg_price

        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, 0, "crash_buy: no price data")

        pnl_pct = (current - entry) / entry * 100

        # 갭상승 → 즉시 전량 매도
        if current > entry:
            # 진입 날짜 기록 제거
            self._crash_entry_dates.pop(ticker, None)
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,
                target_pct=0,
                reason=f"crash_buy: 갭상승 {pnl_pct:+.1f}% → 전량 매도",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={"crash_buy": True, "pnl_pct": pnl_pct},
            )

        # 갭하락 → 손절 규칙 적용 플래그
        if pnl_pct < -8.0:
            self._crash_entry_dates.pop(ticker, None)
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,
                target_pct=0,
                reason=f"crash_buy: 갭하락 손절 {pnl_pct:+.1f}%",
                exit_reason=ExitReason.STOP_LOSS,
                metadata={"crash_buy": True, "pnl_pct": pnl_pct, "gap_down": True},
            )

        # 보합 → 30분 관찰
        return Signal(
            Action.HOLD, ticker, 0, 0,
            f"crash_buy: 보합 {pnl_pct:+.1f}% — 30분 관찰",
            metadata={"crash_buy": True, "observe": True},
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("tickers"):
            errors.append("tickers list required")
        if self.params.get("buy_pct", 0) <= 0 or self.params.get("buy_pct", 0) > 1:
            errors.append("buy_pct must be between 0 and 1")
        return errors

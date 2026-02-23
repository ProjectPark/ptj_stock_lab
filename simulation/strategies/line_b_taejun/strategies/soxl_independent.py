"""
SOXL 독립 매매 전략 (SOXL Independent)
==========================================
v5 매매 규칙 4-7절 — SOXL은 쌍둥이 페어가 아닌 독립 매매 종목.

매수 조건:
  1. SOXX 당일 등락률 +2% 이상
  2. ADX(14) >= 20

매도 구조 (40/60 분할):
  - 40%: SOXX 모멘텀 약화 시 즉시 매도 (당일 +0.5% 미만 or EMA 기울기 음수)
  - 60%: +5% 고정 익절 (미도달 시 시간 손절 or ATR 손절)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import SOXL_INDEPENDENT
from ..common.registry import register

if TYPE_CHECKING:
    import pandas as pd


def _compute_adx(df: "pd.DataFrame", period: int = 14) -> float:
    """ADX(14) 계산."""
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

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)

    tr_s = tr.rolling(period).mean()
    plus_s = plus_dm.rolling(period).mean()
    minus_s = minus_dm.rolling(period).mean()

    plus_di = 100 * plus_s / tr_s.replace(0, float("nan"))
    minus_di = 100 * minus_s / tr_s.replace(0, float("nan"))
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.rolling(period).mean()
    val = adx.iloc[-1]
    import math
    return float(val) if not math.isnan(val) else 0.0


def _ema_slope_positive(df: "pd.DataFrame", span: int = 20) -> bool:
    """20 EMA 기울기 양수 여부."""
    close = df["close"].astype(float)
    ema = close.ewm(span=span, adjust=False).mean()
    if len(ema) < 2:
        return False
    return ema.iloc[-1] > ema.iloc[-2]


@register
class SoxlIndependentStrategy(BaseStrategy):
    """SOXL 독립 매매 — SOXX +2% + ADX>=20 진입, 40/60 분할 매도.

    v5 4-7절: SOXL은 반도체 섹터 모멘텀 기반 독립 전략.
    쌍둥이 갭 매매와 무관하게 독립적으로 운용.
    """

    name = "soxl_independent"
    version = "1.0"
    description = "SOXX +2%+ADX>=20 진입, 40% SOXX모멘텀약화 즉시매도 + 60% +5% 고정익절"

    def __init__(self, params: dict | None = None):
        super().__init__(params or SOXL_INDEPENDENT)
        # 분할 매도 상태 추적
        self._first_sell_done: bool = False   # 40% 매도 완료 여부

    # ------------------------------------------------------------------
    # 진입 조건
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """SOXX +2% + ADX >= 20."""
        soxx_min = self.params.get("soxx_min", 2.0)
        adx_min = self.params.get("adx_min", 20)

        soxx_chg = market.changes.get("SOXX", 0.0)
        if soxx_chg < soxx_min:
            return False

        # ADX 체크 (OHLCV 데이터 있을 때만)
        ohlcv = market.ohlcv or {}
        soxx_df = ohlcv.get("SOXX") or ohlcv.get("SOXL")
        if soxx_df is not None and len(soxx_df) >= 20:
            adx_val = _compute_adx(soxx_df)
            if adx_val < adx_min:
                return False
        # OHLCV 없으면 ADX 체크 스킵 (백테스트 호환)

        return True

    # ------------------------------------------------------------------
    # 매도 조건 분기
    # ------------------------------------------------------------------

    def _is_momentum_weak(self, market: MarketData) -> bool:
        """SOXX 모멘텀 약화 여부.
        - SOXX 당일 등락률 +0.5% 미만
        - 또는 EMA 기울기 음수
        """
        soxx_weak_pct = self.params.get("soxx_weak_pct", 0.5)
        soxx_chg = market.changes.get("SOXX", 0.0)
        if soxx_chg < soxx_weak_pct:
            return True

        ohlcv = market.ohlcv or {}
        soxx_df = ohlcv.get("SOXX") or ohlcv.get("SOXL")
        if soxx_df is not None and len(soxx_df) >= 21:
            if not _ema_slope_positive(soxx_df):
                return True

        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """매도 조건 (40% 1차 or 60% 고정익절)."""
        ticker = self.params.get("ticker", "SOXL")
        current = market.prices.get(ticker, 0.0)
        if current <= 0 or position.avg_price <= 0:
            return False

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        tp_pct = self.params.get("sell_tp_pct", 5.0)

        # 60% 고정 익절 조건
        if pnl_pct >= tp_pct:
            return True

        # 40% 모멘텀 약화 조건 (1차 매도 미완료 시만)
        if not self._first_sell_done and self._is_momentum_weak(market):
            return True

        return False

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signal(
        self, market: MarketData, position: Position | None = None
    ) -> Signal:
        ticker = self.params.get("ticker", "SOXL")

        if position is not None:
            return self._generate_exit_signal(market, position)

        if not self.check_entry(market):
            return Signal(Action.SKIP, ticker, 0, 0, "soxl_independent: conditions not met")

        initial_usd = self.params.get("initial_usd", 2250)
        soxx_chg = market.changes.get("SOXX", 0.0)

        return Signal(
            action=Action.BUY,
            ticker=ticker,
            size=0,  # size=0 → metadata의 amount 사용
            target_pct=self.params.get("sell_tp_pct", 5.0),
            reason=f"soxl_independent: SOXX {soxx_chg:+.1f}% → SOXL ${initial_usd} 진입",
            metadata={
                "amount_usd": initial_usd,
                "strategy": "soxl_independent",
                "sell_tp_ratio": self.params.get("sell_tp_ratio", 0.60),
                "sell_momentum_ratio": self.params.get("sell_momentum_ratio", 0.40),
            },
        )

    def _generate_exit_signal(
        self, market: MarketData, position: Position
    ) -> Signal:
        ticker = self.params.get("ticker", "SOXL")
        current = market.prices.get(ticker, 0.0)
        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, 0, "soxl_independent: no price data")

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        tp_pct = self.params.get("sell_tp_pct", 5.0)
        tp_ratio = self.params.get("sell_tp_ratio", 0.60)         # 60%
        momentum_ratio = self.params.get("sell_momentum_ratio", 0.40)  # 40%

        # 우선순위 1: +5% 고정 익절 → 전량 매도 (60%+40%)
        if pnl_pct >= tp_pct:
            self._first_sell_done = False  # 상태 초기화
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,
                target_pct=0,
                reason=f"soxl_independent: +5% 익절 {pnl_pct:+.1f}% → 전량 매도",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={
                    "split_sell": True,
                    "sell_reason": "fixed_tp",
                    "pnl_pct": pnl_pct,
                    "sell_size": 1.0,
                },
            )

        # 우선순위 2: SOXX 모멘텀 약화 → 40% 즉시 매도 (1차 미완료 시)
        if not self._first_sell_done and self._is_momentum_weak(market):
            self._first_sell_done = True
            soxx_chg = market.changes.get("SOXX", 0.0)
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=momentum_ratio,  # 40%
                target_pct=0,
                reason=(
                    f"soxl_independent: SOXX 모멘텀 약화 ({soxx_chg:+.1f}%) "
                    f"→ {int(momentum_ratio*100)}% 즉시 매도"
                ),
                exit_reason=ExitReason.CONDITION_BREAK,
                metadata={
                    "split_sell": True,
                    "sell_reason": "momentum_weak",
                    "pnl_pct": pnl_pct,
                    "sell_size": momentum_ratio,
                    "remaining_tp_ratio": tp_ratio,  # 60%는 +5% 대기
                    "remaining_tp_pct": tp_pct,
                },
            )

        return Signal(
            Action.HOLD, ticker, 0, tp_pct,
            f"soxl_independent: 보유 중 pnl={pnl_pct:+.1f}%, SOXX 모멘텀 유지",
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("soxx_min", 0) <= 0:
            errors.append("soxx_min must be positive")
        if self.params.get("adx_min", 0) <= 0:
            errors.append("adx_min must be positive")
        return errors

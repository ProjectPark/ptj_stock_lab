"""
섹터 로테이션 전략 (Sector-Rotate) — Q-8 전면 개편
====================================================
1Y 저가 대비 상승률 기반 순차 로테이션.
비트코인 → 반도체 → 은행 → 금(현금) 순서로 순환.

activate_pct 도달시 매수 시작, deactivate_pct 도달시 전액 매도 → 다음 단계.
gold 단계는 현금 보유 + FX 헤지 메타데이터 출력.
전체 사이클 완료 후 처음으로 복귀.

출처: kakaotalk_trading_notes_2026-02-19.csv
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import SECTOR_ROTATE
from ..common.registry import register


@register
class SectorRotate(BaseStrategy):
    """섹터 로테이션 — 1Y 저가 대비 상승률 기반 순차 매수."""

    name = "sector_rotate"
    version = "2.0"
    description = "4대 섹터 순차 로테이션 (1Y 저가 기반 activate/deactivate)"

    def __init__(self, params: dict | None = None):
        super().__init__(params or SECTOR_ROTATE)
        self._current_idx: int = 0          # 현재 로테이션 위치
        self._last_buy_dates: dict[str, datetime] = {}
        self._sequence = self.params.get("rotation_sequence", [])

    # ------------------------------------------------------------------
    # 1Y 저가 대비 상승률 계산
    # ------------------------------------------------------------------

    def _calc_pct_above_1y_low(self, ticker: str, market: MarketData) -> float | None:
        """1Y 저가 대비 현재가 상승률(%)을 계산한다.

        Returns None if data unavailable.
        """
        if not market.history or ticker not in market.history:
            return None
        low_1y = market.history[ticker].get("low_1y", 0)
        if low_1y <= 0:
            return None
        current = market.prices.get(ticker, 0)
        if current <= 0:
            return None
        return (current - low_1y) / low_1y * 100

    # ------------------------------------------------------------------
    # 현재 단계 정보
    # ------------------------------------------------------------------

    def _current_step(self) -> dict | None:
        """현재 로테이션 단계를 반환한다."""
        if not self._sequence or self._current_idx >= len(self._sequence):
            return None
        return self._sequence[self._current_idx]

    def _advance_step(self) -> None:
        """다음 로테이션 단계로 전진. 마지막이면 처음으로 복귀."""
        self._current_idx = (self._current_idx + 1) % len(self._sequence)

    # ------------------------------------------------------------------
    # 매수 간격 체크
    # ------------------------------------------------------------------

    def _is_interval_met(self, step: dict, market: MarketData) -> bool:
        """해당 단계의 매수 간격이 충족되었는지."""
        buy_ticker = step.get("buy", "")
        if not buy_ticker:
            return False

        last = self._last_buy_dates.get(buy_ticker)
        if last is None:
            return True

        interval = timedelta(days=step.get("interval_days", 7))
        return (market.time - last) >= interval

    # ------------------------------------------------------------------
    # 리츠 주의사항
    # ------------------------------------------------------------------

    def _check_caution(self, market: MarketData, buy_ticker: str) -> str | None:
        """리츠 기반 주의사항 체크."""
        caution = self.params.get("caution_rules", {})
        sk_reit = market.changes.get("SK리츠", None)
        if sk_reit is None:
            return None

        if buy_ticker == "GDXU" and sk_reit <= caution.get("gdxu_sk_reit_drop", -1.0):
            return f"caution: SK리츠 {sk_reit:.2f}% → GDXU 매수 조심"

        if buy_ticker == "CONL":
            if sk_reit > 0:
                return f"caution: SK리츠 양전 {sk_reit:.2f}% → CONL 매수 주의"

        return None

    # ------------------------------------------------------------------
    # 진입 / 청산
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        step = self._current_step()
        if step is None:
            return False

        # gold 단계 — 현금 보유, 매수 없음
        if step.get("action") == "cash":
            return False

        proxy = step.get("proxy", "")
        pct = self._calc_pct_above_1y_low(proxy, market)
        if pct is None:
            return False

        if pct < step.get("activate_pct", 0):
            return False

        return self._is_interval_met(step, market)

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """deactivate_pct 도달시 매도 → 다음 단계 전환."""
        step = self._current_step()
        if step is None:
            return False

        proxy = step.get("proxy", "")
        pct = self._calc_pct_above_1y_low(proxy, market)
        if pct is None:
            return False

        return pct >= step.get("deactivate_pct", float("inf"))

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        step = self._current_step()
        if step is None:
            return Signal(Action.SKIP, "", 0, 0, "no rotation sequence")

        sector_name = step.get("name", "unknown")
        proxy = step.get("proxy", "")
        pct = self._calc_pct_above_1y_low(proxy, market)

        # gold 단계 — 현금 보유 메타데이터 출력
        if step.get("action") == "cash":
            fx_hedge = step.get("fx_hedge", {})
            # activate 도달시 다음 사이클로 전환 체크
            if pct is not None and pct >= step.get("activate_pct", float("inf")):
                self._advance_step()
                return Signal(
                    Action.SKIP, "", 0, 0,
                    f"sector_rotate: gold phase complete ({pct:.1f}%), cycling back",
                    metadata={"sector": sector_name, "cycle_restart": True},
                )
            return Signal(
                Action.HOLD, "cash", 0, 0,
                f"sector_rotate: gold phase — hold cash ({proxy} {pct or 0:.1f}% from 1Y low)",
                metadata={"sector": sector_name, "action": "cash",
                          "fx_hedge": fx_hedge},
            )

        buy_ticker = step.get("buy", "")

        # 보유 포지션 매도 체크 (deactivate)
        if position is not None:
            if pct is not None and pct >= step.get("deactivate_pct", float("inf")):
                self._advance_step()
                return Signal(
                    action=Action.SELL,
                    ticker=position.ticker,
                    size=1.0,
                    target_pct=0,
                    reason=f"sector_rotate: {sector_name} deactivate "
                           f"({proxy} {pct:.1f}% >= {step['deactivate_pct']}%)",
                    exit_reason=ExitReason.CONDITION_BREAK,
                    metadata={"sector": sector_name, "next_idx": self._current_idx},
                )
            return Signal(
                Action.HOLD, position.ticker, 0, 0,
                f"sector_rotate: holding {position.ticker} "
                f"({proxy} {pct or 0:.1f}% from 1Y low)",
                metadata={"sector": sector_name},
            )

        # 진입 체크
        if pct is None:
            return Signal(Action.SKIP, buy_ticker, 0, 0,
                         f"no 1Y low data for {proxy}")

        if pct < step.get("activate_pct", 0):
            return Signal(Action.SKIP, buy_ticker, 0, 0,
                         f"sector_rotate: {sector_name} not activated "
                         f"({proxy} {pct:.1f}% < {step['activate_pct']}%)")

        if not self._is_interval_met(step, market):
            last = self._last_buy_dates.get(buy_ticker)
            return Signal(Action.SKIP, buy_ticker, 0, 0,
                         f"interval not met: {step.get('interval_days')}d (last: {last})")

        caution = self._check_caution(market, buy_ticker)
        metadata: dict = {
            "sector": sector_name,
            "interval_days": step.get("interval_days"),
            "qty": step.get("qty", 1),
            "proxy_pct_above_1y_low": round(pct, 2),
        }
        if caution:
            metadata["caution"] = caution

        self._last_buy_dates[buy_ticker] = market.time

        return Signal(
            action=Action.BUY,
            ticker=buy_ticker,
            size=0,  # qty 기반 (금액 아닌 수량)
            target_pct=0,  # 정기 매수 — 목표 수익률 없음
            reason=f"sector_rotate: {sector_name} → buy {buy_ticker} "
                   f"({proxy} {pct:.1f}% from 1Y low, "
                   f"every {step.get('interval_days')}d)",
            metadata=metadata,
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("rotation_sequence"):
            errors.append("rotation_sequence required")
        if not self.params.get("sector_proxies"):
            errors.append("sector_proxies required")
        return errors

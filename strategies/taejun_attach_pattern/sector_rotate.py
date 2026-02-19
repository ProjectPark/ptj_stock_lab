"""
섹터 로테이션 전략 (Sector-Rotate)
====================================
비트코인/반도체/금/은행 4대 섹터 수익률 비교 → 다음 섹터에 정기 매수.
리츠 기반 주의사항 포함.

출처: kakaotalk_trading_notes_2026-02-19.csv
- 비트코인/반도체/금/은행중 비트코인이 수익이크다면 SOXL...
- GDXU 매수시 주의사항은 SK리츠가...
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .base import Action, BaseStrategy, MarketData, Position, Signal
from .params import SECTOR_ROTATE
from .registry import register


@register
class SectorRotate(BaseStrategy):
    """섹터 로테이션 — 1위 섹터 수익률 기반 다음 섹터 정기 매수."""

    name = "sector_rotate"
    version = "1.0"
    description = "4대 섹터 수익률 비교 → 다음 섹터 정기 매수 (3일/7일/14일)"

    def __init__(self, params: dict | None = None):
        super().__init__(params or SECTOR_ROTATE)
        self._last_buy_dates: dict[str, datetime] = {}

    def _get_top_sector(self, market: MarketData) -> str | None:
        """수익률 1위 섹터를 반환한다."""
        proxies = self.params.get("sector_proxies", {})
        sector_returns = {}
        for sector, ticker in proxies.items():
            chg = market.changes.get(ticker, None)
            if chg is not None:
                sector_returns[sector] = chg

        if not sector_returns:
            return None

        return max(sector_returns, key=sector_returns.get)

    def _is_interval_met(self, sector: str, market: MarketData) -> bool:
        """해당 섹터의 매수 간격이 충족되었는지."""
        rotation = self.params.get("rotation", {})
        cfg = rotation.get(sector)
        if not cfg:
            return False

        buy_ticker = cfg["buy"]
        last = self._last_buy_dates.get(buy_ticker)
        if last is None:
            return True

        interval = timedelta(days=cfg["interval_days"])
        return (market.time - last) >= interval

    def _check_caution(self, market: MarketData, buy_ticker: str) -> str | None:
        """리츠 기반 주의사항 체크. 주의 메시지를 반환하거나 None."""
        caution = self.params.get("caution_rules", {})

        sk_reit = market.changes.get("SK리츠", None)
        if sk_reit is None:
            return None

        if buy_ticker == "GDXU" and sk_reit <= caution.get("gdxu_sk_reit_drop", -1.0):
            return f"caution: SK리츠 {sk_reit:.2f}% → GDXU 매수 조심"

        if buy_ticker == "CONL":
            # SK리츠 7일 연속 상승시 CONL 조심 (간략화: 당일 양전만 체크)
            if sk_reit > 0:
                return f"caution: SK리츠 양전 {sk_reit:.2f}% → CONL 매수 주의"

        return None

    def check_entry(self, market: MarketData) -> bool:
        top = self._get_top_sector(market)
        if top is None:
            return False
        return top in self.params.get("rotation", {}) and self._is_interval_met(top, market)

    def check_exit(self, market: MarketData, position: Position) -> bool:
        # 섹터 로테이션은 정기 매수 전략 — 매도 로직은 별도 없음
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        top = self._get_top_sector(market)
        if top is None:
            return Signal(Action.SKIP, "", 0, 0, "no sector data")

        rotation = self.params.get("rotation", {})
        cfg = rotation.get(top)
        if not cfg:
            return Signal(Action.SKIP, "", 0, 0, f"no rotation rule for sector: {top}")

        buy_ticker = cfg["buy"]

        if not self._is_interval_met(top, market):
            last = self._last_buy_dates.get(buy_ticker)
            return Signal(Action.SKIP, buy_ticker, 0, 0,
                         f"interval not met: {cfg['interval_days']}d (last: {last})")

        caution = self._check_caution(market, buy_ticker)
        metadata = {"sector": top, "interval_days": cfg["interval_days"], "qty": cfg["qty"]}
        if caution:
            metadata["caution"] = caution

        self._last_buy_dates[buy_ticker] = market.time

        return Signal(
            action=Action.BUY,
            ticker=buy_ticker,
            size=0,  # qty 기반 (금액 아닌 수량)
            target_pct=0,  # 정기 매수 — 목표 수익률 없음
            reason=f"sector_rotate: top={top} → buy {buy_ticker} (every {cfg['interval_days']}d)",
            metadata=metadata,
        )

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("rotation"):
            errors.append("rotation mapping required")
        if not self.params.get("sector_proxies"):
            errors.append("sector_proxies required")
        return errors

"""
잽모드 SOXL 전략 (Jab-SOXL)
=============================
프리마켓 시간대, 반도체 개별주 모두 상승인데 SOXX/SOXL만 마이너스인 괴리 역전 단타.
11개 개별 반도체 종목 조건을 ALL 충족해야 진입.

출처: kakaotalk_trading_notes_2026-02-19.csv — (1) 잽 모드 SOXL 매수 <공격모드>
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import JAB_SOXL
from ..common.registry import register


@register
class JabSOXL(BaseStrategy):
    """잽모드 SOXL — 반도체 개별주 상승 + SOXX/SOXL 과매도 역전 단타."""

    name = "jab_soxl"
    version = "1.0"
    description = "Polymarket NDX 51%+ & 반도체 11종목 상승 & SOXL 과매도시 진입, +0.9% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or JAB_SOXL)

    def _is_in_window(self, market: MarketData) -> bool:
        start = self.params.get("entry_start_kst", (17, 30))
        h, m = market.time.hour, market.time.minute
        return (h, m) >= tuple(start)

    def check_entry(self, market: MarketData) -> bool:
        """모든 조건 ALL 충족.

        1. 시간 17:30 KST 이후
        2. Polymarket NASDAQ >= 51%
        3. GLD >= +0.1%
        4. QQQ >= +0.3%
        5. SOXX <= -0.2%
        6. SOXL <= -0.6%
        7. 개별 반도체 11종목 각각 최소 변동률 충족
        """
        if not self._is_in_window(market):
            return False

        if not market.poly:
            return False
        ndx_up = market.poly.get("ndx_up", 0)
        if ndx_up < self.params["poly_ndx_min"]:
            return False

        gld = market.changes.get("GLD", 0)
        if gld < self.params["gld_min"]:
            return False

        qqq = market.changes.get("QQQ", 0)
        if qqq < self.params["qqq_min"]:
            return False

        soxx = market.changes.get("SOXX", 0)
        if soxx > self.params["soxx_max"]:
            return False

        soxl = market.changes.get("SOXL", 0)
        if soxl > self.params["soxl_max"]:
            return False

        # 개별 반도체 종목 ALL 충족
        individual = self.params.get("individual", {})
        for ticker, min_chg in individual.items():
            actual = market.changes.get(ticker, 0)
            if actual < min_chg:
                return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        ticker = self.params.get("ticker", "SOXL")
        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "SOXL")

        if position is not None:
            current = market.prices.get(ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=ticker, size=1.0, target_pct=0,
                    reason=f"jab_soxl target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={"pnl_pct": pnl_pct},
                )
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         f"holding SOXL: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            failed = self._get_failed_conditions(market)
            return Signal(Action.SKIP, ticker, 0, 0,
                         f"conditions not met: {failed}")

        return Signal(
            action=Action.BUY, ticker=ticker,
            size=self.params["size"],
            target_pct=self.params["target_pct"],
            reason=(
                f"jab_soxl entry: poly_ndx={market.poly.get('ndx_up', 0):.0%}, "
                f"SOXX={market.changes.get('SOXX', 0):+.2f}%, "
                f"SOXL={market.changes.get('SOXL', 0):+.2f}%"
            ),
            metadata={
                "individual_met": {
                    t: market.changes.get(t, 0)
                    for t in self.params.get("individual", {})
                },
            },
        )

    def _get_failed_conditions(self, market: MarketData) -> str:
        fails = []
        if not market.poly or market.poly.get("ndx_up", 0) < self.params["poly_ndx_min"]:
            fails.append("poly_ndx")
        for key, field, op in [
            ("gld_min", "GLD", ">="), ("qqq_min", "QQQ", ">="),
            ("soxx_max", "SOXX", "<="), ("soxl_max", "SOXL", "<="),
        ]:
            val = market.changes.get(field, 0)
            threshold = self.params[key]
            if op == ">=" and val < threshold:
                fails.append(f"{field}({val:.2f}%)")
            elif op == "<=" and val > threshold:
                fails.append(f"{field}({val:.2f}%)")

        for ticker, min_chg in self.params.get("individual", {}).items():
            actual = market.changes.get(ticker, 0)
            if actual < min_chg:
                fails.append(f"{ticker}({actual:.2f}%<{min_chg}%)")

        return ", ".join(fails) if fails else "unknown"

    def validate_params(self) -> list[str]:
        errors = []
        if not self.params.get("individual"):
            errors.append("individual semiconductor conditions required")
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        return errors

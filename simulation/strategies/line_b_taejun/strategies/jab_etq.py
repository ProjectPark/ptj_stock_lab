"""
숏 잽모드 ETQ 전략 (Jab-ETQ)
================================
이더리움 하락 기대가 높을 때 2x 인버스 ETF 매수.
SETH → ETQ 교체 (v6): 2x 레버리지, 목표 0.5% → 0.8%.

출처: kakaotalk_trading_notes_2026-02-19.csv — (4) 숏 잽모드 ETH 매수 <공격모드>
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import JAB_ETQ
from ..common.registry import register


@register
class JabETQ(BaseStrategy):
    """숏 잽모드 ETQ — ETH 하락 기대 높을 때 2x 인버스 ETF 매수."""

    name = "jab_etq"
    version = "1.0"
    description = "Polymarket 하락기대 최고값이 평균보다 12pp+ 높을 때 진입, +0.8% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or JAB_ETQ)

    def check_entry(self, market: MarketData) -> bool:
        """진입 조건.

        W-7: 17:30 KST 이후만 진입
        1. Polymarket 하락 기대 최고값이 평균보다 12pp 이상 높을 때
        2. GLD >= +0.01%
        3. ETQ >= 0.00% (양전)
        """
        # W-7: 17:30 KST 이후만 진입
        if market.time is not None:
            kst_hour, kst_min = market.time.hour, market.time.minute
            if (kst_hour, kst_min) < (17, 30):
                return False

        if not market.poly:
            return False

        # 모든 _up 키에서 하락 기대(%) 산출
        down_probs: dict[str, float] = {}
        for key, val in market.poly.items():
            if key.endswith("_up"):
                down_probs[key] = (1.0 - val) * 100
        if not down_probs:
            return False

        max_down = max(down_probs.values())
        avg_down = sum(down_probs.values()) / len(down_probs)
        if (max_down - avg_down) < self.params["poly_down_spread_min"]:
            return False

        gld = market.changes.get("GLD", 0)
        if gld < self.params["gld_min"]:
            return False

        ticker = self.params.get("ticker", "ETQ")
        etq = market.changes.get(ticker, 0)
        if etq < self.params.get("etq_min", 0.0):
            return False

        return True

    def check_exit(self, market: MarketData, position: Position) -> bool:
        ticker = self.params.get("ticker", "ETQ")
        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False
        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "ETQ")

        if position is not None:
            current = market.prices.get(ticker, 0)
            if current <= 0:
                return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                             "no price data")
            pnl_pct = (current - position.avg_price) / position.avg_price * 100
            if pnl_pct >= self.params["target_pct"]:
                return Signal(
                    action=Action.SELL, ticker=ticker, size=1.0, target_pct=0,
                    reason=f"jab_etq target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                    exit_reason=ExitReason.TARGET_HIT,
                    metadata={"pnl_pct": pnl_pct},
                )
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         f"holding ETQ: pnl={pnl_pct:.2f}%")

        if not self.check_entry(market):
            return Signal(Action.SKIP, ticker, 0, 0, "conditions not met")

        # 하락 기대 스프레드 계산 (로그용)
        down_probs: dict[str, float] = {}
        for key, val in (market.poly or {}).items():
            if key.endswith("_up"):
                down_probs[key] = (1.0 - val) * 100
        max_down = max(down_probs.values()) if down_probs else 0
        avg_down = (sum(down_probs.values()) / len(down_probs)) if down_probs else 0
        spread = max_down - avg_down

        return Signal(
            action=Action.BUY, ticker=ticker,
            size=self.params.get("size", 1.0),
            target_pct=self.params["target_pct"],
            reason=(
                f"jab_etq entry: down_spread={spread:.1f}pp, "
                f"GLD={market.changes.get('GLD', 0):+.2f}%, "
                f"ETQ={market.changes.get(ticker, 0):+.2f}%"
            ),
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        return errors

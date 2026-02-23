"""
잽모드 BITU 전략 (Jab-BITU)
============================
프리마켓 시간대, 비트코인 개별 코인 상승 + BITU 과매도 괴리를 노리는 단타.
크립토 스팟 가격(BTC/ETH/SOL/XRP)을 직접 수집하여 조건 판단.

출처: kakaotalk_trading_notes_2026-02-19.csv — (2) 잽 모드 BITU 매수 <공격모드>
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import JAB_BITU
from ..common.registry import register


@register
class JabBITU(BaseStrategy):
    """잽모드 BITU — BTC 생태계 상승 + BITU 과매도 역전 단타."""

    name = "jab_bitu"
    version = "1.0"
    description = "Polymarket BTC 63%+ & 크립토 상승 & BITU 과매도시 진입, +0.9% 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or JAB_BITU)

    # ------------------------------------------------------------------
    # 시간 조건
    # ------------------------------------------------------------------

    def _is_in_window(self, market: MarketData) -> bool:
        """매매 가능 시간대인지 확인 (KST 기준)."""
        start = self.params.get("entry_start_kst", (17, 30))
        h, m = market.time.hour, market.time.minute
        return (h, m) >= tuple(start)

    # ------------------------------------------------------------------
    # 진입 조건
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """모든 조건 ALL 충족 여부.

        1. 시간: 17:30 KST 이후
        2. Polymarket BTC 상승 기대 >= 63%
        3. GLD >= +0.1%
        4. BITU <= -0.4%
        5. 크립토 스팟: BTC >= +0.9%, ETH >= +0.9%, SOL >= +2.0%, XRP >= +5.0%
        """
        # 시간 체크
        if not self._is_in_window(market):
            return False

        # Polymarket BTC 조건
        if not market.poly:
            return False
        btc_up = market.poly.get("btc_up", 0)
        if btc_up < self.params["poly_btc_min"]:
            return False

        # GLD 변동률
        gld = market.changes.get("GLD", 0)
        if gld < self.params["gld_min"]:
            return False

        # BITU 과매도 (마이너스)
        bitu = market.changes.get("BITU", 0)
        if bitu > self.params["bitu_max"]:
            return False

        # 크립토 스팟 가격 조건
        if not market.crypto:
            return False
        crypto_conds = self.params.get("crypto_conditions", {})
        for coin, min_chg in crypto_conds.items():
            actual = market.crypto.get(coin, 0)
            if actual < min_chg:
                return False

        return True

    # ------------------------------------------------------------------
    # 청산 조건
    # ------------------------------------------------------------------

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """목표 수익률 +0.9% 도달 여부."""
        ticker = self.params.get("ticker", "BITU")
        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct >= self.params["target_pct"]

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        ticker = self.params.get("ticker", "BITU")

        # 포지션 보유 중 → 청산 검토
        if position is not None:
            return self._generate_exit_signal(market, position, ticker)

        # 신규 진입 검토
        return self._generate_entry_signal(market, ticker)

    def _generate_entry_signal(self, market: MarketData, ticker: str) -> Signal:
        """진입 시그널."""
        if not self._is_in_window(market):
            return Signal(Action.SKIP, ticker, 0, 0,
                         f"outside window (before {self.params['entry_start_kst']})")

        if not self.check_entry(market):
            failed = self._get_failed_conditions(market)
            return Signal(Action.SKIP, ticker, 0, 0,
                         f"conditions not met: {failed}")

        return Signal(
            action=Action.BUY,
            ticker=ticker,
            size=self.params["size"],
            target_pct=self.params["target_pct"],
            reason=self._build_entry_reason(market),
            metadata={
                "poly_btc": market.poly.get("btc_up", 0) if market.poly else 0,
                "gld_chg": market.changes.get("GLD", 0),
                "bitu_chg": market.changes.get("BITU", 0),
                "crypto": dict(market.crypto) if market.crypto else {},
            },
        )

    def _generate_exit_signal(self, market: MarketData,
                              position: Position, ticker: str) -> Signal:
        """청산 시그널."""
        current = market.prices.get(ticker, 0)
        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         "no price data")

        pnl_pct = (current - position.avg_price) / position.avg_price * 100

        if pnl_pct >= self.params["target_pct"]:
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,  # 전액 매도
                target_pct=0,
                reason=f"jab_bitu target hit: {pnl_pct:.2f}% >= {self.params['target_pct']}%",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={"pnl_pct": pnl_pct},
            )

        return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                      f"holding BITU: pnl={pnl_pct:.2f}%")

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _get_failed_conditions(self, market: MarketData) -> str:
        """미충족 조건들을 문자열로 반환."""
        fails = []

        if not market.poly or market.poly.get("btc_up", 0) < self.params["poly_btc_min"]:
            fails.append("poly_btc")

        gld = market.changes.get("GLD", 0)
        if gld < self.params["gld_min"]:
            fails.append(f"GLD({gld:.2f}%<{self.params['gld_min']}%)")

        bitu = market.changes.get("BITU", 0)
        if bitu > self.params["bitu_max"]:
            fails.append(f"BITU({bitu:.2f}%>{self.params['bitu_max']}%)")

        if market.crypto:
            for coin, min_chg in self.params.get("crypto_conditions", {}).items():
                actual = market.crypto.get(coin, 0)
                if actual < min_chg:
                    fails.append(f"{coin}({actual:.2f}%<{min_chg}%)")
        else:
            fails.append("no_crypto_data")

        return ", ".join(fails) if fails else "unknown"

    def _build_entry_reason(self, market: MarketData) -> str:
        """진입 사유 문자열."""
        poly_btc = market.poly.get("btc_up", 0) if market.poly else 0
        gld = market.changes.get("GLD", 0)
        bitu = market.changes.get("BITU", 0)
        return (
            f"jab_bitu entry: poly_btc={poly_btc:.0%}, "
            f"GLD={gld:+.2f}%, BITU={bitu:+.2f}%"
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("poly_btc_min", 0) <= 0:
            errors.append("poly_btc_min must be positive")
        if self.params.get("target_pct", 0) <= 0:
            errors.append("target_pct must be positive")
        if not self.params.get("crypto_conditions"):
            errors.append("crypto_conditions is required")
        return errors

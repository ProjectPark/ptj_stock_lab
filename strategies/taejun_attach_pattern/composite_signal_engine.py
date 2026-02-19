"""
CompositeSignalEngine — Legacy generate_all_signals_vN() 대체
============================================================
필터(MarketModeFilter, GoldFilter, SidewaysDetector)와
전략(TwinPairStrategy, ConditionalCoinStrategy, ConditionalConlStrategy,
BearishDefenseStrategy)를 조합하여 기존 7-key dict를 반환한다.

stop_loss는 포지션 기반이므로 이 엔진에 포함하지 않는다.
(백테스트 엔진에서 직접 처리)

Usage:
    from strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
    from strategies.params import V5Params

    engine = CompositeSignalEngine.from_base_params(V5Params())
    sigs = engine.generate_all_signals(changes, poly_probs, pairs, sideways_active)
    # sigs는 기존 generate_all_signals_v5()와 동일한 7-key dict
"""
from __future__ import annotations

from strategies.params import BaseParams

from .filters import GoldFilter, MarketModeFilter, SidewaysDetector
from .twin_pair import TwinPairStrategy
from .conditional_coin import ConditionalCoinStrategy
from .conditional_conl import ConditionalConlStrategy
from .bearish_defense import BearishDefenseStrategy


class CompositeSignalEngine:
    """필터 + 전략 조합 → 기존 generate_all_signals_vN()과 동일한 7-key dict 반환.

    stop_loss는 포지션 기반이므로 별도 처리 (기존 signals_v2.check_stop_loss_v2 직접 호출).
    """

    def __init__(
        self,
        market_mode_filter: MarketModeFilter,
        gold_filter: GoldFilter,
        sideways_detector: SidewaysDetector,
        twin_pair: TwinPairStrategy,
        conditional_coin: ConditionalCoinStrategy,
        conditional_conl: ConditionalConlStrategy,
        bearish_defense: BearishDefenseStrategy,
        stop_loss_normal: float = -3.0,
        stop_loss_bullish: float = -8.0,
    ):
        self.market_mode_filter = market_mode_filter
        self.gold_filter = gold_filter
        self.sideways_detector = sideways_detector
        self.twin_pair = twin_pair
        self.conditional_coin = conditional_coin
        self.conditional_conl = conditional_conl
        self.bearish_defense = bearish_defense
        self._stop_loss_normal = stop_loss_normal
        self._stop_loss_bullish = stop_loss_bullish

    @classmethod
    def from_base_params(cls, params: BaseParams) -> "CompositeSignalEngine":
        """BaseParams (또는 V3/V4/V5Params)에서 엔진을 생성한다."""
        market_mode_filter = MarketModeFilter()
        gold_filter = GoldFilter()

        # SidewaysDetector params
        sideways_params = {}
        for key in ("sideways_poly_low", "sideways_poly_high", "sideways_gld_threshold",
                     "sideways_gap_fail_count", "sideways_trigger_fail_count",
                     "sideways_index_threshold", "sideways_min_signals"):
            if hasattr(params, key):
                # sideways_poly_low -> poly_low
                short_key = key.replace("sideways_", "")
                sideways_params[short_key] = getattr(params, key)
        # Rename count -> threshold for detector
        if "gap_fail_count" in sideways_params:
            sideways_params["gap_fail_threshold"] = sideways_params.pop("gap_fail_count")
        if "trigger_fail_count" in sideways_params:
            sideways_params["trigger_fail_threshold"] = sideways_params.pop("trigger_fail_count")
        sideways_detector = SidewaysDetector(sideways_params)

        # TwinPairStrategy
        twin_pair = TwinPairStrategy({
            "entry_threshold": params.pair_gap_entry_threshold,
            "sell_threshold": params.pair_gap_sell_threshold,
        })

        # ConditionalCoinStrategy
        conditional_coin = ConditionalCoinStrategy({
            "trigger_pct": params.coin_trigger_pct,
            "sell_profit_pct": params.coin_sell_profit_pct,
            "sell_bearish_pct": params.coin_sell_bearish_pct,
        })

        # ConditionalConlStrategy
        conditional_conl = ConditionalConlStrategy({
            "trigger_pct": params.conl_trigger_pct,
            "sell_profit_pct": params.conl_sell_profit_pct,
            "sell_avg_pct": params.conl_sell_avg_pct,
        })

        # BearishDefenseStrategy
        bearish_defense = BearishDefenseStrategy({
            "brku_weight_pct": params.brku_weight_pct,
        })

        return cls(
            market_mode_filter=market_mode_filter,
            gold_filter=gold_filter,
            sideways_detector=sideways_detector,
            twin_pair=twin_pair,
            conditional_coin=conditional_coin,
            conditional_conl=conditional_conl,
            bearish_defense=bearish_defense,
            stop_loss_normal=params.stop_loss_pct,
            stop_loss_bullish=params.stop_loss_bullish_pct,
        )

    def generate_all_signals(
        self,
        changes: dict,
        poly_probs: dict | None = None,
        pairs: dict | None = None,
        sideways_active: bool = False,
    ) -> dict:
        """기존 generate_all_signals_vN()과 동일한 7-key dict를 반환한다.

        Parameters
        ----------
        changes : dict
            {ticker: {"change_pct": float, ...}, ...}
        poly_probs : dict | None
            Polymarket 확률
        pairs : dict | None
            쌍둥이 페어 설정
        sideways_active : bool
            횡보장 활성 여부

        Returns
        -------
        dict
            market_mode, gold, twin_pairs, conditional_coin,
            conditional_conl, stop_loss, bearish
        """
        if pairs is None:
            pairs = {}

        # 1. 시황 판단
        market_mode = self.market_mode_filter.evaluate(poly_probs, sideways_active)

        # 2. 금 시그널
        gld_data = changes.get("GLD", {})
        gld_pct = gld_data.get("change_pct", 0.0)
        gold = self.gold_filter.evaluate(gld_pct)

        # 3. 쌍둥이 페어
        twin_pairs = self.twin_pair.evaluate(changes, pairs)

        # 4. 조건부 COIN (sideways면 normal 모드로 처리)
        effective_mode = market_mode if market_mode != "sideways" else "normal"
        conditional_coin = self.conditional_coin.evaluate(changes, mode=effective_mode)

        # 5. 조건부 CONL
        conditional_conl = self.conditional_conl.evaluate(changes)

        # 6. 손절 (포지션 없는 시그널 단계 — 변동률 기반만 처리)
        stop_loss = self._check_stop_loss(changes, effective_mode)

        # 7. 하락장 방어
        bearish = self.bearish_defense.evaluate(effective_mode)

        return {
            "market_mode": market_mode,
            "gold": gold,
            "twin_pairs": twin_pairs,
            "conditional_coin": conditional_coin,
            "conditional_conl": conditional_conl,
            "stop_loss": stop_loss,
            "bearish": bearish,
        }

    def _check_stop_loss(self, changes: dict, mode: str) -> list[dict]:
        """시황 연동 손절 체크 — Legacy signals_v2.check_stop_loss_v2() 동일 로직."""
        threshold = self._stop_loss_bullish if mode == "bullish" else self._stop_loss_normal
        results: list[dict] = []

        for ticker, data in changes.items():
            pct = data.get("change_pct", 0.0)
            if pct <= threshold:
                results.append({
                    "ticker": ticker,
                    "change_pct": pct,
                    "stop_loss": True,
                    "threshold": threshold,
                    "message": (
                        f"{ticker} {pct:+.2f}% ≤ {threshold:.1f}% "
                        f"({mode} 모드) → 손절"
                    ),
                })

        return results

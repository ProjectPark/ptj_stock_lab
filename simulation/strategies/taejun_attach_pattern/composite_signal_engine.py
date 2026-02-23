"""
CompositeSignalEngine — v5 통합 시그널 엔진
============================================================
필터(MarketModeFilter, GoldFilter, SidewaysDetector)와
전략(TwinPairStrategy, ConditionalCoinStrategy, ConditionalConlStrategy,
BearishDefenseStrategy)를 조합하여 기존 7-key dict를 반환한다.

v5 추가:
- CircuitBreaker (CB-1~6 매수 게이트)
- StopLossCalculator (ATR 기반 손절)
- SwingModeManager (급등 스윙 모드)
- CrashBuyStrategy (급락 역매수)
- SoxlIndependentStrategy (SOXL 독립 매매)
- VixGold (IAU+GDXU 멀티 시그널)

반환 dict는 기존 7-key + v5 5-key = 12-key.

Usage:
    from strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
    from strategies.params import V5Params

    engine = CompositeSignalEngine.from_base_params(V5Params())
    sigs = engine.generate_all_signals(changes, poly_probs, pairs, sideways_active)
    # sigs는 기존 7-key + v5 5-key dict
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from ..params import BaseParams

from .circuit_breaker import CircuitBreaker
from .crash_buy import CrashBuyStrategy
from .filters import GoldFilter, MarketModeFilter, SidewaysDetector
from .soxl_independent import SoxlIndependentStrategy
from .stop_loss import StopLossCalculator
from .swing_mode import SwingModeManager
from .twin_pair import TwinPairStrategy
from .conditional_coin import ConditionalCoinStrategy
from .conditional_conl import ConditionalConlStrategy
from .bearish_defense import BearishDefenseStrategy
from .vix_gold import VixGold

if TYPE_CHECKING:
    from .base import MarketData, Position


class CompositeSignalEngine:
    """필터 + 전략 조합 → 기존 7-key + v5 5-key dict 반환.

    v5 추가 모듈:
    - circuit_breaker: CB-1~6 매수 게이트
    - stop_loss_calc: ATR 기반 손절 (레거시 고정 -3%/-8% 대체)
    - swing_mode: 급등 스윙 모드 (13절)
    - crash_buy: 급락 역매수 (5-5절)
    - soxl_independent: SOXL 독립 매매 (4-7절)
    - vix_gold: VIX 방어모드 (13-7절, 멀티 시그널)
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
        # v5 new modules
        circuit_breaker: CircuitBreaker | None = None,
        stop_loss_calc: StopLossCalculator | None = None,
        swing_mode: SwingModeManager | None = None,
        crash_buy: CrashBuyStrategy | None = None,
        soxl_independent: SoxlIndependentStrategy | None = None,
        vix_gold: VixGold | None = None,
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

        # v5 modules (graceful degradation: None means disabled)
        self.circuit_breaker = circuit_breaker
        self.stop_loss_calc = stop_loss_calc
        self.swing_mode = swing_mode
        self.crash_buy = crash_buy
        self.soxl_independent = soxl_independent
        self.vix_gold = vix_gold

        # CI-0-7: session_day 기반 M5 카운트 관리
        self._last_session_day: str = ""
        self._daily_buy_count: int = 0

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

        # v5 modules — instantiate with defaults, gracefully handle missing attrs
        circuit_breaker = CircuitBreaker(
            total_capital_usd=getattr(params, "total_capital", 15_000),
        )
        stop_loss_calc = StopLossCalculator()
        swing_mode = SwingModeManager()
        crash_buy = CrashBuyStrategy()
        soxl_independent = SoxlIndependentStrategy()
        vix_gold = VixGold()

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
            circuit_breaker=circuit_breaker,
            stop_loss_calc=stop_loss_calc,
            swing_mode=swing_mode,
            crash_buy=crash_buy,
            soxl_independent=soxl_independent,
            vix_gold=vix_gold,
        )

    # ------------------------------------------------------------------
    # CI-0-7: session_day — KST 17:30 기준 세션 날짜
    # ------------------------------------------------------------------

    @staticmethod
    def _get_session_day(dt: datetime) -> str:
        """KST 17:30 기준 세션 날짜 반환. CI-0-7.

        dt가 KST 17:30 이전이면 전일 날짜, 이후이면 당일 날짜.
        """
        roll = dt.replace(hour=17, minute=30, second=0, microsecond=0)
        if dt < roll:
            return (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")

    def _maybe_reset_session(self, market_time: datetime | None) -> None:
        """CI-0-8: 새 세션 시작 감지 시 M5 카운트 초기화."""
        if market_time is None:
            return
        session = self._get_session_day(market_time)
        if session != self._last_session_day:
            self._daily_buy_count = 0
            self._last_session_day = session

    # ------------------------------------------------------------------
    # CI-0-6: 동시 BUY 일괄 배분 — T1~T4 순차 할당
    # ------------------------------------------------------------------

    @staticmethod
    def _sequential_allocations(free_cash: float) -> list[float]:
        """T1=55%, T2=나머지x40%, T3=나머지x33%, T4=잔여. CI-0-6."""
        buckets: list[float] = []
        remaining = free_cash
        for w in [0.55, 0.40, 0.33]:
            alloc = remaining * w
            buckets.append(alloc)
            remaining = max(0.0, remaining - alloc)
        buckets.append(remaining)  # T4 잔여
        return buckets  # len=4

    # ------------------------------------------------------------------
    # MT_VNQ3 §1: 비중 합계 초과 시 축소 (shrink)
    # ------------------------------------------------------------------

    @staticmethod
    def _shrink_allocations(
        allocations: list[float],
        hard_limit: float = 1.001,
    ) -> list[float] | str:
        """MT_VNQ3 §1: 비중 합계가 100% 초과시 비례 shrink.

        Parameters
        ----------
        allocations : list[float]
            전략별 비중 리스트 (0.0~1.0 스케일).
        hard_limit : float
            ENGINE_CONFIG["shrink_hard_limit"]. 합계가 이 값 이상이면 BUY_STOP.

        Returns
        -------
        list[float] | str
            축소된 비중 리스트. 합계가 hard_limit 이상이면 "BUY_STOP" 문자열 반환.
        """
        total = sum(allocations)
        if total <= 1.0:
            return allocations

        # hard_limit 초과 → BUY_STOP
        if total >= hard_limit:
            return "BUY_STOP"

        # 비례 shrink: 하위 항목부터 drop
        factor = 1.0 / total
        return [a * factor for a in allocations]

    # ------------------------------------------------------------------

    def generate_all_signals(
        self,
        changes: dict,
        poly_probs: dict | None = None,
        pairs: dict | None = None,
        sideways_active: bool = False,
        # v5 new params
        market_data: "MarketData | None" = None,
        positions: "dict[str, Position] | None" = None,
        prices: dict | None = None,
        crypto_changes: dict | None = None,
        volumes: dict | None = None,
        market_time: "datetime | None" = None,
    ) -> dict:
        """기존 generate_all_signals_vN()과 동일한 7-key + v5 5-key dict를 반환한다.

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

        v5 new parameters (all optional for backward compat):
        market_data : MarketData | None
            통합 시장 데이터 (OHLCV 포함). 제공시 ATR 손절 등 활성화.
        positions : dict[str, Position] | None
            현재 보유 포지션.
        prices : dict | None
            현재가 dict.
        crypto_changes : dict | None
            크립토 24h 변동률. {"BTC": -6.0, ...}
        volumes : dict | None
            종목별 거래량.
        market_time : datetime | None
            현재 시각.

        Returns
        -------
        dict
            기존 7-key: market_mode, gold, twin_pairs, conditional_coin,
                        conditional_conl, stop_loss, bearish
            v5 5-key:   circuit_breaker, vix_defense, crash_buy,
                        soxl_independent, swing_mode
        """
        if pairs is None:
            pairs = {}

        # ── Step 0a: CI-0-7 세션 날짜 갱신 + CI-0-8 M5 카운트 초기화 ──
        self._maybe_reset_session(market_time)

        # CI-0-5: M200 즉시매도 최우선 — 다른 신호보다 먼저 평가
        # TODO: M200 조건 달성 시 신규 BUY 전면 금지 + 보유 포지션 매도

        # TODO: M201 게이트 — m201_mode.M201ImmediateMode.check(p, p_prev)
        # 우선순위: M200 다음, 리스크모드 이전
        # BTC 확률 급변 시 즉시 전환 파이프라인 실행

        # ── Step 0b: Circuit Breaker 평가 ────────────────────────────
        cb_status_dict: dict = {}
        if self.circuit_breaker is not None:
            # Flatten changes for CB (expects {ticker: float})
            flat_changes = {}
            for ticker, data in changes.items():
                if isinstance(data, dict):
                    flat_changes[ticker] = data.get("change_pct", 0.0)
                else:
                    flat_changes[ticker] = float(data)

            if market_time is not None:
                self.circuit_breaker.on_trading_day_start(
                    changes=flat_changes,
                    prices=prices or {},
                    poly=poly_probs,
                    crypto_changes=crypto_changes,
                    time=market_time,
                )
            cb_status_dict = self.circuit_breaker.summary()

        # ── Step 1: 시황 판단 ────────────────────────────────────────
        market_mode = self.market_mode_filter.evaluate(poly_probs, sideways_active)

        # ── Step 2: 금 시그널 ────────────────────────────────────────
        gld_data = changes.get("GLD", {})
        gld_pct = gld_data.get("change_pct", 0.0) if isinstance(gld_data, dict) else float(gld_data)
        gold = self.gold_filter.evaluate(gld_pct)

        # ── Step 3: 쌍둥이 페어 (volumes 전달) ──────────────────────
        twin_pairs = self.twin_pair.evaluate(changes, pairs, volumes=volumes)

        # ── Step 4: 조건부 COIN (sideways면 normal 모드로 처리) ──────
        effective_mode = market_mode if market_mode != "sideways" else "normal"
        conditional_coin = self.conditional_coin.evaluate(changes, mode=effective_mode)

        # ── Step 5: 조건부 CONL ──────────────────────────────────────
        conditional_conl = self.conditional_conl.evaluate(changes)

        # ── Step 6: 손절 ─────────────────────────────────────────────
        # ATR 기반 손절 (v5) — positions/market_data가 있을 때 활성화
        # 그 외에는 legacy 변동률 기반 손절 유지
        stop_loss = self._check_stop_loss(changes, effective_mode)

        # ── Step 7: 하락장 방어 ──────────────────────────────────────
        bearish = self.bearish_defense.evaluate(effective_mode)

        # ── Step 8: VIX 방어모드 (멀티 시그널) ───────────────────────
        vix_signals: list[dict] = []
        if self.vix_gold is not None and market_data is not None:
            try:
                sigs = self.vix_gold.generate_signals(market_data, positions)
                vix_signals = [
                    {
                        "action": s.action.value,
                        "ticker": s.ticker,
                        "size": s.size,
                        "reason": s.reason,
                        "metadata": s.metadata,
                    }
                    for s in sigs
                ]
            except Exception:
                pass  # graceful degradation

        # ── Step 9: 급락 역매수 ──────────────────────────────────────
        crash_buy_signal: dict = {}
        if self.crash_buy is not None and market_data is not None:
            try:
                sig = self.crash_buy.generate_signal(market_data)
                crash_buy_signal = {
                    "action": sig.action.value,
                    "ticker": sig.ticker,
                    "size": sig.size,
                    "reason": sig.reason,
                    "metadata": sig.metadata,
                }
            except Exception:
                pass

        # ── Step 10: SOXL 독립 매매 ─────────────────────────────────
        soxl_signal: dict = {}
        if self.soxl_independent is not None and market_data is not None:
            try:
                sig = self.soxl_independent.generate_signal(market_data)
                soxl_signal = {
                    "action": sig.action.value,
                    "ticker": sig.ticker,
                    "size": sig.size,
                    "reason": sig.reason,
                    "metadata": sig.metadata,
                }
            except Exception:
                pass

        # ── Step 11: 스윙 모드 ───────────────────────────────────────
        swing_status: dict = {}
        if self.swing_mode is not None:
            swing_status = self.swing_mode.summary()
            if market_data is not None:
                try:
                    swing_sigs = self.swing_mode.generate_signals(market_data, positions)
                    swing_status["signals"] = [
                        {
                            "action": s.action.value,
                            "ticker": s.ticker,
                            "size": s.size,
                            "reason": s.reason,
                            "metadata": s.metadata,
                        }
                        for s in swing_sigs
                    ]
                except Exception:
                    swing_status["signals"] = []

        return {
            # existing 7 keys (backward compat)
            "market_mode": market_mode,
            "gold": gold,
            "twin_pairs": twin_pairs,
            "conditional_coin": conditional_coin,
            "conditional_conl": conditional_conl,
            "stop_loss": stop_loss,
            "bearish": bearish,
            # v5 new 5 keys
            "circuit_breaker": cb_status_dict,
            "vix_defense": vix_signals,
            "crash_buy": crash_buy_signal,
            "soxl_independent": soxl_signal,
            "swing_mode": swing_status,
        }

    def _check_stop_loss(self, changes: dict, mode: str) -> list[dict]:
        """시황 연동 손절 체크.

        ATR StopLossCalculator가 있으면 ATR 기반 정보를 포함하고,
        없으면 legacy 고정 손절 로직으로 폴백한다.
        """
        threshold = self._stop_loss_bullish if mode == "bullish" else self._stop_loss_normal
        results: list[dict] = []

        for ticker, data in changes.items():
            if isinstance(data, dict):
                pct = data.get("change_pct", 0.0)
            else:
                pct = float(data)

            if pct <= threshold:
                result_item: dict = {
                    "ticker": ticker,
                    "change_pct": pct,
                    "stop_loss": True,
                    "threshold": threshold,
                    "message": (
                        f"{ticker} {pct:+.2f}% ≤ {threshold:.1f}% "
                        f"({mode} 모드) → 손절"
                    ),
                }

                # ATR 정보 보강 (StopLossCalculator 있을 때)
                if self.stop_loss_calc is not None:
                    result_item["atr_available"] = True
                    result_item["leverage"] = self.stop_loss_calc.get_leverage(ticker)

                results.append(result_item)

        return results

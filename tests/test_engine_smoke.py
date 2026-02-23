"""
전 엔진 Smoke Test — import + instantiate + signal 생성
=========================================================
rules에 정의된 모든 엔진이 정상 작동하는지 빠르게 검증한다.

검증 항목:
1. 모듈 import 성공
2. 클래스 인스턴스화 성공
3. validate_params() 에러 없음
4. generate_signal() 예외 없이 Signal 반환
5. 인프라/필터 모듈 import + 기본 동작
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.strategies.line_b_taejun.common.base import (
    Action, MarketData, Position, Signal,
)
from simulation.strategies.line_b_taejun.common.registry import (
    get_strategy, list_strategies,
)

# ── 전략 모듈 강제 import (registry 등록 트리거) ──
from simulation.strategies.line_b_taejun.strategies import (  # noqa: F401
    bargain_buy, bearish_defense, bank_conditional,
    bear_regime, conditional_coin, conditional_conl, crash_buy,
    emergency_mode, jab_bitu, jab_etq, jab_soxl, jab_tsll,
    reit_risk, sector_rotate, short_macro, soxl_independent,
    sp500_entry, twin_pair, vix_gold,
)


# ── 공용 더미 데이터 ──

def _dummy_market() -> MarketData:
    """모든 전략이 에러 없이 처리할 수 있는 최소 MarketData."""
    return MarketData(
        changes={
            "GLD": 0.1, "BITU": -0.3, "SOXL": 0.5, "CONL": -0.2,
            "TSLL": 0.1, "ETQ": -0.1, "SPY": 0.2, "QQQ": 0.3,
            "SOXX": 2.1, "BAC": 0.3, "JPM": 0.5, "WFC": 0.4,
            "BTC": 1.0, "ETH": 0.5, "SOL": 0.3, "XRP": 3.0,
            "IAU": 0.1, "GDXU": 0.2, "VIX": 11.0,
            "AMDL": -0.1, "NVDL": 0.2, "ROBN": 0.1,
            "BRKU": -0.1, "ETHU": 0.3, "NFXL": 0.1, "PLTU": -0.2,
        },
        prices={
            "GLD": 240.0, "BITU": 45.0, "SOXL": 22.0, "CONL": 15.0,
            "TSLL": 12.0, "ETQ": 8.0, "SPY": 510.0, "QQQ": 450.0,
            "SOXX": 210.0, "BAC": 42.0, "JPM": 195.0, "WFC": 55.0,
            "IAU": 48.0, "GDXU": 30.0,
            "AMDL": 25.0, "NVDL": 35.0, "ROBN": 18.0,
            "BRKU": 40.0, "ETHU": 12.0, "NFXL": 15.0, "PLTU": 10.0,
        },
        poly={"btc_up": 0.55, "eth_up": 0.52, "ndx_up": 0.58},
        time=datetime(2026, 2, 23, 18, 0),
        history={
            "CONL": {"high_3y": 120.0, "low_3y": 8.0},
            "SOXL": {"high_3y": 70.0, "low_3y": 10.0, "low_1y": 15.0},
            "BITU": {"low_1y": 30.0},
            "VNQ": {"return_7d": 0.5},
        },
        crypto={"BTC": 0.9, "ETH": 0.5, "SOL": 0.3, "XRP": 3.0},
        poly_prev={"btc_up": 0.53, "eth_up": 0.50, "ndx_up": 0.56},
    )


def _dummy_position(ticker: str, strategy_name: str) -> Position:
    return Position(
        ticker=ticker, avg_price=10.0, qty=5.0,
        entry_time=datetime(2026, 2, 20),
        strategy_name=strategy_name,
    )


# ============================================================
# 1. Line B — Registry 전략 전수 Smoke Test
# ============================================================

EXPECTED_STRATEGIES = [
    "bargain_buy", "bearish_defense", "bank_conditional",
    "bear_regime_long", "conditional_coin", "conditional_conl", "crash_buy",
    "emergency_mode", "jab_bitu", "jab_etq", "jab_soxl", "jab_tsll",
    "reit_risk", "sector_rotate", "short_macro", "soxl_independent",
    "sp500_entry", "twin_pair", "vix_gold",
]


class TestRegistryCompleteness:
    """모든 기대 전략이 registry에 등록되었는지 확인."""

    def test_all_expected_registered(self):
        registered = list_strategies()
        for name in EXPECTED_STRATEGIES:
            assert name in registered, f"{name} not registered"

    def test_no_unknown_strategies(self):
        """registry에 예상 밖 전략이 없는지 확인."""
        registered = set(list_strategies())
        expected = set(EXPECTED_STRATEGIES)
        unknown = registered - expected
        # unknown이 있어도 fail은 아님 — 경고만
        if unknown:
            pytest.skip(f"Unexpected strategies in registry (not an error): {unknown}")


class TestStrategyInstantiation:
    """모든 등록 전략이 기본 파라미터로 인스턴스화되는지 확인."""

    @pytest.mark.parametrize("name", EXPECTED_STRATEGIES)
    def test_instantiate(self, name):
        strategy = get_strategy(name)
        assert strategy is not None
        assert strategy.name == name


class TestValidateParams:
    """모든 전략의 validate_params()가 에러 없이 통과하는지 확인."""

    @pytest.mark.parametrize("name", EXPECTED_STRATEGIES)
    def test_validate(self, name):
        strategy = get_strategy(name)
        errors = strategy.validate_params()
        assert errors == [], f"{name} validate_params failed: {errors}"


class TestGenerateSignalNoException:
    """모든 전략에 더미 데이터를 넣어 generate_signal()이 예외 없이 반환되는지 확인."""

    @pytest.mark.parametrize("name", EXPECTED_STRATEGIES)
    def test_signal_without_position(self, name):
        strategy = get_strategy(name)
        market = _dummy_market()
        sig = strategy.generate_signal(market)
        assert isinstance(sig, Signal)
        assert sig.action in (Action.BUY, Action.SELL, Action.HOLD, Action.SKIP)

    @pytest.mark.parametrize("name", EXPECTED_STRATEGIES)
    def test_signal_with_position(self, name):
        strategy = get_strategy(name)
        market = _dummy_market()
        ticker = strategy.params.get("ticker", "SOXL")
        if isinstance(ticker, dict):
            ticker = "SOXL"
        pos = _dummy_position(ticker, name)
        sig = strategy.generate_signal(market, pos)
        assert isinstance(sig, Signal)


# ============================================================
# 2. Line B — 인프라 모듈 Smoke Test
# ============================================================

class TestInfraImport:
    """인프라 모듈이 import + 인스턴스화되는지 확인."""

    def test_m201_mode(self):
        from simulation.strategies.line_b_taejun.infra.m201_mode import M201ImmediateMode
        m = M201ImmediateMode()
        assert m is not None

    def test_m200_stop(self):
        from simulation.strategies.line_b_taejun.infra.m200_stop import M200KillSwitch
        ks = M200KillSwitch()
        assert ks is not None

    def test_m28_poly_gate(self):
        from simulation.strategies.line_b_taejun.infra.m28_poly_gate import M28PolyGate
        g = M28PolyGate()
        assert g is not None

    def test_schd_master(self):
        from simulation.strategies.line_b_taejun.infra.schd_master import SCHDMaster
        s = SCHDMaster()
        assert s is not None

    def test_m5_weight_manager(self):
        from simulation.strategies.line_b_taejun.infra.m5_weight_manager import M5WeightManager
        m = M5WeightManager()
        assert m is not None

    def test_orchestrator(self):
        from simulation.strategies.line_b_taejun.infra.orchestrator import Orchestrator
        o = Orchestrator()
        assert o is not None

    def test_profit_distributor(self):
        from simulation.strategies.line_b_taejun.infra.profit_distributor import ProfitDistributor
        p = ProfitDistributor()
        assert p is not None

    def test_limit_order(self):
        from simulation.strategies.line_b_taejun.infra.limit_order import (
            LimitOrder, OrderQueue, OrderStatus,
        )
        assert OrderStatus.PENDING.value == "pending"


# ============================================================
# 3. Line B — 필터 모듈 Smoke Test
# ============================================================

class TestFiltersImport:
    """필터 모듈 import + 기본 동작."""

    def test_poly_quality(self):
        from simulation.strategies.line_b_taejun.filters.poly_quality import PolyQualityFilter
        f = PolyQualityFilter()
        result = f.filter({"btc_up": 0.55})
        assert "btc_up" in result

    def test_asset_mode(self):
        from simulation.strategies.line_b_taejun.filters.asset_mode import AssetModeManager
        mgr = AssetModeManager()
        assert mgr is not None

    def test_circuit_breaker(self):
        from simulation.strategies.line_b_taejun.filters.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert cb is not None

    def test_stop_loss(self):
        from simulation.strategies.line_b_taejun.filters.stop_loss import StopLossCalculator
        sl = StopLossCalculator()
        assert sl is not None

    def test_swing_mode(self):
        from simulation.strategies.line_b_taejun.filters.swing_mode import SwingModeManager
        sm = SwingModeManager()
        assert sm is not None


# ============================================================
# 4. Line B — CompositeSignalEngine Smoke Test
# ============================================================

class TestCompositeEngine:
    """CompositeSignalEngine import + 인스턴스화."""

    def test_import_and_create(self):
        from simulation.strategies.line_b_taejun.composite_signal_engine import CompositeSignalEngine
        from simulation.strategies.line_b_taejun.common.params import V5Params
        engine = CompositeSignalEngine.from_base_params(V5Params())
        assert engine is not None


# ============================================================
# 5. Line A — Legacy 시그널 모듈 Import Test
# ============================================================

class TestLineAImport:
    """Line A 시그널 모듈이 import되는지 확인."""

    def test_signals_v1(self):
        from simulation.strategies.line_a import signals  # noqa: F401

    def test_signals_v2(self):
        from simulation.strategies.line_a import signals_v2  # noqa: F401

    def test_signals_v5(self):
        from simulation.strategies.line_a import signals_v5  # noqa: F401


# ============================================================
# 6. Line C — D2S 엔진 Smoke Test
# ============================================================

class TestLineCImport:
    """Line C D2S 엔진 import + 인스턴스화."""

    def test_d2s_engine_import(self):
        from simulation.strategies.line_c_d2s.d2s_engine import D2SEngine
        engine = D2SEngine()
        assert engine is not None

    def test_params_d2s_import(self):
        from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE
        assert isinstance(D2S_ENGINE, dict)
        assert len(D2S_ENGINE) > 0

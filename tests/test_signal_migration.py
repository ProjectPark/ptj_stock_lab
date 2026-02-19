"""
Legacy vs OOP Signal 단위 비교 테스트
=====================================
동일 입력에 대해 Legacy 함수와 OOP 클래스의 출력이 동일한지 검증한다.

Usage:
    pyenv shell ptj_stock_lab && pytest tests/test_signal_migration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def sample_changes():
    """테스트용 변동률 데이터."""
    return {
        "GLD": {"change_pct": 0.5},
        "BITU": {"change_pct": 3.0},
        "MSTU": {"change_pct": 0.5},
        "ETHU": {"change_pct": 5.0},
        "XXRP": {"change_pct": 5.0},
        "SOLT": {"change_pct": 5.0},
        "COIN": {"change_pct": 1.0},
        "CONL": {"change_pct": 2.0},
        "SPY": {"change_pct": 0.1},
        "QQQ": {"change_pct": 0.1},
        "VIX": {"change_pct": -0.5},
        "BRKU": {"change_pct": -0.3},
    }


@pytest.fixture
def sample_pairs():
    return {
        "coin": {"lead": "BITU", "follow": ["MSTU"], "label": "코인"},
    }


@pytest.fixture
def sample_poly():
    return {"btc_up": 0.55, "ndx_up": 0.50, "eth_up": 0.45}


# ============================================================
# MarketModeFilter vs determine_market_mode
# ============================================================

class TestMarketModeFilter:
    def test_normal(self):
        from simulation.strategies.taejun_attach_pattern.filters import MarketModeFilter
        from simulation.strategies import signals_v2
        f = MarketModeFilter()
        poly = {"btc_up": 0.55, "ndx_up": 0.50, "eth_up": 0.45}
        assert f.evaluate(poly) == signals_v2.determine_market_mode(poly)

    def test_bullish(self):
        from simulation.strategies.taejun_attach_pattern.filters import MarketModeFilter
        from simulation.strategies import signals_v2
        f = MarketModeFilter()
        poly = {"btc_up": 0.75, "ndx_up": 0.60, "eth_up": 0.55}
        assert f.evaluate(poly) == "bullish"
        assert f.evaluate(poly) == signals_v2.determine_market_mode(poly)

    def test_bearish(self):
        from simulation.strategies.taejun_attach_pattern.filters import MarketModeFilter
        from simulation.strategies import signals_v2
        f = MarketModeFilter()
        poly = {"btc_up": 0.15, "ndx_up": 0.10, "eth_up": 0.18}
        assert f.evaluate(poly) == "bearish"
        assert f.evaluate(poly) == signals_v2.determine_market_mode(poly)

    def test_none(self):
        from simulation.strategies.taejun_attach_pattern.filters import MarketModeFilter
        from simulation.strategies import signals_v2
        f = MarketModeFilter()
        assert f.evaluate(None) == "normal"
        assert f.evaluate(None) == signals_v2.determine_market_mode(None)

    def test_sideways(self):
        from simulation.strategies.taejun_attach_pattern.filters import MarketModeFilter
        f = MarketModeFilter()
        assert f.evaluate({"btc_up": 0.75}, sideways_active=True) == "sideways"


# ============================================================
# GoldFilter vs check_gold_signal_v2
# ============================================================

class TestGoldFilter:
    @pytest.mark.parametrize("gld_pct", [0.5, -0.3, 0.0, 1.2, -2.5])
    def test_matches_legacy(self, gld_pct):
        from simulation.strategies.taejun_attach_pattern.filters import GoldFilter
        from simulation.strategies import signals_v2
        f = GoldFilter()
        oop = f.evaluate(gld_pct)
        legacy = signals_v2.check_gold_signal_v2(gld_pct)
        assert oop == legacy, f"Mismatch at gld_pct={gld_pct}: {oop} vs {legacy}"


# ============================================================
# SidewaysDetector vs evaluate_sideways
# ============================================================

class TestSidewaysDetector:
    def test_matches_v5(self, sample_changes, sample_poly):
        from simulation.strategies.taejun_attach_pattern.filters import SidewaysDetector
        from simulation.strategies import signals_v5
        d = SidewaysDetector()
        oop = d.evaluate(poly_probs=sample_poly, changes=sample_changes,
                         gap_fail_count=3, trigger_fail_count=3)
        legacy = signals_v5.evaluate_sideways(
            poly_probs=sample_poly, changes=sample_changes,
            gap_fail_count=3, trigger_fail_count=3)
        assert oop["is_sideways"] == legacy["is_sideways"]
        assert oop["count"] == legacy["count"]
        assert oop["indicators"] == legacy["indicators"]

    def test_v4_dual_path(self):
        """v4 dual-path: indicators dict를 직접 전달하는 경우."""
        from simulation.strategies.taejun_attach_pattern.filters import SidewaysDetector
        indicators = {"poly_range": True, "gld_flat": True, "gap_fail": True,
                      "trigger_fail": False, "index_flat": False}
        d = SidewaysDetector()
        oop = d.evaluate(indicators=indicators)
        # 3/5 충족 → sideways
        assert oop["is_sideways"] is True
        assert oop["count"] == 3


# ============================================================
# TwinPairStrategy vs check_twin_pairs_v2/v5
# ============================================================

class TestTwinPairStrategy:
    def test_matches_v2(self, sample_changes, sample_pairs):
        from simulation.strategies.taejun_attach_pattern.twin_pair import TwinPairStrategy
        from simulation.strategies import signals_v2
        s = TwinPairStrategy({"entry_threshold": 1.5, "sell_threshold": 0.9})
        oop = s.evaluate(sample_changes, sample_pairs)
        legacy = signals_v2.check_twin_pairs_v2(sample_changes, sample_pairs)
        assert len(oop) == len(legacy)
        for o, l in zip(oop, legacy):
            assert o["signal"] == l["signal"]
            assert abs(o["gap"] - l["gap"]) < 0.01

    def test_matches_v5(self, sample_changes, sample_pairs):
        from simulation.strategies.taejun_attach_pattern.twin_pair import TwinPairStrategy
        from simulation.strategies import signals_v5
        s = TwinPairStrategy({"entry_threshold": 2.2, "sell_threshold": 0.9})
        oop = s.evaluate(sample_changes, sample_pairs)
        legacy = signals_v5.check_twin_pairs_v5(sample_changes, sample_pairs)
        assert len(oop) == len(legacy)
        for o, l in zip(oop, legacy):
            assert o["signal"] == l["signal"]


# ============================================================
# ConditionalCoinStrategy vs check_conditional_coin_v2
# ============================================================

class TestConditionalCoinStrategy:
    def test_matches_v2(self, sample_changes):
        from simulation.strategies.taejun_attach_pattern.conditional_coin import ConditionalCoinStrategy
        from simulation.strategies import signals_v2
        s = ConditionalCoinStrategy({"trigger_pct": 3.0, "sell_profit_pct": 3.0,
                                      "sell_bearish_pct": 0.3})
        oop = s.evaluate(sample_changes, mode="normal")
        legacy = signals_v2.check_conditional_coin_v2(sample_changes, mode="normal")
        assert oop["buy_signal"] == legacy["buy_signal"]
        assert oop["all_above_threshold"] == legacy["all_above_threshold"]
        assert abs(oop["trigger_avg_pct"] - legacy["trigger_avg_pct"]) < 0.01

    def test_bearish_mode(self, sample_changes):
        from simulation.strategies.taejun_attach_pattern.conditional_coin import ConditionalCoinStrategy
        from simulation.strategies import signals_v2
        s = ConditionalCoinStrategy({"trigger_pct": 3.0, "sell_profit_pct": 3.0,
                                      "sell_bearish_pct": 0.3})
        oop = s.evaluate(sample_changes, mode="bearish")
        legacy = signals_v2.check_conditional_coin_v2(sample_changes, mode="bearish")
        assert oop["sell_target_pct"] == legacy["sell_target_pct"]


# ============================================================
# ConditionalConlStrategy vs check_conditional_conl_v2
# ============================================================

class TestConditionalConlStrategy:
    def test_matches_v2(self, sample_changes):
        from simulation.strategies.taejun_attach_pattern.conditional_conl import ConditionalConlStrategy
        from simulation.strategies import signals_v2
        s = ConditionalConlStrategy({"trigger_pct": 3.0, "sell_profit_pct": 2.8,
                                      "sell_avg_pct": 1.0})
        oop = s.evaluate(sample_changes)
        legacy = signals_v2.check_conditional_conl_v2(sample_changes)
        assert oop["buy_signal"] == legacy["buy_signal"]
        assert abs(oop["trigger_avg_pct"] - legacy["trigger_avg_pct"]) < 0.01
        assert oop["sell_on_avg_drop"] == legacy["sell_on_avg_drop"]


# ============================================================
# BearishDefenseStrategy vs check_bearish_v2
# ============================================================

class TestBearishDefenseStrategy:
    @pytest.mark.parametrize("mode", ["normal", "bullish", "bearish"])
    def test_matches_v2(self, mode):
        from simulation.strategies.taejun_attach_pattern.bearish_defense import BearishDefenseStrategy
        from simulation.strategies import signals_v2
        s = BearishDefenseStrategy({"brku_weight_pct": 10.0})
        oop = s.evaluate(mode)
        legacy = signals_v2.check_bearish_v2(mode)
        assert oop["buy_brku"] == legacy["buy_brku"]
        assert oop["brku_weight_pct"] == legacy["brku_weight_pct"]


# ============================================================
# CompositeSignalEngine vs generate_all_signals_v2/v5
# ============================================================

class TestCompositeSignalEngine:
    def test_matches_v2(self, sample_changes, sample_pairs, sample_poly):
        from simulation.strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
        from simulation.strategies.params import BaseParams
        from simulation.strategies import signals_v2

        engine = CompositeSignalEngine.from_base_params(BaseParams())
        oop = engine.generate_all_signals(sample_changes, sample_poly, sample_pairs)
        legacy = signals_v2.generate_all_signals_v2(sample_changes, sample_poly, sample_pairs)

        assert oop["market_mode"] == legacy["market_mode"]
        assert oop["gold"] == legacy["gold"]
        assert oop["conditional_coin"]["buy_signal"] == legacy["conditional_coin"]["buy_signal"]
        assert oop["conditional_conl"]["buy_signal"] == legacy["conditional_conl"]["buy_signal"]
        assert oop["bearish"]["buy_brku"] == legacy["bearish"]["buy_brku"]
        assert len(oop["twin_pairs"]) == len(legacy["twin_pairs"])

    def test_matches_v5(self, sample_changes, sample_pairs, sample_poly):
        from simulation.strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
        from simulation.strategies.params import V5Params
        from simulation.strategies import signals_v5

        engine = CompositeSignalEngine.from_base_params(V5Params())
        oop = engine.generate_all_signals(sample_changes, sample_poly, sample_pairs,
                                          sideways_active=False)
        legacy = signals_v5.generate_all_signals_v5(sample_changes, sample_poly, sample_pairs,
                                                     sideways_active=False)

        assert oop["market_mode"] == legacy["market_mode"]
        assert oop["gold"] == legacy["gold"]
        assert oop["conditional_coin"]["buy_signal"] == legacy["conditional_coin"]["buy_signal"]
        assert oop["conditional_conl"]["buy_signal"] == legacy["conditional_conl"]["buy_signal"]
        assert oop["bearish"]["buy_brku"] == legacy["bearish"]["buy_brku"]

    def test_seven_keys(self, sample_changes):
        from simulation.strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
        from simulation.strategies.params import V5Params

        engine = CompositeSignalEngine.from_base_params(V5Params())
        sigs = engine.generate_all_signals(sample_changes)
        expected_keys = {"market_mode", "gold", "twin_pairs", "conditional_coin",
                         "conditional_conl", "stop_loss", "bearish"}
        assert set(sigs.keys()) == expected_keys

    def test_sideways_mode(self, sample_changes, sample_poly):
        from simulation.strategies.taejun_attach_pattern.composite_signal_engine import CompositeSignalEngine
        from simulation.strategies.params import V5Params

        engine = CompositeSignalEngine.from_base_params(V5Params())
        sigs = engine.generate_all_signals(sample_changes, sample_poly,
                                           sideways_active=True)
        assert sigs["market_mode"] == "sideways"

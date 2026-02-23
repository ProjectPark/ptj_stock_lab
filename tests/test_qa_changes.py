"""
QA Q1~Q11 반영 검증 테스트
===========================
2026-02-19 전략 리뷰에서 발생한 변경사항을 검증한다.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# 프로젝트 루트를 path에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.strategies.line_b_taejun.common.params import (
    ASSET_MODE,
    BARGAIN_BUY,
    EMERGENCY_MODE,
    JAB_BITU,
    JAB_ETQ,
    JAB_TSLL,
    POLY_QUALITY,
    REIT_RISK,
    SECTOR_ROTATE,
)
from simulation.strategies.line_b_taejun.common.base import (
    Action,
    MarketData,
    Position,
)
from simulation.strategies.line_b_taejun.strategies.bargain_buy import BargainBuy
from simulation.strategies.line_b_taejun.strategies.reit_risk import ReitRisk
from simulation.strategies.line_b_taejun.strategies.sector_rotate import SectorRotate
from simulation.strategies.line_b_taejun.strategies.emergency_mode import EmergencyMode
from simulation.strategies.line_b_taejun.strategies.jab_etq import JabETQ
from simulation.strategies.line_b_taejun.filters.poly_quality import PolyQualityFilter
from simulation.strategies.line_b_taejun.filters.asset_mode import (
    AssetMode,
    AssetModeManager,
)


# ============================================================
# Phase 1: 단순 파라미터 변경
# ============================================================


class TestQ1_XRP_Threshold:
    """Q-1: XRP 임계값 5.0% → 2.5%"""

    def test_xrp_threshold_is_2_5(self):
        assert JAB_BITU["crypto_conditions"]["XRP"] == 2.5


class TestQ5_GLD_Max:
    """Q-5: TSLL GLD 조건 → 0.1% (MT_VNQ3 반영)"""

    def test_gld_max_is_0_1(self):
        assert JAB_TSLL["gld_max"] == 0.1


class TestQ10_SNXX_OKLL_Removed:
    """Q-10: SNXX, OKLL 제거"""

    def test_snxx_not_in_bargain_buy(self):
        assert "SNXX" not in BARGAIN_BUY["tickers"]

    def test_okll_not_in_bargain_buy(self):
        assert "OKLL" not in BARGAIN_BUY["tickers"]

    def test_existing_tickers_preserved(self):
        """기존 종목은 유지되어야 한다."""
        for ticker in ["CONL", "SOXL", "AMDL", "NVDL", "ROBN", "ETHU", "BRKU", "NFXL", "PLTU"]:
            assert ticker in BARGAIN_BUY["tickers"], f"{ticker} missing"


class TestQ6_Q7_Confirmed:
    """Q-6, Q-7: MT_VNQ3 반영 — BRKU -31, ETHU add_size=0"""

    def test_brku_drop_pct(self):
        assert BARGAIN_BUY["tickers"]["BRKU"]["drop_pct"] == -31

    def test_ethu_add_size(self):
        assert BARGAIN_BUY["tickers"]["ETHU"]["add_size"] == 0


# ============================================================
# Phase 2: 로직 변경
# ============================================================


class TestQ2_ETQ_Entry:
    """Q-2: ETQ 진입 조건 — Polymarket 하락기대 스프레드 12pp"""

    def _make_market(self, poly: dict) -> MarketData:
        return MarketData(
            changes={"GLD": 0.05, "ETQ": 0.01},
            prices={"ETQ": 10.0},
            poly=poly,
            time=datetime(2026, 2, 19, 18, 0),
        )

    def test_param_renamed(self):
        """poly_eth_down_min → poly_down_spread_min"""
        assert "poly_down_spread_min" in JAB_ETQ
        assert "poly_eth_down_min" not in JAB_ETQ

    def test_entry_when_spread_high(self):
        """하락기대 스프레드 >= 12pp → 진입"""
        strat = JabETQ()
        # btc_up=0.5 → down=50, eth_up=0.1 → down=90, ndx_up=0.5 → down=50
        # max_down=90, avg=63.3, spread=26.7 >= 12 → True
        market = self._make_market({"btc_up": 0.5, "eth_up": 0.1, "ndx_up": 0.5})
        assert strat.check_entry(market) is True

    def test_no_entry_when_spread_low(self):
        """하락기대 스프레드 < 12pp → 미진입"""
        strat = JabETQ()
        # btc_up=0.5 → down=50, eth_up=0.45 → down=55, ndx_up=0.5 → down=50
        # max_down=55, avg=51.67, spread=3.33 < 12 → False
        market = self._make_market({"btc_up": 0.5, "eth_up": 0.45, "ndx_up": 0.5})
        assert strat.check_entry(market) is False

    def test_no_entry_without_poly(self):
        market = MarketData(
            changes={"GLD": 0.05, "ETQ": 0.01},
            prices={"ETQ": 10.0}, poly=None,
            time=datetime(2026, 2, 19),
        )
        strat = JabETQ()
        assert strat.check_entry(market) is False


class TestQ3Q4_Deadline:
    """Q-3/Q-4: CONL/SOXL 기한 + 30일 연장"""

    def test_conl_has_deadline(self):
        cfg = BARGAIN_BUY["tickers"]["CONL"]
        assert cfg["deadline_days"] > 0
        assert cfg["deadline_extension"] == 30

    def test_soxl_has_deadline(self):
        cfg = BARGAIN_BUY["tickers"]["SOXL"]
        assert cfg["deadline_days"] > 0
        assert cfg["deadline_extension"] == 30

    def test_deadline_extension_then_sell(self):
        """기한 도달 → 1회 연장(HOLD) → 연장 소진 → 매도"""
        strat = BargainBuy()
        entry_time = datetime(2025, 1, 1)
        deadline = BARGAIN_BUY["tickers"]["CONL"]["deadline_days"]
        extension = BARGAIN_BUY["tickers"]["CONL"]["deadline_extension"]

        pos = Position(
            ticker="CONL", avg_price=10.0, qty=5.0,
            entry_time=entry_time, strategy_name="bargain_buy",
        )

        # 기한 직전 — HOLD (수익률 미달)
        market_before = MarketData(
            changes={}, prices={"CONL": 10.5},
            poly=None,
            time=entry_time + timedelta(days=deadline - 1),
            history={"CONL": {"high_3y": 100.0}},
        )
        sig = strat.generate_signal(market_before, pos)
        assert sig.action == Action.HOLD

        # 기한 도달 → 1회 연장 (HOLD)
        market_deadline = MarketData(
            changes={}, prices={"CONL": 10.5},
            poly=None,
            time=entry_time + timedelta(days=deadline),
            history={"CONL": {"high_3y": 100.0}},
        )
        sig = strat.generate_signal(market_deadline, pos)
        assert sig.action == Action.HOLD
        assert sig.metadata.get("deadline_extended") is True

        # 연장 기한 소진 → 매도
        market_expired = MarketData(
            changes={}, prices={"CONL": 10.5},
            poly=None,
            time=entry_time + timedelta(days=deadline + extension),
            history={"CONL": {"high_3y": 100.0}},
        )
        sig = strat.generate_signal(market_expired, pos)
        assert sig.action == Action.SELL
        assert sig.metadata.get("deadline_expired") is True


class TestQ9_ReitRisk:
    """Q-9: 조심모드 — 3개 리츠 전부 7일 수익률 ≥ 1%"""

    def test_param_changed(self):
        """reits_avg_min/fear_greed_min 제거, reits_7d_return_min 추가"""
        cond = REIT_RISK["conditions"]
        assert "reits_7d_return_min" in cond
        assert "reits_avg_min" not in cond
        assert "fear_greed_min" not in cond

    def test_cautious_mode_in_params(self):
        assert "cautious_mode" in REIT_RISK
        assert REIT_RISK["cautious_mode"]["attack_leverage_pct"] == 50

    def test_entry_all_reits_above_1pct(self):
        """3개 리츠 전부 7일 수익률 >= 1% → 트리거"""
        strat = ReitRisk()
        market = MarketData(
            changes={}, prices={}, poly=None,
            time=datetime(2026, 2, 19),
            history={
                "SK리츠": {"return_7d": 1.5},
                "TIGER 리츠부동산인프라": {"return_7d": 1.2},
                "롯데리츠": {"return_7d": 1.0},
            },
        )
        assert strat.check_entry(market) is True

    def test_no_entry_one_below(self):
        """REIT_MIX 평균 1% 미만 → 미트리거 (MT_VNQ3: VNQ + KR aux 평균)"""
        strat = ReitRisk()
        market = MarketData(
            changes={}, prices={}, poly=None,
            time=datetime(2026, 2, 19),
            history={
                "VNQ": {"return_7d": 0.5},
                "SK리츠": {"return_7d": 0.3},
                "TIGER 리츠부동산인프라": {"return_7d": 0.2},
                "롯데리츠": {"return_7d": 0.4},
            },
        )
        assert strat.check_entry(market) is False

    def test_signal_has_cautious_metadata(self):
        """트리거시 cautious_mode 메타데이터 포함"""
        strat = ReitRisk()
        market = MarketData(
            changes={}, prices={}, poly=None,
            time=datetime(2026, 2, 19),
            history={
                "SK리츠": {"return_7d": 2.0},
                "TIGER 리츠부동산인프라": {"return_7d": 1.5},
                "롯데리츠": {"return_7d": 1.1},
            },
        )
        sig = strat.generate_signal(market)
        assert sig.metadata.get("cautious_mode") is True
        assert sig.metadata.get("attack_leverage_pct") == 50


# ============================================================
# Phase 3: 신규 기능
# ============================================================


class TestEmergencyMode:
    """이머전시 모드 — Polymarket 30pp+ 급변"""

    def test_params_exist(self):
        assert EMERGENCY_MODE["poly_swing_min"] == 30.0
        assert "btc_surge" in EMERGENCY_MODE
        assert "ndx_bull" in EMERGENCY_MODE
        assert "ndx_bear" in EMERGENCY_MODE

    def test_detect_swing(self):
        strat = EmergencyMode()
        market = MarketData(
            changes={}, prices={},
            poly={"btc_up": 0.90},
            poly_prev={"btc_up": 0.50},
            time=datetime(2026, 2, 19),
        )
        key, swing = strat._detect_poly_swing(market)
        assert key == "btc_up"
        assert abs(swing - 40.0) < 0.1

    def test_entry_triggers_on_large_swing(self):
        strat = EmergencyMode()
        market = MarketData(
            changes={}, prices={},
            poly={"btc_up": 0.90},
            poly_prev={"btc_up": 0.50},
            time=datetime(2026, 2, 19),
        )
        assert strat.check_entry(market) is True

    def test_no_entry_on_small_swing(self):
        strat = EmergencyMode()
        market = MarketData(
            changes={}, prices={},
            poly={"btc_up": 0.55},
            poly_prev={"btc_up": 0.50},
            time=datetime(2026, 2, 19),
        )
        assert strat.check_entry(market) is False

    def test_sub_mode_btc_surge(self):
        strat = EmergencyMode()
        market = MarketData(
            changes={}, prices={"BITU": 50.0},
            poly={"btc_up": 0.90},
            poly_prev={"btc_up": 0.50},
            time=datetime(2026, 2, 19),
        )
        sig = strat.generate_signal(market)
        assert sig.action == Action.BUY
        assert sig.ticker == "BITU"
        assert sig.metadata.get("emergency_mode") is True

    def test_profitable_position_sold(self):
        strat = EmergencyMode()
        strat._triggered = True
        pos = Position(
            ticker="SOXL", avg_price=10.0, qty=5.0,
            entry_time=datetime(2026, 2, 18),
            strategy_name="jab_soxl",
        )
        market = MarketData(
            changes={}, prices={"SOXL": 11.0},
            poly={"btc_up": 0.90},
            poly_prev={"btc_up": 0.50},
            time=datetime(2026, 2, 19),
        )
        sig = strat.generate_signal(market, pos)
        assert sig.action == Action.SELL
        assert sig.metadata.get("emergency_mode") is True


class TestPolyQualityFilter:
    """Polymarket 데이터 품질 필터"""

    def test_extreme_values_filtered(self):
        f = PolyQualityFilter()
        poly = {"btc_up": 0.01, "ndx_up": 0.55, "eth_up": 1.0}
        result = f.filter(poly)
        assert "btc_up" not in result   # < 0.02
        assert "ndx_up" in result
        assert "eth_up" not in result   # > 0.99

    def test_stale_data_filtered(self):
        f = PolyQualityFilter()
        now = datetime(2026, 2, 19, 18, 0)
        poly = {"btc_up": 0.55, "ndx_up": 0.60}
        timestamps = {
            "btc_up": now - timedelta(hours=6),  # stale
            "ndx_up": now - timedelta(hours=1),  # fresh
        }
        result = f.filter(poly, timestamps, now)
        assert "btc_up" not in result
        assert "ndx_up" in result

    def test_ndx_reliable_when_fresh(self):
        f = PolyQualityFilter()
        now = datetime(2026, 2, 19, 18, 0)
        ts = {"ndx_up": now - timedelta(hours=1)}
        assert f.ndx_is_reliable(ts, now) is True

    def test_ndx_unreliable_when_stale(self):
        f = PolyQualityFilter()
        now = datetime(2026, 2, 19, 18, 0)
        ts = {"ndx_up": now - timedelta(hours=6)}
        assert f.ndx_is_reliable(ts, now) is False

    def test_ndx_unreliable_when_missing(self):
        f = PolyQualityFilter()
        now = datetime(2026, 2, 19, 18, 0)
        ts = {"btc_up": now}  # ndx_up not present
        assert f.ndx_is_reliable(ts, now) is False


class TestSectorRotate:
    """Q-8: 섹터 로테이션 — 1Y low 기반 전환"""

    def test_params_has_rotation_sequence(self):
        assert "rotation_sequence" in SECTOR_ROTATE
        assert len(SECTOR_ROTATE["rotation_sequence"]) == 4

    def test_gold_step_is_cash(self):
        gold = SECTOR_ROTATE["rotation_sequence"][3]
        assert gold["name"] == "gold"
        assert gold["action"] == "cash"

    def test_activate_on_pct_above_1y_low(self):
        strat = SectorRotate()
        market = MarketData(
            changes={}, prices={"BITU": 114, "SOXL": 20.0},
            poly=None,
            time=datetime(2026, 2, 19),
            history={"BITU": {"low_1y": 100}},  # 14% above
        )
        assert strat.check_entry(market) is True

    def test_no_activate_below_threshold(self):
        strat = SectorRotate()
        market = MarketData(
            changes={}, prices={"BITU": 105, "SOXL": 20.0},
            poly=None,
            time=datetime(2026, 2, 19),
            history={"BITU": {"low_1y": 100}},  # 5% above — below 14%
        )
        assert strat.check_entry(market) is False

    def test_deactivate_triggers_sell(self):
        strat = SectorRotate()
        pos = Position(
            ticker="SOXL", avg_price=20.0, qty=5.0,
            entry_time=datetime(2026, 1, 1),
            strategy_name="sector_rotate",
        )
        market = MarketData(
            changes={}, prices={"BITU": 165, "SOXL": 25.0},
            poly=None,
            time=datetime(2026, 2, 19),
            history={"BITU": {"low_1y": 100}},  # 65% above → deactivate at 60%
        )
        sig = strat.generate_signal(market, pos)
        assert sig.action == Action.SELL

    def test_cycle_advances(self):
        strat = SectorRotate()
        assert strat._current_idx == 0
        strat._advance_step()
        assert strat._current_idx == 1
        strat._advance_step()
        assert strat._current_idx == 2
        strat._advance_step()
        assert strat._current_idx == 3
        strat._advance_step()
        assert strat._current_idx == 0  # wraps around


class TestAssetMode:
    """자산 모드 시스템"""

    def test_params_exist(self):
        assert "attack_strategies" in ASSET_MODE
        assert "defense_strategies" in ASSET_MODE
        assert "jab_soxl" in ASSET_MODE["attack_strategies"]
        assert "sector_rotate" in ASSET_MODE["defense_strategies"]

    def test_default_mode_is_attack(self):
        mgr = AssetModeManager()
        assert mgr.mode == AssetMode.ATTACK

    def test_emergency_overrides(self):
        mgr = AssetModeManager()
        mgr.activate_emergency()
        assert mgr.mode == AssetMode.EMERGENCY
        assert mgr.is_emergency_active()

    def test_cautious_reduces_leverage(self):
        mgr = AssetModeManager()
        mgr.activate_cautious(attack_leverage_pct=50)
        assert mgr.is_cautious_active()
        assert mgr.get_leverage_multiplier("jab_soxl") == 0.5
        assert mgr.get_leverage_multiplier("sector_rotate") == 1.0

    def test_deactivate_emergency(self):
        mgr = AssetModeManager()
        mgr.activate_emergency()
        mgr.deactivate_emergency()
        assert mgr.mode == AssetMode.ATTACK


# ============================================================
# 등록 확인
# ============================================================


def _ensure_strategies_imported():
    """@register 데코레이터 실행을 위해 전략 모듈을 강제 임포트."""
    from simulation.strategies.line_b_taejun.strategies import (  # noqa: F401
        bargain_buy, bearish_defense, bank_conditional,
        conditional_coin, conditional_conl, crash_buy,
        emergency_mode, jab_bitu, jab_etq, jab_soxl, jab_tsll,
        reit_risk, sector_rotate, short_macro, soxl_independent,
        sp500_entry, twin_pair, vix_gold,
    )


class TestRegistration:
    """신규 전략이 레지스트리에 등록되었는지 확인."""

    def test_emergency_mode_registered(self):
        _ensure_strategies_imported()
        from simulation.strategies.line_b_taejun.common.registry import list_strategies
        strategies = list_strategies()
        assert "emergency_mode" in strategies

    def test_all_expected_strategies(self):
        _ensure_strategies_imported()
        from simulation.strategies.line_b_taejun.common.registry import list_strategies
        strategies = list_strategies()
        expected = [
            "bargain_buy", "jab_bitu", "jab_tsll", "jab_etq", "jab_soxl",
            "vix_gold", "sp500_entry", "sector_rotate", "bank_conditional",
            "short_macro", "reit_risk", "emergency_mode",
        ]
        for s in expected:
            assert s in strategies, f"{s} not registered"


# ============================================================
# MT_VNQ3 변경 검증 테스트
# ============================================================


class TestMTVNQ3_ParamsCorrections:
    """MT_VNQ3 파라미터 정정 검증."""

    def test_jab_soxl_target(self):
        from simulation.strategies.line_b_taejun.common.params import JAB_SOXL
        assert JAB_SOXL["target_pct"] == 1.15, f"M20 +0.25%: expected 1.15, got {JAB_SOXL['target_pct']}"

    def test_jab_bitu_target(self):
        from simulation.strategies.line_b_taejun.common.params import JAB_BITU
        assert JAB_BITU["target_pct"] == 1.15, f"M20 +0.25%: expected 1.15, got {JAB_BITU['target_pct']}"

    def test_jab_tsll_target(self):
        from simulation.strategies.line_b_taejun.common.params import JAB_TSLL
        assert JAB_TSLL["target_pct"] == 1.25, f"M20 +0.25%: expected 1.25, got {JAB_TSLL['target_pct']}"

    def test_jab_tsll_gld_max(self):
        from simulation.strategies.line_b_taejun.common.params import JAB_TSLL
        assert JAB_TSLL["gld_max"] == 0.1, f"Q-5: expected 0.1, got {JAB_TSLL['gld_max']}"

    def test_jab_etq_target(self):
        from simulation.strategies.line_b_taejun.common.params import JAB_ETQ
        assert JAB_ETQ["target_pct"] == 1.05, f"CI-9: expected 1.05, got {JAB_ETQ['target_pct']}"

    def test_bank_conditional_target(self):
        from simulation.strategies.line_b_taejun.common.params import BANK_CONDITIONAL
        assert BANK_CONDITIONAL["target_pct"] == 1.05, f"CI-9: expected 1.05, got {BANK_CONDITIONAL['target_pct']}"

    def test_sp500_entry_target(self):
        from simulation.strategies.line_b_taejun.common.params import SP500_ENTRY
        assert SP500_ENTRY["target_pct"] == 1.75, f"M20 +0.25%: expected 1.75, got {SP500_ENTRY['target_pct']}"

    def test_vix_gold_target(self):
        from simulation.strategies.line_b_taejun.common.params import VIX_GOLD
        assert VIX_GOLD["target_pct"] == 10.25, f"M20 +0.25%: expected 10.25, got {VIX_GOLD['target_pct']}"

    def test_brku_drop(self):
        from simulation.strategies.line_b_taejun.common.params import BARGAIN_BUY
        assert BARGAIN_BUY["tickers"]["BRKU"]["drop_pct"] == -31, \
            f"Q-6: expected -31, got {BARGAIN_BUY['tickers']['BRKU']['drop_pct']}"

    def test_ethu_add_size_zero(self):
        from simulation.strategies.line_b_taejun.common.params import BARGAIN_BUY
        assert BARGAIN_BUY["tickers"]["ETHU"]["add_size"] == 0, \
            f"Q-7: expected 0, got {BARGAIN_BUY['tickers']['ETHU']['add_size']}"

    def test_reit_risk_vnq(self):
        from simulation.strategies.line_b_taejun.common.params import REIT_RISK
        assert REIT_RISK["conditions"]["reits"] == ["VNQ"], \
            f"VNQ primary: expected ['VNQ'], got {REIT_RISK['conditions']['reits']}"

    def test_reit_risk_kr_aux(self):
        from simulation.strategies.line_b_taejun.common.params import REIT_RISK
        assert "reits_kr_aux" in REIT_RISK["conditions"], "KR auxiliary reits key missing"

    def test_engine_config_no_chase(self):
        from simulation.strategies.line_b_taejun.common.params import ENGINE_CONFIG
        assert ENGINE_CONFIG.get("no_chase_buy") is True, "no_chase_buy should be True"

    def test_engine_config_t5_timeout(self):
        from simulation.strategies.line_b_taejun.common.params import ENGINE_CONFIG
        assert ENGINE_CONFIG.get("m5_t5_reserve_timeout_sec") == 10, "T5 timeout should be 10"


class TestMTVNQ3_M201ShortThreshold:
    """M201 SHORT 청산 기준 0.55 검증."""

    def test_short_close_at_055(self):
        from simulation.strategies.line_b_taejun.infra.m201_mode import M201ImmediateMode
        m201 = M201ImmediateMode()
        result = m201.check_short(0.55, 0.50)
        assert result is not None, "p=0.55 should trigger CLOSE_SHORT"

    def test_short_no_close_at_054(self):
        from simulation.strategies.line_b_taejun.infra.m201_mode import M201ImmediateMode
        m201 = M201ImmediateMode()
        result = m201.check_short(0.54, 0.50)
        assert result is None, "p=0.54 should NOT trigger CLOSE_SHORT"

    def test_short_close_at_060(self):
        from simulation.strategies.line_b_taejun.infra.m201_mode import M201ImmediateMode
        m201 = M201ImmediateMode()
        result = m201.check_short(0.60, 0.50)
        assert result is not None, "p=0.60 should still trigger CLOSE_SHORT"


class TestMTVNQ3_NewModules:
    """MT_VNQ3 신규 모듈 임포트 검증."""

    def test_limit_order_import(self):
        from simulation.strategies.line_b_taejun.infra.limit_order import (
            LimitOrder, OrderQueue, OrderStatus, FillEvent, CancelEvent,
        )
        assert OrderStatus.PENDING.value == "pending"

    def test_m200_import(self):
        from simulation.strategies.line_b_taejun.infra.m200_stop import M200KillSwitch
        ks = M200KillSwitch()
        assert ks._poly_btc_enabled is False  # P-5: OFF

    def test_m28_import(self):
        from simulation.strategies.line_b_taejun.infra.m28_poly_gate import M28PolyGate
        gate = M28PolyGate()
        assert gate.btc_gate(0.55) == "LONG"
        assert gate.btc_gate(0.45) == "SHORT"
        assert gate.btc_gate(0.50) == "NEUTRAL"

    def test_schd_import(self):
        from simulation.strategies.line_b_taejun.infra.schd_master import SCHDMaster
        schd = SCHDMaster()
        assert schd.is_sell_blocked("SCHD") is True
        assert schd.is_sell_blocked("SOXL") is False

    def test_m5_weight_manager(self):
        from simulation.strategies.line_b_taejun.infra.m5_weight_manager import M5WeightManager
        m5 = M5WeightManager()
        amounts = m5.sequential_allocations(10000)
        assert len(amounts) == 4
        assert abs(sum(amounts) - 10000) < 0.01

    def test_orchestrator_import(self):
        from simulation.strategies.line_b_taejun.infra.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.is_locked is False

    def test_profit_distributor_import(self):
        from simulation.strategies.line_b_taejun.infra.profit_distributor import ProfitDistributor
        pd = ProfitDistributor()
        assert pd.sequence == ["SOXL", "ROBN", "GLD", "CONL"]


class TestMTVNQ3_SCHDSellBlock:
    """SCHD 매도 차단 테스트."""

    def test_schd_blocked_in_all_modes(self):
        from simulation.strategies.line_b_taejun.infra.schd_master import SCHDMaster
        for mode in ("M200", "M201", "emergency", "risk", "normal"):
            assert SCHDMaster.should_exclude_from_sell("SCHD", mode) is True, \
                f"SCHD should be sell-blocked in {mode} mode"

    def test_non_schd_not_blocked(self):
        from simulation.strategies.line_b_taejun.infra.schd_master import SCHDMaster
        for ticker in ("SOXL", "CONL", "BITU"):
            assert SCHDMaster.should_exclude_from_sell(ticker, "M200") is False


class TestMTVNQ3_M28Gate:
    """M28 게이트 테스트."""

    def test_btc_primary_selection(self):
        from simulation.strategies.line_b_taejun.infra.m28_poly_gate import M28PolyGate
        gate = M28PolyGate()
        # Clear difference → A
        assert gate.select_btc_primary(1000, 500) == "A"
        # Clear difference → B
        assert gate.select_btc_primary(500, 1000) == "B"
        # Within 1% → keep previous (B from last call)
        assert gate.select_btc_primary(500, 505) == "B"

    def test_evaluate(self):
        from simulation.strategies.line_b_taejun.infra.m28_poly_gate import M28PolyGate
        gate = M28PolyGate()
        result = gate.evaluate({"btc_up": 0.63, "ndx_up": 0.45})
        assert result["btc_direction"] == "LONG"
        assert result["ndx_direction"] == "SHORT"

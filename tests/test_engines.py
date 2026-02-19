"""
PTJ 매매법 — 백테스트 엔진 검증 테스트
======================================
v1~v4 엔진의 import, 인스턴스화, 실행, 메트릭 추출을 검증한다.

테스트 기간: 2026-01-02 ~ 2026-01-30 (약 20 거래일, 빠른 실행)

Usage:
    pyenv shell ptj_stock_lab && pytest tests/test_engines.py -v
    pyenv shell ptj_stock_lab && pytest tests/test_engines.py -v -k v4   # v4만
    pyenv shell ptj_stock_lab && pytest tests/test_engines.py -v --tb=short
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tests.conftest import TEST_START, TEST_END


# ============================================================
# v1 BacktestEngine
# ============================================================
class TestV1Engine:
    """v1 BacktestEngine (독립, 5분봉)."""

    def test_import(self):
        from simulation.backtests.backtest import BacktestEngine
        assert BacktestEngine is not None

    def test_instantiate(self):
        from simulation.backtests.backtest import BacktestEngine
        engine = BacktestEngine(
            initial_capital=1000.0,
            start_date=TEST_START,
            end_date=TEST_END,
            use_fees=True,
        )
        assert engine.initial_capital == 1000.0
        assert engine.cash == 1000.0

    def test_has_methods(self):
        from simulation.backtests.backtest import BacktestEngine
        for method in ["run", "print_report", "summary", "save_trade_log"]:
            assert hasattr(BacktestEngine, method), f"Missing: {method}"

    def test_run(self, test_period):
        from simulation.backtests.backtest import BacktestEngine
        start, end = test_period
        engine = BacktestEngine(
            initial_capital=1000.0,
            start_date=start,
            end_date=end,
            use_fees=True,
        )
        result = engine.run(verbose=False)
        assert result is engine
        assert len(engine.equity_curve) > 0
        assert engine.total_trading_days > 0

    def test_metrics(self, test_period):
        from simulation.backtests.backtest import BacktestEngine
        start, end = test_period
        engine = BacktestEngine(
            initial_capital=1000.0,
            start_date=start,
            end_date=end,
            use_fees=True,
        )
        engine.run(verbose=False)
        metrics = engine.summary()

        assert "final_equity" in metrics
        assert "total_return_pct" in metrics
        assert "mdd_pct" in metrics
        assert "sharpe" in metrics
        assert "win_rate" in metrics
        assert metrics["final_equity"] > 0
        assert isinstance(metrics["total_return_pct"], (int, float))


# ============================================================
# v2 BacktestEngineV2
# ============================================================
class TestV2Engine:
    """v2 BacktestEngineV2 (BacktestBase 상속, 1분봉, USD)."""

    def test_import(self):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        # MRO에 BacktestBase가 포함되어야 함 (모듈 경로 차이로 issubclass 대신 이름 검사)
        mro_names = [c.__name__ for c in BacktestEngineV2.__mro__]
        assert "BacktestBase" in mro_names

    def test_instantiate(self):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        engine = BacktestEngineV2(
            start_date=TEST_START,
            end_date=TEST_END,
            use_fees=True,
        )
        assert engine.initial_capital == 15_000
        assert engine._version_label() == "v2 USD"

    def test_backward_compat_properties(self):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        engine = BacktestEngineV2(start_date=TEST_START, end_date=TEST_END)
        assert engine.initial_capital_usd == engine.initial_capital
        assert engine.cash_usd == engine.cash

    def test_abstract_methods_implemented(self):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        for method in [
            "_init_version_state", "_load_extra_data", "_warmup_extra",
            "_on_day_start", "_on_bar", "_on_day_end", "_version_label",
        ]:
            assert hasattr(BacktestEngineV2, method), f"Missing: {method}"

    def test_run(self, test_period):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        start, end = test_period
        engine = BacktestEngineV2(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        assert len(engine.equity_curve) > 0
        assert engine.total_trading_days > 0

    def test_metrics(self, test_period):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        start, end = test_period
        engine = BacktestEngineV2(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)

        final_equity = engine.equity_curve[-1][1]
        assert final_equity > 0
        assert engine.calc_mdd() >= 0
        assert isinstance(engine.calc_sharpe(), float)

    def test_trades_recorded(self, test_period):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        start, end = test_period
        engine = BacktestEngineV2(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        # v2는 활발한 매매 — 최소 1건 이상
        assert len(engine.trades) > 0

    def test_fees_accumulated(self, test_period):
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        start, end = test_period
        engine = BacktestEngineV2(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        total_fees = engine.total_buy_fees + engine.total_sell_fees
        assert total_fees > 0, "수수료가 0이면 use_fees가 동작하지 않음"


# ============================================================
# v3 BacktestEngineV3
# ============================================================
class TestV3Engine:
    """v3 BacktestEngineV3 (BacktestBase 상속, 1분봉, KRW)."""

    def test_import(self):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        mro_names = [c.__name__ for c in BacktestEngineV3.__mro__]
        assert "BacktestBase" in mro_names

    def test_instantiate(self):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        engine = BacktestEngineV3(
            start_date=TEST_START, end_date=TEST_END,
        )
        assert engine.initial_capital == 20_000_000
        assert engine._version_label() == "v3 선별 매매형"

    def test_fx_override(self):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        engine = BacktestEngineV3(start_date=TEST_START, end_date=TEST_END)
        # v3는 _get_fx_multiplier를 오버라이드
        assert "_get_fx_multiplier" in BacktestEngineV3.__dict__
        assert "_snapshot_equity" in BacktestEngineV3.__dict__

    def test_sideways_state_init(self):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        engine = BacktestEngineV3(start_date=TEST_START, end_date=TEST_END)
        assert hasattr(engine, "_sideways_active")
        assert engine._sideways_active is False
        assert engine.sideways_days == 0

    def test_run(self, test_period):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        start, end = test_period
        engine = BacktestEngineV3(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        assert len(engine.equity_curve) > 0
        assert engine.total_trading_days > 0

    def test_metrics(self, test_period):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        start, end = test_period
        engine = BacktestEngineV3(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)

        final_equity = engine.equity_curve[-1][1]
        assert final_equity > 0
        assert engine.calc_mdd() >= 0

    def test_v3_specific_stats(self, test_period):
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        start, end = test_period
        engine = BacktestEngineV3(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        # v3 통계 필드 존재 확인
        assert hasattr(engine, "sideways_days")
        assert hasattr(engine, "sideways_blocks")
        assert hasattr(engine, "entry_cutoff_blocks")
        assert hasattr(engine, "daily_limit_blocks")


# ============================================================
# v4 BacktestEngineV4
# ============================================================
class TestV4Engine:
    """v4 BacktestEngineV4 (BacktestBase 상속, 1분봉, USD, CB/스윙)."""

    def test_import(self):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        mro_names = [c.__name__ for c in BacktestEngineV4.__mro__]
        assert "BacktestBase" in mro_names

    def test_instantiate(self):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        engine = BacktestEngineV4(
            start_date=TEST_START, end_date=TEST_END,
        )
        assert engine.initial_capital == 15_000
        assert engine._version_label() == "v4 선별 매매형"

    def test_abstract_methods_implemented(self):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        for method in [
            "_init_version_state", "_load_extra_data", "_warmup_extra",
            "_on_day_start", "_on_bar", "_on_day_end", "_version_label",
        ]:
            assert hasattr(BacktestEngineV4, method), f"Missing: {method}"

    def test_v4_specific_methods(self):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        for method in ["_process_sells", "_process_buys", "_augment_prices"]:
            assert hasattr(BacktestEngineV4, method), f"Missing: {method}"

    def test_run(self, test_period):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        start, end = test_period
        engine = BacktestEngineV4(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        assert len(engine.equity_curve) > 0
        assert engine.total_trading_days > 0

    def test_metrics(self, test_period):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        start, end = test_period
        engine = BacktestEngineV4(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)

        final_equity = engine.equity_curve[-1][1]
        assert final_equity > 0
        assert engine.calc_mdd() >= 0
        assert isinstance(engine.calc_sharpe(), float)

    def test_v4_specific_stats(self, test_period):
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        start, end = test_period
        engine = BacktestEngineV4(
            start_date=start, end_date=end, use_fees=True,
        )
        engine.run(verbose=False)
        assert hasattr(engine, "sideways_days")


# ============================================================
# Pipeline 통합 테스트
# ============================================================
class TestPipeline:
    """pipeline.py 통합 진입점 테스트."""

    def test_import(self):
        from simulation.pipeline import run_backtest, get_metrics, export_for_inference
        assert callable(run_backtest)
        assert callable(get_metrics)
        assert callable(export_for_inference)

    @pytest.mark.parametrize("version", ["v2", "v3", "v4"])
    def test_run_backtest(self, version, test_period):
        from simulation.pipeline import run_backtest
        start, end = test_period
        engine = run_backtest(
            version,
            start_date=start,
            end_date=end,
            use_fees=True,
            verbose=False,
        )
        assert len(engine.equity_curve) > 0
        assert engine.total_trading_days > 0

    @pytest.mark.parametrize("version", ["v2", "v3", "v4"])
    def test_get_metrics(self, version, test_period):
        from simulation.pipeline import run_backtest, get_metrics
        start, end = test_period
        engine = run_backtest(
            version,
            start_date=start,
            end_date=end,
            use_fees=True,
            verbose=False,
        )
        metrics = get_metrics(engine)
        assert hasattr(metrics, "final_equity")
        assert hasattr(metrics, "total_return_pct")
        assert hasattr(metrics, "mdd")
        assert metrics.final_equity > 0

    @pytest.mark.parametrize("version", ["v2", "v3", "v4"])
    def test_export_for_inference(self, version):
        from simulation.pipeline import export_for_inference
        result = export_for_inference(version)
        assert result["version"] == version
        assert "params" in result
        assert isinstance(result["params"], dict)
        assert len(result["params"]) > 0


# ============================================================
# Params 테스트
# ============================================================
class TestParams:
    """strategies/params.py 파라미터 관리 테스트."""

    def test_frozen_dataclass(self):
        from simulation.strategies.params import BaseParams
        p = BaseParams()
        with pytest.raises(AttributeError):
            p.total_capital = 999  # frozen이므로 mutation 불가

    def test_inheritance_chain(self):
        from simulation.strategies.params import BaseParams, V3Params, V4Params, V5Params
        assert issubclass(V3Params, BaseParams)
        assert issubclass(V4Params, V3Params)
        assert issubclass(V5Params, V4Params)

    def test_to_dict_from_dict(self):
        from simulation.strategies.params import V4Params
        p = V4Params()
        d = p.to_dict()
        assert isinstance(d, dict)
        assert "total_capital" in d
        assert "fee_config" not in d  # fee_config는 제외

        p2 = V4Params.from_dict(d)
        assert p2.total_capital == p.total_capital
        assert p2.stop_loss_pct == p.stop_loss_pct

    def test_from_config(self):
        from simulation.strategies.params import (
            v2_params_from_config, v3_params_from_config,
            v4_params_from_config, v5_params_from_config,
        )
        p2 = v2_params_from_config()
        p3 = v3_params_from_config()
        p4 = v4_params_from_config()
        p5 = v5_params_from_config()

        assert p2.total_capital > 0
        assert p3.use_krw is True
        assert p4.use_krw is False
        assert p5.unix_defense_trigger_pct > 0

    def test_fee_config(self):
        from simulation.strategies.params import FeeConfig
        fc = FeeConfig()
        assert fc.buy_fee_pct > 0
        assert fc.sell_fee_pct > 0
        assert fc.round_trip_pct == fc.buy_fee_pct + fc.sell_fee_pct

        net, fee = fc.calc_buy_fee(10000)
        assert fee > 0
        assert net + fee == 10000

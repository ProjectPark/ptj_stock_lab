"""
Optimizer 리팩토링 검증 테스트
==============================
Phase 1: shared_params, Phase 2: report_sections, Phase 3: hooks
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================
# Phase 1: shared_params 검증
# ============================================================


class TestSharedParams:
    def test_keys_count(self):
        """공유 파라미터 11개 확인."""
        from optimizers.shared_params import get_shared_baseline_params

        params = get_shared_baseline_params()
        assert len(params) == 11

    def test_expected_keys(self):
        """11개 키 이름이 정확한지 확인."""
        from optimizers.shared_params import get_shared_baseline_params

        params = get_shared_baseline_params()
        expected = {
            "STOP_LOSS_PCT",
            "STOP_LOSS_BULLISH_PCT",
            "COIN_SELL_PROFIT_PCT",
            "COIN_SELL_BEARISH_PCT",
            "CONL_SELL_PROFIT_PCT",
            "CONL_SELL_AVG_PCT",
            "DCA_DROP_PCT",
            "MAX_HOLD_HOURS",
            "TAKE_PROFIT_PCT",
            "PAIR_GAP_SELL_THRESHOLD_V2",
            "PAIR_SELL_FIRST_PCT",
        }
        assert set(params.keys()) == expected

    def test_v3_includes_shared(self):
        """V3Optimizer.get_baseline_params()에 공유 키 포함 확인."""
        from optimizers.optimize_v3_optuna import V3Optimizer
        from optimizers.shared_params import get_shared_baseline_params

        opt = V3Optimizer()
        baseline = opt.get_baseline_params()
        shared = get_shared_baseline_params()
        for key in shared:
            assert key in baseline, f"v3 baseline에 공유 키 {key!r} 누락"
            assert baseline[key] == shared[key]

    def test_v4_includes_shared(self):
        """V4Optimizer.get_baseline_params()에 공유 키 포함 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer
        from optimizers.shared_params import get_shared_baseline_params

        opt = V4Optimizer()
        baseline = opt.get_baseline_params()
        shared = get_shared_baseline_params()
        for key in shared:
            assert key in baseline, f"v4 baseline에 공유 키 {key!r} 누락"
            assert baseline[key] == shared[key]

    def test_v5_includes_shared(self):
        """V5Optimizer.get_baseline_params()에 공유 키 포함 확인."""
        from optimizers.optimize_v5_optuna import V5Optimizer
        from optimizers.shared_params import get_shared_baseline_params

        opt = V5Optimizer()
        baseline = opt.get_baseline_params()
        shared = get_shared_baseline_params()
        for key in shared:
            assert key in baseline, f"v5 baseline에 공유 키 {key!r} 누락"
            assert baseline[key] == shared[key]


# ============================================================
# Phase 2: report_sections 검증
# ============================================================


def _make_mock_study(n_trials: int = 5) -> MagicMock:
    """mock Optuna study 생성."""
    import optuna

    trials = []
    for i in range(n_trials):
        t = MagicMock()
        t.number = i
        t.value = 10.0 + i * 2.0
        t.state = optuna.trial.TrialState.COMPLETE
        t.params = {"STOP_LOSS_PCT": -3.0 - i * 0.5, "TAKE_PROFIT_PCT": 3.0 + i}
        t.user_attrs = {
            "final_equity": 1_100_000 + i * 50_000,
            "mdd": 5.0 + i,
            "sharpe": 1.0 + i * 0.1,
            "total_fees": 10_000,
            "total_sells": 50 + i * 10,
            "total_buys": 55 + i * 10,
            "win_rate": 50.0 + i,
            "stop_loss_count": 5,
            "time_stop_count": 3,
            "sideways_days": 10,
            "sideways_blocks": 5,
            "entry_cutoff_blocks": 2,
            "daily_limit_blocks": 1,
            "cb_buy_blocks": 0,
        }
        trials.append(t)

    study = MagicMock()
    study.trials = trials
    study.best_trial = trials[-1]  # highest value
    return study


class TestReportSections:
    def test_section_execution_info(self):
        """실행 정보 섹션이 올바른 마크다운을 생성하는지 확인."""
        from optimizers.report_sections import section_execution_info

        study = _make_mock_study()
        lines = section_execution_info(study, elapsed=120.0, n_jobs=6, version="v5")
        text = "\n".join(lines)
        assert "## 1. 실행 정보" in text
        assert "병렬 Worker" in text
        assert "6" in text

    def test_section_execution_info_extra_rows(self):
        """extra_rows가 추가되는지 확인."""
        from optimizers.report_sections import section_execution_info

        study = _make_mock_study()
        lines = section_execution_info(
            study, elapsed=60.0, n_jobs=1, version="v4",
            extra_rows=[("Phase", "2"), ("모드", "balanced")]
        )
        text = "\n".join(lines)
        assert "Phase" in text
        assert "balanced" in text

    def test_section_baseline_vs_best(self):
        """Baseline vs Best 섹션 생성 확인."""
        from optimizers.report_sections import section_baseline_vs_best

        study = _make_mock_study()
        baseline = {
            "total_return_pct": 5.0,
            "mdd": 3.0,
            "sharpe": 0.8,
            "win_rate": 45.0,
            "total_sells": 40,
            "stop_loss_count": 3,
            "time_stop_count": 2,
            "sideways_days": 8,
            "total_fees": 8_000,
        }
        lines = section_baseline_vs_best(study, baseline)
        text = "\n".join(lines)
        assert "## 2. Baseline vs Best 비교" in text
        assert "수익률" in text

    def test_section_importance_empty_study(self):
        """trial 5개 미만 시 빈 리스트 반환."""
        from optimizers.report_sections import section_importance

        study = _make_mock_study(n_trials=3)
        lines = section_importance(study)
        assert lines == []

    def test_build_report_structure(self):
        """build_report()가 올바른 마크다운 헤더 포함."""
        from optimizers.report_sections import build_report

        section1 = ["## 1. 테스트", "", "내용"]
        section2 = ["## 2. 테스트2", "", "내용2"]
        md = build_report("v5", [section1, section2])
        assert md.startswith("# PTJ v5 Optuna 최적화 리포트")
        assert "## 1. 테스트" in md
        assert "## 2. 테스트2" in md

    def test_section_train_test(self):
        """Train/Test 섹션 생성 확인."""
        from optimizers.report_sections import section_train_test

        test_results = [
            {
                "rank": 1,
                "trial_number": 10,
                "train_score": 25.0,
                "test_result": {
                    "total_return_pct": 15.0,
                    "mdd": 8.0,
                    "sharpe": 1.2,
                    "win_rate": 55.0,
                },
            },
            {
                "rank": 2,
                "trial_number": 20,
                "train_score": 22.0,
                "test_result": None,
            },
        ]
        lines = section_train_test(
            test_results, train_end="2025-09-30",
            test_start="2025-10-01", test_end="2026-01-31"
        )
        text = "\n".join(lines)
        assert "Train/Test" in text
        assert "양호" in text
        assert "실패" in text


# ============================================================
# Phase 3: 훅 메서드 검증
# ============================================================


class TestHooks:
    def test_v5_uses_base_run_stage2(self):
        """V5Optimizer가 run_stage2()를 오버라이드하지 않는지 확인."""
        from optimizers.optimize_v5_optuna import V5Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V5Optimizer.run_stage2 is BaseOptimizer.run_stage2

    def test_v3_no_run_stage2_override(self):
        """V3Optimizer가 더 이상 run_stage2()를 오버라이드하지 않는지 확인."""
        from optimizers.optimize_v3_optuna import V3Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V3Optimizer.run_stage2 is BaseOptimizer.run_stage2

    def test_v4_no_run_stage2_override(self):
        """V4Optimizer가 더 이상 run_stage2()를 오버라이드하지 않는지 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V4Optimizer.run_stage2 is BaseOptimizer.run_stage2

    def test_v3_has_progress_callbacks(self):
        """V3Optimizer가 _get_progress_callbacks()를 오버라이드하는지 확인."""
        from optimizers.optimize_v3_optuna import V3Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V3Optimizer._get_progress_callbacks is not BaseOptimizer._get_progress_callbacks
        opt = V3Optimizer()
        cbs = opt._get_progress_callbacks(n_trials=20)
        assert len(cbs) == 1
        assert callable(cbs[0])

    def test_v4_has_pre_optimize_setup(self):
        """V4Optimizer가 _pre_optimize_setup()를 오버라이드하는지 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V4Optimizer._pre_optimize_setup is not BaseOptimizer._pre_optimize_setup

    def test_v4_has_post_optimize(self):
        """V4Optimizer가 _post_optimize()를 오버라이드하는지 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V4Optimizer._post_optimize is not BaseOptimizer._post_optimize

    def test_v4_has_report_sections(self):
        """V4Optimizer가 _get_report_sections()를 오버라이드하는지 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer
        from optimizers.optimizer_base import BaseOptimizer

        assert V4Optimizer._get_report_sections is not BaseOptimizer._get_report_sections

    def test_v4_no_make_v4_objective(self):
        """V4Optimizer에 _make_v4_objective가 더 이상 없는지 확인."""
        from optimizers.optimize_v4_optuna import V4Optimizer

        assert not hasattr(V4Optimizer, "_make_v4_objective")

    def test_base_make_objective_passes_kwargs(self):
        """base _make_objective가 kwargs를 run_single_trial에 전달하는지 확인."""
        from optimizers.optimizer_base import BaseOptimizer

        # _make_objective의 kwargs 전달 확인을 위한 간접 테스트
        # (실제 엔진 없이 서명만 확인)
        import inspect
        sig = inspect.signature(BaseOptimizer._make_objective)
        params = list(sig.parameters.keys())
        assert "kwargs" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

    def test_base_pre_optimize_returns_zero(self):
        """base _pre_optimize_setup이 0을 반환하는지 확인."""
        from optimizers.optimize_v5_optuna import V5Optimizer

        opt = V5Optimizer()
        mock_study = MagicMock()
        result = opt._pre_optimize_setup(mock_study, {"KEY": 1})
        assert result == 0
        mock_study.enqueue_trial.assert_called_once_with({"KEY": 1})

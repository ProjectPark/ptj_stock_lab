#!/usr/bin/env python3
"""
PTJ v4 — Optuna 기반 파라미터 최적화
=====================================
Stage 1: 현재 config 가중치로 baseline 실행 → 리포트 저장
Stage 2: Optuna TPE sampler로 최적 파라미터 탐색 → 리포트 저장

파라미터 그룹:
  Group A: v3 검증 공유 파라미터 (범위를 v3 최적값 중심으로 재설정)
  Group B: v4 횡보장 신규 파라미터 (ATR/Volume/EMA/RSI/Range)
  Group C: v4 고변동성/CB 파라미터 (VIX, High-Vol, BTC/GLD CB)
  Group D: v4 CONL 필터 / 분할매도 파라미터
  Group E: 진입 타이밍 / 자금 파라미터
  Group F: v4 스윙 매매 파라미터 (Phase 2 전용)

Usage:
    pyenv shell ptj_stock_lab && python optimizers/optimize_v4_optuna.py --stage 1
    pyenv shell ptj_stock_lab && python optimizers/optimize_v4_optuna.py --stage 2 [--n-trials 400] [--n-jobs 10]
    pyenv shell ptj_stock_lab && python optimizers/optimize_v4_optuna.py              # 1 → 2 연속 실행

    # Phase 1: wide search (v3 warm start 포함)
    pyenv shell ptj_stock_lab && python optimizers/optimize_v4_optuna.py --stage 2 \\
      --n-trials 400 --n-jobs 10 --objective balanced --gap-max 12.0 \\
      --study-name ptj_v4_phase1 --db sqlite:///data/optuna/optuna_v4_phase1.db

    # Phase 2: swing 파라미터 추가 + train/test 분리 (Phase 1 결과 warm start)
    pyenv shell ptj_stock_lab && python optimizers/optimize_v4_optuna.py --stage 2 \\
      --n-trials 250 --n-jobs 10 --objective balanced --gap-max 12.0 \\
      --phase 2 \\
      --train-end 2025-09-30 --test-start 2025-10-01 \\
      --phase1-db sqlite:///data/optuna/optuna_v4_phase1.db \\
      --phase1-study ptj_v4_phase1 \\
      --study-name ptj_v4_phase2 --db sqlite:///data/optuna/optuna_v4_phase2.db
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import optuna
from optuna.samplers import TPESampler

import config
from optimizers.optimizer_base import BaseOptimizer, TrialResult
from optimizers.shared_params import get_shared_baseline_params
from optimizers import report_sections as rs


# ============================================================
# V4Optimizer
# ============================================================


class V4Optimizer(BaseOptimizer):
    """v4 Optuna 최적화 — Group A~F 전체 파라미터."""

    version = "v4"

    def __init__(
        self,
        gap_max: float = 12.0,
        objective_mode: str = "balanced",
        phase: int = 1,
        train_end: str | None = None,
    ):
        super().__init__()
        self.gap_max = gap_max
        self.objective_mode = objective_mode
        self.phase = phase
        self.train_end = train_end
        # v4 리포트 경로 추가
        self._optuna_reports_dir = _ROOT / "docs" / "reports" / "optuna"

    def get_baseline_params(self) -> dict:
        """현재 config.py에 설정된 v4 파라미터를 반환한다."""
        return {
            # Group E: 진입 타이밍 / 자금
            "V4_PAIR_GAP_ENTRY_THRESHOLD": config.V4_PAIR_GAP_ENTRY_THRESHOLD,
            "V4_DCA_MAX_COUNT": config.V4_DCA_MAX_COUNT,
            "V4_MAX_PER_STOCK": config.V4_MAX_PER_STOCK,
            "V4_INITIAL_BUY": config.V4_INITIAL_BUY,
            "V4_DCA_BUY": config.V4_DCA_BUY,
            "V4_COIN_TRIGGER_PCT": config.V4_COIN_TRIGGER_PCT,
            "V4_CONL_TRIGGER_PCT": config.V4_CONL_TRIGGER_PCT,
            "V4_SPLIT_BUY_INTERVAL_MIN": config.V4_SPLIT_BUY_INTERVAL_MIN,
            "V4_ENTRY_CUTOFF_HOUR": config.V4_ENTRY_CUTOFF_HOUR,
            "V4_ENTRY_CUTOFF_MINUTE": config.V4_ENTRY_CUTOFF_MINUTE,
            # v4 횡보장 기본
            "V4_SIDEWAYS_MIN_SIGNALS": config.V4_SIDEWAYS_MIN_SIGNALS,
            "V4_SIDEWAYS_POLY_LOW": config.V4_SIDEWAYS_POLY_LOW,
            "V4_SIDEWAYS_POLY_HIGH": config.V4_SIDEWAYS_POLY_HIGH,
            "V4_SIDEWAYS_GLD_THRESHOLD": config.V4_SIDEWAYS_GLD_THRESHOLD,
            "V4_SIDEWAYS_GAP_FAIL_COUNT": config.V4_SIDEWAYS_GAP_FAIL_COUNT,
            "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": config.V4_SIDEWAYS_TRIGGER_FAIL_COUNT,
            "V4_SIDEWAYS_INDEX_THRESHOLD": config.V4_SIDEWAYS_INDEX_THRESHOLD,
            # Group B: v4 횡보장 신규
            "V4_SIDEWAYS_ATR_DECLINE_PCT": config.V4_SIDEWAYS_ATR_DECLINE_PCT,
            "V4_SIDEWAYS_VOLUME_DECLINE_PCT": config.V4_SIDEWAYS_VOLUME_DECLINE_PCT,
            "V4_SIDEWAYS_EMA_SLOPE_MAX": config.V4_SIDEWAYS_EMA_SLOPE_MAX,
            "V4_SIDEWAYS_RSI_LOW": config.V4_SIDEWAYS_RSI_LOW,
            "V4_SIDEWAYS_RSI_HIGH": config.V4_SIDEWAYS_RSI_HIGH,
            "V4_SIDEWAYS_RANGE_MAX_PCT": config.V4_SIDEWAYS_RANGE_MAX_PCT,
            # Group C: CB / 고변동성
            "V4_CB_GLD_SPIKE_PCT": config.V4_CB_GLD_SPIKE_PCT,
            "V4_CB_GLD_COOLDOWN_DAYS": config.V4_CB_GLD_COOLDOWN_DAYS,
            "V4_CB_BTC_CRASH_PCT": config.V4_CB_BTC_CRASH_PCT,
            "V4_CB_BTC_SURGE_PCT": config.V4_CB_BTC_SURGE_PCT,
            "V4_CB_VIX_SPIKE_PCT": config.V4_CB_VIX_SPIKE_PCT,
            "V4_CB_VIX_COOLDOWN_DAYS": config.V4_CB_VIX_COOLDOWN_DAYS,
            "V4_HIGH_VOL_MOVE_PCT": config.V4_HIGH_VOL_MOVE_PCT,
            "V4_HIGH_VOL_HIT_COUNT": config.V4_HIGH_VOL_HIT_COUNT,
            "V4_HIGH_VOL_STOP_LOSS_PCT": config.V4_HIGH_VOL_STOP_LOSS_PCT,
            # Group D: CONL 필터 / 분할매도
            "V4_CONL_ADX_MIN": config.V4_CONL_ADX_MIN,
            "V4_CONL_EMA_SLOPE_MIN_PCT": config.V4_CONL_EMA_SLOPE_MIN_PCT,
            "V4_PAIR_IMMEDIATE_SELL_PCT": config.V4_PAIR_IMMEDIATE_SELL_PCT,
            "V4_PAIR_FIXED_TP_PCT": config.V4_PAIR_FIXED_TP_PCT,
            # Group A: v2 공유 파라미터
            **get_shared_baseline_params(),
        }

    def create_engine(self, params: dict, **kwargs) -> Any:
        """v4 백테스트 엔진 인스턴스를 생성한다."""
        from backtest_v4 import BacktestEngineV4
        from datetime import date as _date

        engine_kwargs: dict = {}
        if kwargs.get("start_date"):
            sd = kwargs["start_date"]
            engine_kwargs["start_date"] = _date.fromisoformat(sd) if isinstance(sd, str) else sd
        if kwargs.get("end_date"):
            ed = kwargs["end_date"]
            engine_kwargs["end_date"] = _date.fromisoformat(ed) if isinstance(ed, str) else ed
        return BacktestEngineV4(**engine_kwargs)

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """v4 Optuna 탐색 공간을 정의한다 (Group A~F)."""
        # Group A: v3 검증 공유 파라미터
        params: dict = {
            "STOP_LOSS_PCT": trial.suggest_float("STOP_LOSS_PCT", -7.0, -2.5, step=0.25),
            "STOP_LOSS_BULLISH_PCT": trial.suggest_float("STOP_LOSS_BULLISH_PCT", -16.0, -8.0, step=0.5),
            "COIN_SELL_PROFIT_PCT": trial.suggest_float("COIN_SELL_PROFIT_PCT", 2.0, 7.0, step=0.5),
            "COIN_SELL_BEARISH_PCT": trial.suggest_float("COIN_SELL_BEARISH_PCT", 0.1, 0.8, step=0.1),
            "CONL_SELL_PROFIT_PCT": trial.suggest_float("CONL_SELL_PROFIT_PCT", 1.5, 6.0, step=0.5),
            "CONL_SELL_AVG_PCT": trial.suggest_float("CONL_SELL_AVG_PCT", 0.5, 2.0, step=0.25),
            "DCA_DROP_PCT": trial.suggest_float("DCA_DROP_PCT", -1.5, -0.3, step=0.05),
            "MAX_HOLD_HOURS": trial.suggest_int("MAX_HOLD_HOURS", 2, 8),
            "TAKE_PROFIT_PCT": trial.suggest_float("TAKE_PROFIT_PCT", 2.0, 6.0, step=0.5),
            "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float("PAIR_GAP_SELL_THRESHOLD_V2", 5.0, 12.0, step=0.2),
            "PAIR_SELL_FIRST_PCT": trial.suggest_float("PAIR_SELL_FIRST_PCT", 0.7, 1.0, step=0.05),
        }

        # Group E: 진입 타이밍 / 자금 파라미터
        params.update({
            "V4_PAIR_GAP_ENTRY_THRESHOLD": trial.suggest_float("V4_PAIR_GAP_ENTRY_THRESHOLD", 1.0, self.gap_max, step=0.2),
            "V4_DCA_MAX_COUNT": trial.suggest_int("V4_DCA_MAX_COUNT", 1, 12),
            "V4_MAX_PER_STOCK": trial.suggest_int("V4_MAX_PER_STOCK", 3_000, 10_000, step=500),
            "V4_INITIAL_BUY": trial.suggest_int("V4_INITIAL_BUY", 1_500, 3_500, step=250),
            "V4_DCA_BUY": trial.suggest_int("V4_DCA_BUY", 250, 1_500, step=125),
            "V4_COIN_TRIGGER_PCT": trial.suggest_float("V4_COIN_TRIGGER_PCT", 3.0, 8.0, step=0.5),
            "V4_CONL_TRIGGER_PCT": trial.suggest_float("V4_CONL_TRIGGER_PCT", 3.0, 9.0, step=0.5),
            "V4_SPLIT_BUY_INTERVAL_MIN": trial.suggest_int("V4_SPLIT_BUY_INTERVAL_MIN", 10, 30, step=5),
            "V4_ENTRY_CUTOFF_HOUR": trial.suggest_int("V4_ENTRY_CUTOFF_HOUR", 9, 14),
            "V4_ENTRY_CUTOFF_MINUTE": trial.suggest_categorical("V4_ENTRY_CUTOFF_MINUTE", [0, 30]),
        })

        # v4 횡보장 기본
        params.update({
            "V4_SIDEWAYS_MIN_SIGNALS": trial.suggest_int("V4_SIDEWAYS_MIN_SIGNALS", 2, 5),
            "V4_SIDEWAYS_POLY_LOW": trial.suggest_float("V4_SIDEWAYS_POLY_LOW", 0.30, 0.50, step=0.05),
            "V4_SIDEWAYS_POLY_HIGH": trial.suggest_float("V4_SIDEWAYS_POLY_HIGH", 0.50, 0.70, step=0.05),
            "V4_SIDEWAYS_GLD_THRESHOLD": trial.suggest_float("V4_SIDEWAYS_GLD_THRESHOLD", 0.1, 0.8, step=0.1),
            "V4_SIDEWAYS_INDEX_THRESHOLD": trial.suggest_float("V4_SIDEWAYS_INDEX_THRESHOLD", 0.2, 1.2, step=0.1),
            "V4_SIDEWAYS_GAP_FAIL_COUNT": trial.suggest_int("V4_SIDEWAYS_GAP_FAIL_COUNT", 1, 4),
            "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": trial.suggest_int("V4_SIDEWAYS_TRIGGER_FAIL_COUNT", 1, 4),
        })

        # Group B: v4 횡보장 신규 파라미터
        params.update({
            "V4_SIDEWAYS_ATR_DECLINE_PCT": trial.suggest_float("V4_SIDEWAYS_ATR_DECLINE_PCT", 10.0, 40.0, step=5.0),
            "V4_SIDEWAYS_VOLUME_DECLINE_PCT": trial.suggest_float("V4_SIDEWAYS_VOLUME_DECLINE_PCT", 15.0, 50.0, step=5.0),
            "V4_SIDEWAYS_EMA_SLOPE_MAX": trial.suggest_float("V4_SIDEWAYS_EMA_SLOPE_MAX", 0.05, 0.30, step=0.05),
            "V4_SIDEWAYS_RSI_LOW": trial.suggest_float("V4_SIDEWAYS_RSI_LOW", 35.0, 50.0, step=5.0),
            "V4_SIDEWAYS_RSI_HIGH": trial.suggest_float("V4_SIDEWAYS_RSI_HIGH", 50.0, 65.0, step=5.0),
            "V4_SIDEWAYS_RANGE_MAX_PCT": trial.suggest_float("V4_SIDEWAYS_RANGE_MAX_PCT", 0.5, 5.0, step=0.5),
        })

        # Group C: CB / 고변동성 파라미터
        params.update({
            "V4_CB_GLD_SPIKE_PCT": trial.suggest_float("V4_CB_GLD_SPIKE_PCT", 1.5, 5.0, step=0.5),
            "V4_CB_GLD_COOLDOWN_DAYS": trial.suggest_int("V4_CB_GLD_COOLDOWN_DAYS", 1, 7),
            "V4_CB_BTC_CRASH_PCT": trial.suggest_float("V4_CB_BTC_CRASH_PCT", -10.0, -3.0, step=0.5),
            "V4_CB_BTC_SURGE_PCT": trial.suggest_float("V4_CB_BTC_SURGE_PCT", 3.0, 10.0, step=0.5),
            "V4_CB_VIX_SPIKE_PCT": trial.suggest_float("V4_CB_VIX_SPIKE_PCT", 3.0, 10.0, step=1.0),
            "V4_CB_VIX_COOLDOWN_DAYS": trial.suggest_int("V4_CB_VIX_COOLDOWN_DAYS", 3, 14),
            "V4_HIGH_VOL_MOVE_PCT": trial.suggest_float("V4_HIGH_VOL_MOVE_PCT", 6.0, 15.0, step=1.0),
            "V4_HIGH_VOL_HIT_COUNT": trial.suggest_int("V4_HIGH_VOL_HIT_COUNT", 1, 4),
            "V4_HIGH_VOL_STOP_LOSS_PCT": trial.suggest_float("V4_HIGH_VOL_STOP_LOSS_PCT", -8.0, -2.0, step=0.5),
        })

        # Group D: CONL 필터 / 분할매도 파라미터
        params.update({
            "V4_CONL_ADX_MIN": trial.suggest_float("V4_CONL_ADX_MIN", 10.0, 30.0, step=2.0),
            "V4_CONL_EMA_SLOPE_MIN_PCT": trial.suggest_float("V4_CONL_EMA_SLOPE_MIN_PCT", -0.5, 0.5, step=0.1),
            "V4_PAIR_IMMEDIATE_SELL_PCT": trial.suggest_float("V4_PAIR_IMMEDIATE_SELL_PCT", 0.2, 0.6, step=0.1),
            "V4_PAIR_FIXED_TP_PCT": trial.suggest_float("V4_PAIR_FIXED_TP_PCT", 2.0, 8.0, step=0.5),
        })

        # Group F: 스윙 매매 파라미터 (Phase 2 전용)
        if self.phase >= 2:
            params.update({
                "V4_SWING_TRIGGER_PCT": trial.suggest_float("V4_SWING_TRIGGER_PCT", 10.0, 30.0, step=5.0),
                "V4_SWING_STAGE1_DRAWDOWN_PCT": trial.suggest_float("V4_SWING_STAGE1_DRAWDOWN_PCT", -20.0, -8.0, step=1.0),
                "V4_SWING_STAGE1_ATR_MULT": trial.suggest_float("V4_SWING_STAGE1_ATR_MULT", 0.5, 3.0, step=0.5),
            })

        return params

    def calc_score(self, result: TrialResult) -> float:
        """v4 목적함수 모드에 따라 스코어를 계산한다."""
        return _calc_score(result.to_dict(), self.objective_mode, self.phase)

    def get_trial_user_attrs(self, result: TrialResult) -> dict[str, Any]:
        """v4 추가 지표 (cb_buy_blocks, cb_sell_halt_bars)."""
        attrs = super().get_trial_user_attrs(result)
        attrs["cb_sell_halt_bars"] = result.cb_sell_halt_bars
        return attrs

    # ── Stage 1: Baseline (v4 전용 데이터 체크 포함) ──────────

    def run_stage1(self) -> tuple[TrialResult, dict]:
        """Stage 1: baseline 실행 → JSON 저장 → 마크다운 리포트."""
        print("\n" + "=" * 70)
        print("  [Stage 1] Baseline — 현재 config 가중치 시뮬레이션 (v4)")
        print("=" * 70)

        # 데이터 경로 사전 체크
        ohlcv_path = config.OHLCV_DIR / "backtest_1min_v2.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError(
                f"1분봉 데이터 없음: {ohlcv_path}\n"
                "  data/market/ohlcv/backtest_1min_v2.parquet 를 확인하세요."
            )
        daily_path = config.DAILY_DIR / "market_daily.parquet"
        if daily_path.exists():
            print(f"  [OK] market_daily 발견: {daily_path}")
        else:
            print(f"  [WARN] market_daily 없음: {daily_path} — 1분봉 집계로 대체")

        params = self.get_baseline_params()

        print("\n  실행 중...")
        t0 = time.time()
        result = self.run_single_trial(params)
        elapsed = time.time() - t0
        print(f"  완료 ({elapsed:.1f}초)")

        self._print_result_summary(result)
        # v4: CB 차단도 출력
        print(f"    CB 차단: {result.cb_buy_blocks}회")

        self.save_baseline_json(result, params)

        self._docs_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_baseline_report(result, params)
        self._baseline_report.write_text(md, encoding="utf-8")
        print(f"  Baseline 리포트: {self._baseline_report}")

        return result, params

    # ── 훅 오버라이드 ─────────────────────────────────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        """v4: Phase 1/2 warm start + baseline enqueue + warm-up 순차 실행."""
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            return already_done

        phase1_db = kwargs.get("phase1_db")
        phase1_study = kwargs.get("phase1_study", "ptj_v4_phase1")

        # Phase 2: Phase 1 best를 warm start로, Phase 1: v3 warm start
        if self.phase >= 2 and phase1_db:
            warm_start = _get_phase1_best_params(phase1_db, phase1_study)
            if not warm_start:
                warm_start = _get_v3_warm_start()
                print("  [FALLBACK] v3 warm start 사용")
            else:
                import config as _config
                warm_start.setdefault("V4_SWING_TRIGGER_PCT", _config.V4_SWING_TRIGGER_PCT)
                warm_start.setdefault("V4_SWING_STAGE1_DRAWDOWN_PCT", _config.V4_SWING_STAGE1_DRAWDOWN_PCT)
                warm_start.setdefault("V4_SWING_STAGE1_ATR_MULT", _config.V4_SWING_STAGE1_ATR_MULT)
        else:
            warm_start = _get_v3_warm_start()

        study.enqueue_trial(warm_start)
        label = "Phase 1 best" if (self.phase >= 2 and phase1_db) else "v3"
        print(f"  {label} warm start enqueue 완료 (Trial #0)")

        # baseline도 Phase 2 swing 파라미터 보충
        if self.phase >= 2:
            import config as _config
            bp2 = dict(baseline_params)
            bp2.setdefault("V4_SWING_TRIGGER_PCT", _config.V4_SWING_TRIGGER_PCT)
            bp2.setdefault("V4_SWING_STAGE1_DRAWDOWN_PCT", _config.V4_SWING_STAGE1_DRAWDOWN_PCT)
            bp2.setdefault("V4_SWING_STAGE1_ATR_MULT", _config.V4_SWING_STAGE1_ATR_MULT)
            study.enqueue_trial(bp2)
        else:
            study.enqueue_trial(baseline_params)
        print("  baseline enqueue 완료 (Trial #1)")

        # warm-up 순차 실행
        warmup_needed = 2
        print(f"  warm-up {warmup_needed}개 trial 순차 실행 중...")
        obj_kwargs = {}
        if self.train_end:
            obj_kwargs["end_date"] = self.train_end
        obj_single = self._make_objective(**obj_kwargs)
        study.optimize(obj_single, n_trials=warmup_needed, show_progress_bar=False)
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  warm-up 완료 (총 완료: {already_done}회)")
        return already_done

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        """v4: test split 평가 + Phase별 커스텀 리포트 경로."""
        test_start = kwargs.get("test_start")
        test_end = kwargs.get("test_end")

        # Test 기간 평가
        test_results = None
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if test_start and completed:
            print(f"\n  Test 기간 평가 중 ({test_start} ~ {test_end or '기본값'})...")
            test_results = _eval_test_split(
                self, study, top_n=5, test_start=test_start, test_end=test_end or None
            )
            print(f"\n  Train/Test 비교 (Top 5):")
            print(f"  {'#':>4s}  {'Train':>8s}  {'TestRet':>8s}  {'TestMDD':>8s}  {'TestSharpe':>10s}")
            for item in test_results:
                tr = item["test_result"]
                if tr:
                    print(
                        f"  {item['trial_number']:4d}  {item['train_score']:+7.2f}"
                        f"  {tr['total_return_pct']:+7.2f}%"
                        f"  -{tr['mdd']:6.2f}%"
                        f"  {tr['sharpe']:>10.4f}"
                    )

        # 마크다운 리포트
        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs,
            test_results=test_results, test_start=test_start, test_end=test_end,
        )
        if self.phase >= 2:
            self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = self._optuna_reports_dir / f"v4_phase{self.phase}_optuna_report.md"
        else:
            self._docs_dir.mkdir(parents=True, exist_ok=True)
            report_path = self._optuna_report
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Optuna 리포트: {report_path}")

    def _get_report_sections(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        """v4: 실행 정보에 Phase/목적함수/기간 추가, CB 차단 비교행 추가."""
        opt_period = "전체 기간"
        if self.train_end:
            opt_period = f"Train 기간 (~ {self.train_end})"

        extra_exec_rows = [
            ("Phase", str(self.phase)),
            ("목적함수 모드", self.objective_mode),
            ("최적화 기간", opt_period),
        ]
        if kwargs.get("test_start"):
            extra_exec_rows.append(
                ("Test 기간", f"{kwargs['test_start']} ~ {kwargs.get('test_end') or '엔진 기본값'}")
            )

        sections = [
            rs.section_execution_info(study, elapsed, n_jobs, self.version,
                                      extra_rows=extra_exec_rows),
            rs.section_baseline_vs_best(study, baseline,
                                        extra_rows=[("CB 차단", "cb_buy_blocks", "cb_buy_blocks")]),
            rs.section_best_params(study, baseline_params),
            rs.section_top5_table(study,
                                  extra_columns=[("CB차단", "cb_buy_blocks")]),
            rs.section_importance(study),
            rs.section_top5_detail(study),
            rs.section_config_code(study),
        ]
        # Train/Test 비교 섹션
        test_results = kwargs.get("test_results")
        if test_results:
            sections.append(rs.section_train_test(
                test_results,
                train_end=self.train_end,
                test_start=kwargs.get("test_start"),
                test_end=kwargs.get("test_end"),
            ))
        return sections


# ============================================================
# 스코어 계산 (모듈 레벨 — 워커에서도 호출 가능)
# ============================================================


def _calc_score(result: dict, objective: str, phase: int = 1) -> float:
    """목적함수 모드에 따라 최적화 스코어를 계산한다."""
    ret = result["total_return_pct"]

    if objective == "return":
        return ret

    # balanced: 수익률 - MDD 페널티 - 승률 페널티 - 과소거래 페널티
    mdd = result["mdd"]
    win_rate = result["win_rate"]
    total_sells = result["total_sells"]

    score = ret
    if phase >= 2:
        score -= 2.0 * max(0.0, mdd - 7.5)
        if mdd > 12.0:
            score -= 5.0
    else:
        score -= 1.5 * max(0.0, mdd - 8.0)
        if mdd > 15.0:
            score -= 5.0
    score -= 0.8 * max(0.0, 40.0 - win_rate)
    score -= 0.05 * max(0.0, 80 - total_sells)
    return score


# ============================================================
# v3 warm start / Phase 1 best 로드
# ============================================================


def _get_v3_warm_start() -> dict:
    """v3 Best Trial #301 파라미터를 v4 이름으로 변환하여 반환한다."""
    return {
        "V4_PAIR_GAP_ENTRY_THRESHOLD": 7.6,
        "V4_DCA_MAX_COUNT": 9,
        "V4_COIN_TRIGGER_PCT": 5.5,
        "V4_CONL_TRIGGER_PCT": 6.0,
        "V4_SPLIT_BUY_INTERVAL_MIN": 25,
        "V4_SIDEWAYS_MIN_SIGNALS": 2,
        "V4_SIDEWAYS_POLY_LOW": 0.35,
        "V4_SIDEWAYS_POLY_HIGH": 0.50,
        "V4_SIDEWAYS_GLD_THRESHOLD": 0.30,
        "V4_SIDEWAYS_INDEX_THRESHOLD": 0.90,
        "STOP_LOSS_PCT": -5.25,
        "STOP_LOSS_BULLISH_PCT": -14.0,
        "COIN_SELL_PROFIT_PCT": 5.0,
        "COIN_SELL_BEARISH_PCT": config.COIN_SELL_BEARISH_PCT,
        "CONL_SELL_PROFIT_PCT": 3.5,
        "CONL_SELL_AVG_PCT": config.CONL_SELL_AVG_PCT,
        "DCA_DROP_PCT": -0.95,
        "MAX_HOLD_HOURS": 4,
        "TAKE_PROFIT_PCT": 4.0,
        "PAIR_GAP_SELL_THRESHOLD_V2": 8.8,
        "PAIR_SELL_FIRST_PCT": 0.95,
        "V4_INITIAL_BUY": config.V4_INITIAL_BUY,
        "V4_DCA_BUY": config.V4_DCA_BUY,
        "V4_MAX_PER_STOCK": config.V4_MAX_PER_STOCK,
        "V4_ENTRY_CUTOFF_HOUR": 10,
        "V4_ENTRY_CUTOFF_MINUTE": 30,
        "V4_SIDEWAYS_GAP_FAIL_COUNT": config.V4_SIDEWAYS_GAP_FAIL_COUNT,
        "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": config.V4_SIDEWAYS_TRIGGER_FAIL_COUNT,
        "V4_SIDEWAYS_ATR_DECLINE_PCT": config.V4_SIDEWAYS_ATR_DECLINE_PCT,
        "V4_SIDEWAYS_VOLUME_DECLINE_PCT": config.V4_SIDEWAYS_VOLUME_DECLINE_PCT,
        "V4_SIDEWAYS_EMA_SLOPE_MAX": config.V4_SIDEWAYS_EMA_SLOPE_MAX,
        "V4_SIDEWAYS_RSI_LOW": config.V4_SIDEWAYS_RSI_LOW,
        "V4_SIDEWAYS_RSI_HIGH": config.V4_SIDEWAYS_RSI_HIGH,
        "V4_SIDEWAYS_RANGE_MAX_PCT": config.V4_SIDEWAYS_RANGE_MAX_PCT,
        "V4_CB_GLD_SPIKE_PCT": config.V4_CB_GLD_SPIKE_PCT,
        "V4_CB_GLD_COOLDOWN_DAYS": config.V4_CB_GLD_COOLDOWN_DAYS,
        "V4_CB_BTC_CRASH_PCT": config.V4_CB_BTC_CRASH_PCT,
        "V4_CB_BTC_SURGE_PCT": config.V4_CB_BTC_SURGE_PCT,
        "V4_CB_VIX_SPIKE_PCT": config.V4_CB_VIX_SPIKE_PCT,
        "V4_CB_VIX_COOLDOWN_DAYS": config.V4_CB_VIX_COOLDOWN_DAYS,
        "V4_HIGH_VOL_MOVE_PCT": config.V4_HIGH_VOL_MOVE_PCT,
        "V4_HIGH_VOL_HIT_COUNT": config.V4_HIGH_VOL_HIT_COUNT,
        "V4_HIGH_VOL_STOP_LOSS_PCT": config.V4_HIGH_VOL_STOP_LOSS_PCT,
        "V4_CONL_ADX_MIN": config.V4_CONL_ADX_MIN,
        "V4_CONL_EMA_SLOPE_MIN_PCT": config.V4_CONL_EMA_SLOPE_MIN_PCT,
        "V4_PAIR_IMMEDIATE_SELL_PCT": config.V4_PAIR_IMMEDIATE_SELL_PCT,
        "V4_PAIR_FIXED_TP_PCT": config.V4_PAIR_FIXED_TP_PCT,
    }


def _get_phase1_best_params(
    phase1_db: str,
    phase1_study_name: str = "ptj_v4_phase1",
) -> dict:
    """Phase 1 최적 Trial 파라미터를 로드한다."""
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    try:
        study = _optuna.load_study(study_name=phase1_study_name, storage=phase1_db)
        best = study.best_trial
        print(f"  Phase 1 Best Trial #{best.number} 로드 (score={best.value:+.2f})")
        return dict(best.params)
    except Exception as exc:
        print(f"  [WARN] Phase 1 best 로드 실패: {exc}")
        return {}


# ============================================================
# 병렬 워커 / Test 평가
# ============================================================


def _study_worker(
    study_name: str,
    storage_url: str,
    n_trials: int,
    gap_max: float,
    objective: str,
    phase: int = 1,
    train_end: str | None = None,
) -> None:
    """병렬 워커 프로세스에서 독립적으로 study.optimize()를 실행한다."""
    import optuna as _optuna
    from optuna.samplers import TPESampler as _TPESampler

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    sampler = _TPESampler(n_startup_trials=20)
    study = _optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )
    opt = V4Optimizer(gap_max=gap_max, objective_mode=objective, phase=phase, train_end=train_end)
    obj_kwargs = {}
    if train_end:
        obj_kwargs["end_date"] = train_end
    obj = opt._make_objective(**obj_kwargs)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)


def _eval_test_split(
    optimizer: V4Optimizer,
    study: optuna.Study,
    top_n: int,
    test_start: str,
    test_end: str | None = None,
) -> list[dict]:
    """Top-N trials을 test 기간으로 평가하여 결과를 반환한다."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

    results = []
    for rank, t in enumerate(top_trials, 1):
        print(f"  Test 평가 [{rank}/{top_n}] Trial #{t.number} (train score={t.value:+.2f})...")
        try:
            res = optimizer.run_single_trial(
                t.params,
                start_date=test_start or None,
                end_date=test_end or None,
            )
            results.append({
                "rank": rank,
                "trial_number": t.number,
                "train_score": t.value,
                "test_result": res.to_dict(),
            })
        except Exception as exc:
            print(f"    [WARN] Trial #{t.number} test 평가 실패: {exc}")
            results.append({
                "rank": rank,
                "trial_number": t.number,
                "train_score": t.value,
                "test_result": None,
            })
    return results


# ── 워커 함수 (mp.Pool 호환) ──────────────────────────────────


def _worker(params: dict) -> dict:
    """mp.Pool에서 호출되는 최상위 함수."""
    opt = V4Optimizer()
    result = opt.run_single_trial(params)
    return result.to_dict()


# ── 메인 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PTJ v4 Optuna 최적화")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=400, help="Optuna trial 수 (기본: 400)")
    parser.add_argument("--n-jobs", type=int, default=10, help="병렬 프로세스 수 (기본: 10)")
    parser.add_argument("--timeout", type=int, default=None, help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v4_opt", help="study 이름")
    parser.add_argument("--db", type=str, default=None, help="Optuna DB URL (기본: in-memory)")
    parser.add_argument("--gap-max", type=float, default=12.0,
                        help="V4_PAIR_GAP_ENTRY_THRESHOLD 탐색 상한 (%%, 기본: 12.0)")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"],
                        default="balanced",
                        help="목적함수 모드: return / balanced (기본)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="최적화 Phase (1: Group A~E, 2: A~F + swing, 기본: 1)")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Train 기간 마감일 (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, default=None,
                        help="Test 기간 시작일 (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, default=None,
                        help="Test 기간 마감일 (YYYY-MM-DD)")
    parser.add_argument("--phase1-db", type=str, default=None,
                        help="Phase 2 warm start용 Phase 1 DB URL")
    parser.add_argument("--phase1-study", type=str, default="ptj_v4_phase1",
                        help="Phase 1 study 이름 (기본: ptj_v4_phase1)")
    args = parser.parse_args()

    opt = V4Optimizer(
        gap_max=args.gap_max,
        objective_mode=args.objective,
        phase=args.phase,
        train_end=args.train_end,
    )

    print("=" * 70)
    print(f"  PTJ v4 — Optuna 파라미터 최적화 (Group A~{'F' if args.phase >= 2 else 'E'})")
    print(f"  Phase={args.phase} | objective={args.objective} | gap-max={args.gap_max}% | n-jobs={args.n_jobs}")
    if args.train_end:
        print(f"  Train: ~ {args.train_end}  /  Test: {args.test_start or '미설정'} ~ {args.test_end or '기본값'}")
    print("=" * 70)

    if args.stage == 1:
        opt.run_stage1()

    elif args.stage == 2:
        opt.run_stage2(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            study_name=args.study_name,
            db=args.db,
            end_date=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            phase1_db=args.phase1_db,
            phase1_study=args.phase1_study,
        )

    else:
        result, params = opt.run_stage1()
        print()
        opt.run_stage2(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            study_name=args.study_name,
            db=args.db,
            baseline=result.to_dict(),
            baseline_params=params,
            end_date=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            phase1_db=args.phase1_db,
            phase1_study=args.phase1_study,
        )

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

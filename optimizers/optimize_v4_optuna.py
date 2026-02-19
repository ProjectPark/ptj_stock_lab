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
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import optuna
from optuna.samplers import TPESampler

import config
from optimizers.optimizer_base import BaseOptimizer, TrialResult, extract_metrics


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
            "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
            "STOP_LOSS_BULLISH_PCT": config.STOP_LOSS_BULLISH_PCT,
            "COIN_SELL_PROFIT_PCT": config.COIN_SELL_PROFIT_PCT,
            "COIN_SELL_BEARISH_PCT": config.COIN_SELL_BEARISH_PCT,
            "CONL_SELL_PROFIT_PCT": config.CONL_SELL_PROFIT_PCT,
            "CONL_SELL_AVG_PCT": config.CONL_SELL_AVG_PCT,
            "DCA_DROP_PCT": config.DCA_DROP_PCT,
            "MAX_HOLD_HOURS": config.MAX_HOLD_HOURS,
            "TAKE_PROFIT_PCT": config.TAKE_PROFIT_PCT,
            "PAIR_GAP_SELL_THRESHOLD_V2": config.PAIR_GAP_SELL_THRESHOLD_V2,
            "PAIR_SELL_FIRST_PCT": config.PAIR_SELL_FIRST_PCT,
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

    def run_single_trial(self, params: dict, **kwargs) -> TrialResult:
        """v4 백테스트 1회를 실행하고 지표를 반환한다.

        train_end가 설정된 경우 end_date로 전달한다.
        """
        import config as _config

        originals = {}
        for key, value in params.items():
            originals[key] = getattr(_config, key)
            setattr(_config, key, value)

        try:
            engine = self.create_engine(params, **kwargs)
            engine.run(verbose=False)
            return extract_metrics(engine)
        finally:
            for key, value in originals.items():
                setattr(_config, key, value)

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

    # ── Stage 2: Optuna (v4 전용 병렬/warm start) ─────────────

    def run_stage2(
        self,
        n_trials: int = 400,
        n_jobs: int = 10,
        timeout: int | None = None,
        study_name: str | None = None,
        db: str | None = None,
        baseline: dict | None = None,
        baseline_params: dict | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        phase1_db: str | None = None,
        phase1_study: str = "ptj_v4_phase1",
        **kwargs,
    ) -> None:
        """Stage 2: v4 Optuna 최적화."""
        if study_name is None:
            study_name = "ptj_v4_opt"

        # baseline 로드
        if baseline is None or baseline_params is None:
            baseline, baseline_params = self.load_baseline_json()
            print(f"  Baseline 로드: {self._baseline_json}")
            print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")

        print(f"\n{'=' * 70}")
        print(f"  [Stage 2] Optuna 최적화 ({n_trials} trials, {n_jobs} workers)")
        print(f"  Phase: {self.phase} | 목적함수: {self.objective_mode} | GAP max: {self.gap_max}%")
        if self.train_end:
            print(f"  Train 기간: ~ {self.train_end}  /  Test 기간: {test_start or 'N/A'} ~ {test_end or '기본값'}")
        print(f"{'=' * 70}")

        sampler = TPESampler(seed=42, n_startup_trials=min(20, n_trials))
        storage = db if db else None

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        # warm start / baseline enqueue
        already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        warmup_needed = 0

        if already_done == 0:
            # Phase 2: Phase 1 best를 warm start로, Phase 1: v3 warm start
            if self.phase >= 2 and phase1_db:
                warm_start = _get_phase1_best_params(phase1_db, phase1_study)
                if not warm_start:
                    warm_start = _get_v3_warm_start()
                    print(f"  [FALLBACK] v3 warm start 사용")
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
            print(f"  baseline enqueue 완료 (Trial #1)")
            warmup_needed = 2

        if warmup_needed > 0:
            print(f"  warm-up {warmup_needed}개 trial 순차 실행 중...")
            obj_single = self._make_v4_objective()
            study.optimize(obj_single, n_trials=warmup_needed, show_progress_bar=False)
            already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"  warm-up 완료 (총 완료: {already_done}회)")

        t0 = time.time()

        remaining = n_trials - already_done
        if remaining <= 0:
            print(f"  목표 trial {n_trials}회 이미 완료됨 (완료: {already_done}회)")
        elif n_jobs > 1:
            if not db:
                raise ValueError(
                    "n_jobs > 1 병렬 실행에는 --db 옵션이 필요합니다 (SQLite 공유 저장소).\n"
                    "예: --db sqlite:///data/optuna/optuna_v4_phase1.db"
                )
            n_trials_per_worker = max(1, remaining // n_jobs)
            remainder_extra = remaining - n_trials_per_worker * n_jobs
            print(f"  남은 trials: {remaining}  워커당: {n_trials_per_worker} x {n_jobs}")

            ctx = mp.get_context("spawn")
            worker_trials = [n_trials_per_worker + remainder_extra] + [n_trials_per_worker] * (n_jobs - 1)
            worker_args = [
                (study_name, storage, wt, self.gap_max, self.objective_mode, self.phase, self.train_end)
                for wt in worker_trials
            ]
            with ctx.Pool(processes=n_jobs) as pool:
                pool.starmap(_study_worker, worker_args)

            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            obj = self._make_v4_objective()
            study.optimize(
                obj,
                n_trials=remaining,
                timeout=timeout,
                show_progress_bar=True,
            )

        elapsed = time.time() - t0

        # 콘솔 요약
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best = study.best_trial
            print(f"\n  실행 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
            print(f"  Trial당 평균: {elapsed / max(len(study.trials), 1):.1f}초")
            print(f"\n  BEST Trial #{best.number}  (score={best.value:+.2f})")
            print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
            print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
            print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")

            top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
            print(f"\n  Top 5:")
            print(f"  {'#':>4s}  {'Score':>8s}  {'MDD':>8s}  {'Sharpe':>8s}  {'승률':>6s}")
            for t in top5:
                print(
                    f"  {t.number:4d}  {t.value:+7.2f}"
                    f"  -{t.user_attrs.get('mdd', 0):6.2f}%"
                    f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
                    f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
                )
        else:
            print("\n  완료된 trial 없음")
            return

        # Test 기간 평가
        test_results = None
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
        md = self._generate_v4_optuna_report(
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

    def _make_v4_objective(self):
        """v4 전용 objective (train_end 지원)."""
        optimizer = self

        class _V4Objective:
            def __call__(self_obj, trial: optuna.Trial) -> float:
                params = optimizer.define_search_space(trial)
                kwargs = {}
                if optimizer.train_end:
                    kwargs["end_date"] = optimizer.train_end
                result = optimizer.run_single_trial(params, **kwargs)
                for attr_key, value in optimizer.get_trial_user_attrs(result).items():
                    trial.set_user_attr(attr_key, value)
                return optimizer.calc_score(result)

        return _V4Objective()

    def _generate_v4_optuna_report(
        self,
        study: optuna.Study,
        baseline: dict,
        baseline_params: dict,
        elapsed: float,
        n_jobs: int,
        test_results: list[dict] | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
    ) -> str:
        """v4 Optuna 마크다운 리포트를 생성한다."""
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        best = study.best_trial
        bl = baseline
        best_score = best.value
        bl_ret = bl["total_return_pct"]

        opt_period = "전체 기간"
        if self.train_end:
            opt_period = f"Train 기간 (~ {self.train_end})"

        lines = [
            "# PTJ v4 Optuna 최적화 리포트",
            "",
            f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 1. 실행 정보",
            "",
            "| 항목 | 값 |",
            "|---|---|",
            f"| Phase | {self.phase} |",
            f"| 총 Trial | {len(study.trials)} (완료: {len(completed)}, 실패: {len(failed)}) |",
            f"| 병렬 Worker | {n_jobs} |",
            f"| 실행 시간 | {elapsed:.1f}초 ({elapsed / 60:.1f}분) |",
            f"| Trial당 평균 | {elapsed / max(len(study.trials), 1):.1f}초 |",
            f"| Sampler | TPE (seed=42) |",
            f"| 목적함수 모드 | {self.objective_mode} |",
            f"| 최적화 기간 | {opt_period} |",
        ]
        if test_start:
            lines.append(f"| Test 기간 | {test_start} ~ {test_end or '엔진 기본값'} |")

        lines += [
            "",
            "## 2. Baseline vs Best 비교",
            "",
            "| 지표 | Baseline | Best | 차이 |",
            "|---|---|---|---|",
            f"| **수익률** | {bl_ret:+.2f}% | **{best_score:+.2f}** (score) | {best_score - bl_ret:+.2f} |",
            f"| MDD | -{bl['mdd']:.2f}% | -{best.user_attrs.get('mdd', 0):.2f}% | {best.user_attrs.get('mdd', 0) - bl['mdd']:+.2f}% |",
            f"| Sharpe | {bl['sharpe']:.4f} | {best.user_attrs.get('sharpe', 0):.4f} | {best.user_attrs.get('sharpe', 0) - bl['sharpe']:+.4f} |",
            f"| 승률 | {bl['win_rate']:.1f}% | {best.user_attrs.get('win_rate', 0):.1f}% | {best.user_attrs.get('win_rate', 0) - bl['win_rate']:+.1f}% |",
            f"| 매도 횟수 | {bl['total_sells']} | {best.user_attrs.get('total_sells', 0)} | {best.user_attrs.get('total_sells', 0) - bl['total_sells']:+d} |",
            f"| 손절 횟수 | {bl['stop_loss_count']} | {best.user_attrs.get('stop_loss_count', 0)} | {best.user_attrs.get('stop_loss_count', 0) - bl['stop_loss_count']:+d} |",
            f"| 시간손절 | {bl['time_stop_count']} | {best.user_attrs.get('time_stop_count', 0)} | {best.user_attrs.get('time_stop_count', 0) - bl['time_stop_count']:+d} |",
            f"| 횡보장 일수 | {bl['sideways_days']} | {best.user_attrs.get('sideways_days', 0)} | {best.user_attrs.get('sideways_days', 0) - bl['sideways_days']:+d} |",
            f"| CB 차단 | {bl.get('cb_buy_blocks', 0)} | {best.user_attrs.get('cb_buy_blocks', 0)} | {best.user_attrs.get('cb_buy_blocks', 0) - bl.get('cb_buy_blocks', 0):+d} |",
            f"| 수수료 | {bl['total_fees']:,.0f}원 | {best.user_attrs.get('total_fees', 0):,.0f}원 | {best.user_attrs.get('total_fees', 0) - bl['total_fees']:+,.0f}원 |",
            "",
            f"## 3. 최적 파라미터 (Best Trial #{best.number})",
            "",
            "| 파라미터 | 최적값 | Baseline | 변경 |",
            "|---|---|---|---|",
        ]
        for key, value in sorted(best.params.items()):
            bl_val = baseline_params.get(key, "N/A")
            changed = ""
            if isinstance(bl_val, (int, float)):
                if isinstance(value, float):
                    changed = f"{value - bl_val:+.2f}" if value != bl_val else "-"
                else:
                    changed = f"{value - bl_val:+d}" if value != bl_val else "-"
            if isinstance(value, float):
                bl_str = f"{bl_val:.2f}" if isinstance(bl_val, float) else str(bl_val)
                lines.append(f"| `{key}` | **{value:.2f}** | {bl_str} | {changed} |")
            elif isinstance(value, int) and value >= 1_000_000:
                bl_str = f"{bl_val:,}" if isinstance(bl_val, int) else str(bl_val)
                lines.append(f"| `{key}` | **{value:,}** | {bl_str} | {changed} |")
            else:
                lines.append(f"| `{key}` | **{value}** | {bl_val} | {changed} |")

        # Top 5
        top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        lines += [
            "",
            "## 4. Top 5 Trials",
            "",
            "| # | Score | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 | CB차단 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for t in top5:
            lines.append(
                f"| {t.number} | {t.value:+.2f} "
                f"| -{t.user_attrs.get('mdd', 0):.2f}% "
                f"| {t.user_attrs.get('sharpe', 0):.4f} "
                f"| {t.user_attrs.get('win_rate', 0):.1f}% "
                f"| {t.user_attrs.get('total_sells', 0)} "
                f"| {t.user_attrs.get('stop_loss_count', 0)} "
                f"| {t.user_attrs.get('sideways_days', 0)} "
                f"| {t.user_attrs.get('cb_buy_blocks', 0)} |"
            )

        # Parameter importance
        if len(completed) >= 5:
            try:
                importance = optuna.importance.get_param_importances(study)
                lines += [
                    "",
                    "## 5. 파라미터 중요도 (fANOVA)",
                    "",
                    "| 파라미터 | 중요도 |",
                    "|---|---|",
                ]
                for param, score in sorted(importance.items(), key=lambda x: -x[1]):
                    bar = "█" * int(score * 30)
                    lines.append(f"| `{param}` | {score:.4f} {bar} |")
            except Exception:
                pass

        # Top 5 파라미터 상세
        lines += ["", "## 6. Top 5 파라미터 상세", ""]
        for rank, t in enumerate(top5, 1):
            lines.append(f"### #{rank} — Trial {t.number} (score={t.value:+.2f})")
            lines.append("")
            lines.append("```")
            for key, value in sorted(t.params.items()):
                if isinstance(value, float):
                    lines.append(f"{key} = {value:.2f}")
                elif isinstance(value, int) and value >= 1_000_000:
                    lines.append(f"{key} = {value:_}")
                else:
                    lines.append(f"{key} = {value}")
            lines.append("```")
            lines.append("")

        # config.py 적용 코드
        lines += [f"## 7. config.py 적용 코드 (Best Trial #{best.number})", "", "```python"]
        for key, value in sorted(best.params.items()):
            if isinstance(value, int):
                lines.append(f"{key} = {value:_}" if value >= 1_000_000 else f"{key} = {value}")
            else:
                lines.append(f"{key} = {value:.2f}" if abs(value) >= 0.01 else f"{key} = {value}")
        lines += ["```", ""]

        # Train/Test 비교 섹션
        if test_results:
            lines += [
                "## 8. Train/Test 성과 비교 (과적합 검증)",
                "",
                f"> Train 기간: {self.train_end or '전체'} 이전  |  "
                f"Test 기간: {test_start or 'N/A'} ~ {test_end or '전체'}",
                "",
                "| Rank | Trial# | Train Score | Test 수익률 | Test MDD | Test Sharpe | Test 승률 | 판정 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for item in test_results:
                tr = item["test_result"]
                if tr is None:
                    lines.append(
                        f"| {item['rank']} | #{item['trial_number']} "
                        f"| {item['train_score']:+.2f} | - | - | - | - | 실패 |"
                    )
                    continue
                test_ret = tr["total_return_pct"]
                test_mdd = tr["mdd"]
                if test_ret > 10.0 and test_mdd < 12.0:
                    verdict = "양호"
                elif test_ret > 0.0:
                    verdict = "보통"
                else:
                    verdict = "과적합"
                lines.append(
                    f"| {item['rank']} | #{item['trial_number']} "
                    f"| {item['train_score']:+.2f} "
                    f"| {test_ret:+.2f}% "
                    f"| -{test_mdd:.2f}% "
                    f"| {tr['sharpe']:.4f} "
                    f"| {tr['win_rate']:.1f}% "
                    f"| {verdict} |"
                )
            lines.append("")

        return "\n".join(lines)


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
    obj = opt._make_v4_objective()
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
            test_start=args.test_start,
            test_end=args.test_end,
            phase1_db=args.phase1_db,
            phase1_study=args.phase1_study,
        )

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

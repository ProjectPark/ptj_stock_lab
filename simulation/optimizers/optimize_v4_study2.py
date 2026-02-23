#!/usr/bin/env python3
"""
PTJ v4 — Study 2: Swing Stage 1 수술
======================================
목적:
  Study 1 (3y) 확정 파라미터를 고정 베이스로 두고,
  swing_stage1 손실 문제를 해결하기 위해
  스윙 파라미터만 집중 탐색한다.

탐색 파라미터 (4개):
  1. V4_SWING_TRIGGER_PCT         [10.0, 30.0, step=2.5]   현재=15.0
  2. V4_SWING_STAGE1_DRAWDOWN_PCT [-22.0,  -8.0, step=1.0] 현재=-15.0
  3. V4_SWING_STAGE1_ATR_MULT     [0.5,    4.0,  step=0.5] 현재=1.5
  4. V4_SWING_STAGE1_HOLD_DAYS    [21,     90,   step=7]   현재=63

고정값: Study 1 (3y) 확정 파라미터 (= Phase 1 Best)

목적함수: balanced (phase=2) + swing1 손실 페널티
  - swing_stage1 net PnL < 0 → -2.0점 flat
  - swing_stage1 승률 < 50% → -(50 - wr) × 0.05점

데이터: backtest_1min_3y.parquet (start_date=2023-01-03 ~ )
OOS 평가: 2026-01-01 이후

Usage:
    # Stage 1: baseline (Study 1 3y 확정 파라미터 + config 기본 swing값)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study2.py --stage 1

    # Stage 2: Study 2 탐색 (150 trials)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study2.py --stage 2 \\
      --n-trials 150 --n-jobs 8 \\
      --train-start 2023-01-03 --oos-start 2026-01-01

    # 연속 실행 (Stage 1 → 2)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study2.py \\
      --n-trials 150 --n-jobs 8 \\
      --train-start 2023-01-03 --oos-start 2026-01-01
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
from optuna.samplers import TPESampler

import config
from simulation.optimizers.optimize_v4_optuna import _calc_score
from simulation.optimizers.optimize_v4_study1 import (
    V4Study1Optimizer,
    _PHASE1_BEST_OVERRIDES,
    _run_oos_eval,
)
from simulation.optimizers.optimizer_base import TrialResult


# ============================================================
# Study 1 (3y) 확정값 — Study 2 고정 베이스
# ============================================================
# Study 1 3y 최적화 결과: Phase 1 Best warm-start가 3y 기준 최적
# → _PHASE1_BEST_OVERRIDES 값 전체를 고정 베이스로 사용

# Study 2 탐색 시 고정할 swing 초기값 (warm start용)
_SWING_DEFAULTS: dict[str, Any] = {
    "V4_SWING_TRIGGER_PCT":         config.V4_SWING_TRIGGER_PCT,         # 15.0
    "V4_SWING_STAGE1_DRAWDOWN_PCT": config.V4_SWING_STAGE1_DRAWDOWN_PCT, # -15.0
    "V4_SWING_STAGE1_ATR_MULT":     config.V4_SWING_STAGE1_ATR_MULT,     # 1.5
    "V4_SWING_STAGE1_HOLD_DAYS":    config.V4_SWING_STAGE1_HOLD_DAYS,    # 63
}


# ============================================================
# V4Study2Optimizer
# ============================================================


class V4Study2Optimizer(V4Study1Optimizer):
    """Study 2: Study 1 (3y) Best 고정 + swing 4개 파라미터만 탐색.

    목적함수에 swing_stage1 손실 페널티를 추가하여
    swing1 수익성을 개선한다.
    """

    version = "v4"

    # Study 2 탐색 파라미터 (이름: (low, high, step, best_value))
    # Study 1의 SEARCH_PARAMS / SEARCH_INT_PARAMS를 swing 전용으로 재정의
    SEARCH_PARAMS = {
        "V4_SWING_TRIGGER_PCT":         (10.0,  30.0,  2.5,  15.0),
        "V4_SWING_STAGE1_DRAWDOWN_PCT": (-22.0,  -8.0,  1.0, -15.0),
        "V4_SWING_STAGE1_ATR_MULT":     (0.5,    4.0,   0.5,   1.5),
    }
    SEARCH_INT_PARAMS = {
        "V4_SWING_STAGE1_HOLD_DAYS": (21, 90, 63),  # (low, high, best_value)
    }

    STUDY2_STUDY_NAME = "ptj_v4_study2"
    STUDY1_DB = "sqlite:///data/optuna/optuna_v4_study1.db"
    STUDY1_STUDY = "ptj_v4_study1"

    def __init__(
        self,
        phase1_db: str = "sqlite:///data/optuna/optuna_v4_phase1.db",
        objective_mode: str = "balanced",
        train_start: str | None = None,
    ):
        super().__init__(phase1_db=phase1_db, objective_mode=objective_mode)
        self._train_start = train_start  # Stage 1 실행 시 사용
        self._optuna_reports_dir = _PROJECT_ROOT / "docs" / "reports" / "optuna"

    # ── Stage 1: 3y 데이터로 baseline 실행 ───────────────────

    def run_stage1(self) -> tuple[TrialResult, dict]:
        """Study 2 baseline: Study 1 (3y) 확정값 + config 기본 swing으로 실행.

        train_start가 지정되어 있으면 3y 데이터를 사용한다.
        """
        from simulation.optimizers.optimizer_base import TrialResult
        import time as _time

        print("\n" + "=" * 70)
        print("  [Stage 1] Baseline — Study 2 기준 (Study 1 (3y) 확정 + swing 기본값)")
        print("=" * 70)

        params = self.get_baseline_params()

        run_kwargs: dict = {}
        if self._train_start:
            run_kwargs["start_date"] = self._train_start
            print(f"  start_date = {self._train_start}  (3y 데이터 사용)")

        print("\n  실행 중...")
        t0 = _time.time()
        result = self.run_single_trial(params, **run_kwargs)
        elapsed = _time.time() - t0
        print(f"  완료 ({elapsed:.1f}초)")

        self._print_result_summary(result)

        self.save_baseline_json(result, params)

        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_baseline_report(result, params)
        report_path = self._optuna_reports_dir / "v4_study2_baseline_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"  Baseline 리포트: {report_path}")

        return result, params

    # ── 베이스라인: Study 1 (3y) 확정 + swing 기본값 ──────────

    def get_baseline_params(self) -> dict:
        """Study 1 (3y) 확정 파라미터 + swing 기본값."""
        # Study 1과 동일한 고정 베이스 (Phase 1 Best)
        base = super().get_baseline_params()
        # swing 기본값 추가 (Study 2 baseline: config 기본값)
        base.update(_SWING_DEFAULTS)
        return base

    # ── 탐색 공간: swing 4개만 ────────────────────────────────

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Study 1 (3y) 확정 파라미터 고정 + swing 4개만 탐색."""
        # 베이스: Study 1 고정 + swing 기본값
        params = self.get_baseline_params()

        # 탐색 대상: swing float 3개
        for name, (low, high, step, _) in self.SEARCH_PARAMS.items():
            params[name] = trial.suggest_float(name, low, high, step=step)

        # 탐색 대상: swing int 1개
        for name, (low, high, _) in self.SEARCH_INT_PARAMS.items():
            params[name] = trial.suggest_int(name, low, high, step=7)

        return params

    # ── 목적함수: balanced(phase=2) + swing1 손실 페널티 ──────

    def calc_score(self, result: TrialResult) -> float:
        """Study 2: balanced (phase=2) + swing_stage1 손실 페널티."""
        score = _calc_score(result.to_dict(), self.objective_mode, phase=2)

        # swing_stage1 pnl / 승률 집계 (exit_stats 기반)
        swing1_net_pnl = 0.0
        swing1_count = 0
        swing1_wins = 0
        for key, stats in result.exit_stats.items():
            if key.startswith("swing_stage1"):
                swing1_net_pnl += stats.get("pnl", 0.0)
                swing1_count += stats.get("count", 0)
                swing1_wins += stats.get("wins", 0)

        if swing1_count > 0:
            swing1_wr = swing1_wins / swing1_count * 100

            # 패널티 1: swing1 net PnL 음수 → flat -2.0점
            if swing1_net_pnl < 0:
                score -= 2.0

            # 패널티 2: swing1 승률 < 50% → 비례 감점
            if swing1_wr < 50.0:
                score -= 0.05 * (50.0 - swing1_wr)

        return score

    # ── warm start: config 기본 swing + Study 1 Best swing 값 ─

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            print(f"  기존 완료 trial {already_done}개 발견 — warm start skip")
            return already_done

        # warm start 1: config 기본 swing 값 (현재 설정)
        warm_params = dict(baseline_params)
        for name, (_, _, _, best_val) in self.SEARCH_PARAMS.items():
            warm_params[name] = best_val
        for name, (_, _, best_val) in self.SEARCH_INT_PARAMS.items():
            warm_params[name] = best_val
        study.enqueue_trial(warm_params)
        print("  warm start #0: config 기본 swing 값")

        # warm start 2: baseline (config 기본값 그대로)
        study.enqueue_trial(baseline_params)
        print("  warm start #1: baseline")

        # warm-up 2개 순차 실행
        print("  warm-up 2개 trial 순차 실행 중...")
        obj = self._make_objective(**kwargs)
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  warm-up 완료 (총 완료: {already_done}회)")
        return already_done

    # ── 리포트 ────────────────────────────────────────────────

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._optuna_reports_dir / "v4_study2_optuna_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Study 2 리포트: {report_path}")

        # 콘솔 요약
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return
        best = study.best_trial
        print("\n" + "=" * 60)
        print("  [Study 2 결과] swing_stage1 최적 파라미터")
        print("=" * 60)
        for name in list(self.SEARCH_PARAMS.keys()) + list(self.SEARCH_INT_PARAMS.keys()):
            val = best.params.get(name, "N/A")
            info = self.SEARCH_PARAMS.get(name)
            if info:
                _, _, _, default_val = info
            else:
                _, _, default_val = self.SEARCH_INT_PARAMS.get(name, (None, None, None))
            print(f"  {name:<44s} = {val}  (config: {default_val})")
        print("=" * 60)


# ============================================================
# 병렬 워커
# ============================================================


def _study2_worker(
    study_name: str,
    storage_url: str,
    n_trials: int,
    phase1_db: str,
    objective: str,
    start_date: str | None = None,
    train_end: str | None = None,
) -> None:
    """병렬 워커: 독립 프로세스에서 study.optimize()를 실행한다."""
    import optuna as _optuna
    from optuna.samplers import TPESampler as _TPE

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    sampler = _TPE(n_startup_trials=10, seed=None)
    study = _optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )
    opt = V4Study2Optimizer(phase1_db=phase1_db, objective_mode=objective)
    obj_kwargs: dict = {}
    if start_date:
        obj_kwargs["start_date"] = start_date
    if train_end:
        obj_kwargs["end_date"] = train_end
    obj = opt._make_objective(**obj_kwargs)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)


# ============================================================
# 진짜 병렬 Stage 2 실행기
# ============================================================


def _run_parallel_stage2(
    args: argparse.Namespace,
    opt: "V4Study2Optimizer",
    baseline: dict | None = None,
    baseline_params: dict | None = None,
) -> None:
    """N개 독립 프로세스가 각자 study.optimize()를 실행한다."""
    train_start: str | None = getattr(args, "train_start", None)
    train_end: str | None = getattr(args, "train_end", None)
    oos_start: str | None = getattr(args, "oos_start", None)

    # 1. baseline 로드
    if baseline is None or baseline_params is None:
        baseline, baseline_params = opt.load_baseline_json()
        print(f"  Baseline 로드: {opt._baseline_json}")
    print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")
    if train_start:
        print(f"  훈련 시작: {train_start}  (3y 데이터 자동 선택)")
    if train_end:
        print(f"  훈련 종료: {train_end}")

    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Study 2 Optuna ({args.n_trials} trials, {args.n_jobs} 병렬 프로세스)")
    print(f"  탐색: swing 4개 파라미터  /  고정: Study 1 (3y) 확정값 전체")
    print(f"{'=' * 70}")

    # 2. study 생성 + warm start
    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.db,
        load_if_exists=True,
    )

    obj_kwargs: dict = {}
    if train_start:
        obj_kwargs["start_date"] = train_start
    if train_end:
        obj_kwargs["end_date"] = train_end

    already_done = opt._pre_optimize_setup(study, baseline_params, **obj_kwargs)
    remaining = args.n_trials - already_done

    if remaining <= 0:
        print(f"  목표 trial {args.n_trials}회 이미 완료됨 (완료: {already_done}회)")
        return

    # 3. 병렬 워커 spawn
    n_workers = min(args.n_jobs, remaining)
    base_n = remaining // n_workers
    extra = remaining % n_workers

    worker_args = [
        (
            args.study_name,
            args.db,
            base_n + (1 if i < extra else 0),
            args.phase1_db,
            args.objective,
            train_start,
            train_end,
        )
        for i in range(n_workers)
        if base_n + (1 if i < extra else 0) > 0
    ]

    print(
        f"  워커 {len(worker_args)}개 시작 "
        f"(각 {base_n}~{base_n + (1 if extra else 0)}개 trial) ..."
    )
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        pool.starmap(_study2_worker, worker_args)
    elapsed = time.time() - t0

    # 4. 결과 로드 + 콘솔 요약
    study = optuna.load_study(study_name=args.study_name, storage=args.db)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\n  완료된 trial 없음")
        return

    best = study.best_trial
    diff = best.value - baseline["total_return_pct"]
    print(f"\n  실행 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    print(f"  Trial당 평균: {elapsed / max(len(study.trials), 1):.1f}초")
    print(f"\n  BEST Trial #{best.number}")
    print(f"  스코어  : {best.value:+.2f}  (baseline 대비 {diff:+.2f})")
    print(f"  수익률  : {best.user_attrs.get('total_return_pct', best.value):+.2f}%")
    print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
    print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5 (스코어 기준):")
    print(f"  {'#':>4s}  {'스코어':>8s}  {'MDD':>8s}  {'Sharpe':>8s}  {'승률':>6s}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+7.2f}"
            f"  -{t.user_attrs.get('mdd', 0):6.2f}%"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
            f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
        )

    # swing1 파라미터 비교
    print(f"\n  Top 5 swing 파라미터:")
    swing_keys = list(opt.SEARCH_PARAMS.keys()) + list(opt.SEARCH_INT_PARAMS.keys())
    header = "  " + "  ".join(f"{k:<30s}" for k in swing_keys)
    print(header)
    for t in top5:
        row = "  " + "  ".join(
            f"{t.params.get(k, 'N/A')!s:<30}"
            for k in swing_keys
        )
        print(f"  #{t.number:<4d} {row}")

    # 5. OOS 평가
    if oos_start:
        _run_oos_eval(opt, study, oos_start, top_n=5)

    # 6. 리포트 생성
    opt._post_optimize(study, baseline, baseline_params, elapsed, n_workers)


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="PTJ v4 Study 2 — swing_stage1 수술 (4개 파라미터)"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=150,
                        help="Optuna trial 수 (기본: 150)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="병렬 프로세스 수 (기본: 8)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v4_study2")
    parser.add_argument("--db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_study2.db",
                        help="Optuna DB URL")
    parser.add_argument("--phase1-db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_phase1.db",
                        help="Phase 1 DB URL (Study 1 warm start용)")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"],
                        default="balanced")
    parser.add_argument("--train-start", type=str, default="2023-01-03",
                        help="훈련 시작일 (기본: 2023-01-03, 3y 데이터 자동 선택)")
    parser.add_argument("--train-end", type=str, default=None,
                        help="훈련 종료일 (미지정=엔진 기본값)")
    parser.add_argument("--oos-start", type=str, default="2026-01-01",
                        help="OOS 평가 시작일 (기본: 2026-01-01)")
    args = parser.parse_args()

    opt = V4Study2Optimizer(
        phase1_db=args.phase1_db,
        objective_mode=args.objective,
        train_start=args.train_start,
    )

    print("=" * 70)
    print("  PTJ v4 — Study 2: Swing Stage 1 수술")
    print("  탐색 파라미터: 4개 (V4_SWING_TRIGGER_PCT / DRAWDOWN_PCT / ATR_MULT / HOLD_DAYS)")
    print(f"  고정 기준: Study 1 (3y) 확정값 (Phase 1 Best)")
    print(f"  objective={args.objective} | n-trials={args.n_trials} | n-jobs={args.n_jobs}")
    print(f"  train: {args.train_start} ~ {args.train_end or '엔진 기본'}")
    print(f"  OOS  : {args.oos_start} ~")
    print("=" * 70)

    if args.stage == 1:
        result, params = opt.run_stage1()
        print(f"\n  [Study 2 Baseline] swing defaults:")
        print(f"  TRIGGER={config.V4_SWING_TRIGGER_PCT}%  "
              f"DRAWDOWN={config.V4_SWING_STAGE1_DRAWDOWN_PCT}%  "
              f"ATR_MULT={config.V4_SWING_STAGE1_ATR_MULT}  "
              f"HOLD_DAYS={config.V4_SWING_STAGE1_HOLD_DAYS}d")

    elif args.stage == 2:
        _run_parallel_stage2(args, opt)

    else:
        # Stage 1 → 2 연속 실행
        result, params = opt.run_stage1()
        print()
        _run_parallel_stage2(args, opt, baseline=result.to_dict(), baseline_params=params)

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

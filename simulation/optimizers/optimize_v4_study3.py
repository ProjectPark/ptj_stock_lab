#!/usr/bin/env python3
"""
PTJ v4 — Study 3: CB 필터 + 진입 트리거 최적화
=================================================
목적:
  Study 1 (3y) + Study 2 확정 파라미터를 고정 베이스로 두고,
  아직 탐색되지 않은 Circuit Breaker / 진입 트리거 파라미터를 최적화한다.

탐색 파라미터 (7개):
  Group C — Circuit Breaker / VIX
  1. V4_CB_VIX_SPIKE_PCT       [2.0,  8.0, step=0.5]   현재=3.0
  2. V4_CB_VIX_COOLDOWN_DAYS   [3,   20,   step=1]     현재=13
  3. V4_CB_BTC_CRASH_PCT       [-8.0,-2.0, step=0.5]   현재=-3.5
  4. V4_CB_BTC_SURGE_PCT       [4.0, 12.0, step=0.5]   현재=8.5
  Group E — 진입 트리거
  5. V4_COIN_TRIGGER_PCT       [2.0,  6.0, step=0.5]   현재=3.0
  6. V4_CONL_TRIGGER_PCT       [2.0,  7.0, step=0.5]   현재=3.0
  7. V4_PAIR_GAP_ENTRY_THRESHOLD [1.0, 5.0, step=0.2]  현재=2.0

고정값: Study 1 (3y) 확정 파라미터 + Study 2 swing 최적값

목적함수: balanced (phase=2) + CB 과차단 페널티
  - cb_buy_blocks 이 baseline 대비 200% 초과 시 추가 감점

데이터: backtest_1min_3y.parquet (start_date=2023-01-03 ~ )
OOS 평가: 2026-01-01 이후

Usage:
    # 연속 실행 (Stage 1 → 2)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study3.py \\
      --n-trials 150 --n-jobs 8 \\
      --train-start 2023-01-03 --oos-start 2026-01-01

    # Stage 2만 (baseline 기존 로드)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study3.py --stage 2 \\
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
from simulation.optimizers.optimize_v4_study1 import _PHASE1_BEST_OVERRIDES
from simulation.optimizers.optimize_v4_study2 import V4Study2Optimizer, _run_oos_eval
from simulation.optimizers.optimizer_base import TrialResult


# ============================================================
# Study 2 확정 swing 파라미터
# ============================================================
_STUDY2_BEST_SWING: dict[str, Any] = {
    "V4_SWING_TRIGGER_PCT":         27.5,
    "V4_SWING_STAGE1_DRAWDOWN_PCT": -11.0,
    "V4_SWING_STAGE1_ATR_MULT":     2.5,
    "V4_SWING_STAGE1_HOLD_DAYS":    21,
}

# CB 과차단 페널티 계산을 위한 baseline cb_buy_blocks 참조값
# (Study 2 Stage 1 baseline: 695건 / 1y, 3y 기준은 실행 후 동적 측정)
_CB_BASELINE_BLOCKS: int = 695  # 근사값, 실제 baseline 실행 시 갱신됨


# ============================================================
# V4Study3Optimizer
# ============================================================


class V4Study3Optimizer(V4Study2Optimizer):
    """Study 3: Study 1 (3y) + Study 2 swing 고정 + CB/진입 트리거 7개 탐색.

    목적함수에 CB 과차단 페널티를 추가하여
    너무 보수적인 CB 설정을 방지한다.
    """

    version = "v4"

    # Study 3 탐색 파라미터 (이름: (low, high, step, best_value))
    SEARCH_PARAMS = {
        # Group C: CB / VIX
        "V4_CB_VIX_SPIKE_PCT":       (2.0,  8.0,  0.5,  3.0),
        "V4_CB_BTC_CRASH_PCT":       (-8.0, -2.0,  0.5, -3.5),
        "V4_CB_BTC_SURGE_PCT":       (4.0,  12.0,  0.5,  8.5),
        # Group E: 진입 트리거
        "V4_COIN_TRIGGER_PCT":       (2.0,   6.0,  0.5,  3.0),
        "V4_CONL_TRIGGER_PCT":       (2.0,   7.0,  0.5,  3.0),
        "V4_PAIR_GAP_ENTRY_THRESHOLD": (1.0, 5.0,  0.2,  2.0),
    }
    SEARCH_INT_PARAMS = {
        "V4_CB_VIX_COOLDOWN_DAYS": (3, 20, 13),  # (low, high, best_value)
    }

    STUDY3_STUDY_NAME = "ptj_v4_study3"

    def __init__(
        self,
        phase1_db: str = "sqlite:///data/optuna/optuna_v4_phase1.db",
        objective_mode: str = "balanced",
        train_start: str | None = None,
    ):
        super().__init__(
            phase1_db=phase1_db,
            objective_mode=objective_mode,
            train_start=train_start,
        )
        self._optuna_reports_dir = _PROJECT_ROOT / "docs" / "reports" / "optuna"
        self._cb_baseline_blocks: int = _CB_BASELINE_BLOCKS

    # ── Stage 1 override: Study 2 swing best 포함 ─────────────

    def run_stage1(self) -> tuple[TrialResult, dict]:
        """Study 3 baseline: Study 1(3y) + Study 2 swing best + CB 기본값."""
        import time as _time

        print("\n" + "=" * 70)
        print("  [Stage 1] Baseline — Study 3 기준")
        print("  고정: Study 1 (3y) 확정 + Study 2 swing best")
        print("  CB/진입 트리거: config 기본값")
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
        print(f"    CB 차단: {result.cb_buy_blocks}회")

        # CB 과차단 기준 업데이트
        self._cb_baseline_blocks = result.cb_buy_blocks
        print(f"  CB 과차단 기준: {self._cb_baseline_blocks}회")

        self.save_baseline_json(result, params)

        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_baseline_report(result, params)
        report_path = self._optuna_reports_dir / "v4_study3_baseline_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"  Baseline 리포트: {report_path}")

        return result, params

    # ── 베이스라인: Study 1(3y) + Study 2 swing best + CB 기본값 ──

    def get_baseline_params(self) -> dict:
        """Study 1(3y) 확정 + Study 2 swing best + CB/진입 config 기본값."""
        # Study 2와 동일한 고정 베이스 (Phase 1 Best + swing defaults)
        base = super().get_baseline_params()
        # Study 2 swing best 덮어쓰기
        base.update(_STUDY2_BEST_SWING)
        return base

    # ── 탐색 공간: CB/진입 트리거 7개 ────────────────────────────

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Study 3: Study 1+2 고정 + CB/진입 7개 탐색."""
        params = self.get_baseline_params()

        # 탐색 대상: float 6개
        for name, (low, high, step, _) in self.SEARCH_PARAMS.items():
            params[name] = trial.suggest_float(name, low, high, step=step)

        # 탐색 대상: int 1개
        for name, (low, high, _) in self.SEARCH_INT_PARAMS.items():
            params[name] = trial.suggest_int(name, low, high)

        return params

    # ── 목적함수: balanced(phase=2) + CB 과차단 페널티 ──────────

    def calc_score(self, result: TrialResult) -> float:
        """Study 3: balanced (phase=2) + CB 과차단 페널티."""
        score = _calc_score(result.to_dict(), self.objective_mode, phase=2)

        # CB 과차단 페널티: baseline 대비 3배 초과 시 감점
        cb_blocks = result.cb_buy_blocks
        if self._cb_baseline_blocks > 0 and cb_blocks > self._cb_baseline_blocks * 3:
            excess_ratio = cb_blocks / self._cb_baseline_blocks - 3
            score -= 1.0 * excess_ratio

        return score

    # ── warm start ────────────────────────────────────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            print(f"  기존 완료 trial {already_done}개 발견 — warm start skip")
            return already_done

        # warm start: config 기본값 (현재 Phase 1 Best)
        warm_params = dict(baseline_params)
        for name, (_, _, _, best_val) in self.SEARCH_PARAMS.items():
            warm_params[name] = best_val
        for name, (_, _, best_val) in self.SEARCH_INT_PARAMS.items():
            warm_params[name] = best_val
        study.enqueue_trial(warm_params)
        print("  warm start #0: Phase 1 Best CB/진입 값")

        study.enqueue_trial(baseline_params)
        print("  warm start #1: baseline")

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
        report_path = self._optuna_reports_dir / "v4_study3_optuna_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Study 3 리포트: {report_path}")

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return
        best = study.best_trial
        print("\n" + "=" * 60)
        print("  [Study 3 결과] CB/진입 트리거 최적 파라미터")
        print("=" * 60)
        for name in list(self.SEARCH_PARAMS.keys()) + list(self.SEARCH_INT_PARAMS.keys()):
            val = best.params.get(name, "N/A")
            info = self.SEARCH_PARAMS.get(name)
            if info:
                _, _, _, default_val = info
            else:
                _, _, default_val = self.SEARCH_INT_PARAMS.get(name, (None, None, None))
            print(f"  {name:<40s} = {val}  (Phase1 Best: {default_val})")
        print("=" * 60)


# ============================================================
# 병렬 워커
# ============================================================


def _study3_worker(
    study_name: str,
    storage_url: str,
    n_trials: int,
    phase1_db: str,
    objective: str,
    cb_baseline_blocks: int,
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
    opt = V4Study3Optimizer(phase1_db=phase1_db, objective_mode=objective)
    opt._cb_baseline_blocks = cb_baseline_blocks
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
    opt: "V4Study3Optimizer",
    baseline: dict | None = None,
    baseline_params: dict | None = None,
) -> None:
    """N개 독립 프로세스가 각자 study.optimize()를 실행한다."""
    train_start: str | None = getattr(args, "train_start", None)
    train_end: str | None = getattr(args, "train_end", None)
    oos_start: str | None = getattr(args, "oos_start", None)

    if baseline is None or baseline_params is None:
        baseline, baseline_params = opt.load_baseline_json()
        print(f"  Baseline 로드: {opt._baseline_json}")
    print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")
    if train_start:
        print(f"  훈련 시작: {train_start}  (3y 데이터 자동 선택)")

    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Study 3 Optuna ({args.n_trials} trials, {args.n_jobs} 병렬 프로세스)")
    print(f"  탐색: CB/VIX 4개 + 진입 트리거 3개")
    print(f"  고정: Study 1 (3y) + Study 2 swing 확정값")
    print(f"{'=' * 70}")

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
            opt._cb_baseline_blocks,
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
        pool.starmap(_study3_worker, worker_args)
    elapsed = time.time() - t0

    # 결과 로드 + 콘솔 요약
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
    print(f"  스코어  : {best.value:+.4f}  (baseline 대비 {diff:+.2f})")
    print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
    print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  CB 차단 : {best.user_attrs.get('cb_buy_blocks', 0)}")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    print(f"  {'#':>4s}  {'스코어':>8s}  {'MDD':>7s}  {'Sharpe':>8s}  {'WR':>6s}  {'CB차단':>6s}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+8.4f}"
            f"  -{t.user_attrs.get('mdd', 0):5.2f}%"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
            f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
            f"  {t.user_attrs.get('cb_buy_blocks', 0):>6}"
        )

    # 파라미터 중요도
    search_keys = list(opt.SEARCH_PARAMS.keys()) + list(opt.SEARCH_INT_PARAMS.keys())
    print(f"\n  Best CB/진입 파라미터:")
    for k in search_keys:
        val = best.params.get(k, "N/A")
        info = opt.SEARCH_PARAMS.get(k)
        default = info[3] if info else opt.SEARCH_INT_PARAMS.get(k, (None, None, None))[2]
        print(f"    {k:<40s} = {val}  (현재: {default})")

    if oos_start:
        _run_oos_eval(opt, study, oos_start, top_n=5)

    opt._post_optimize(study, baseline, baseline_params, elapsed, n_workers)


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="PTJ v4 Study 3 — CB 필터 + 진입 트리거 최적화 (7개 파라미터)"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=150,
                        help="Optuna trial 수 (기본: 150)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="병렬 프로세스 수 (기본: 8)")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="ptj_v4_study3")
    parser.add_argument("--db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_study3.db")
    parser.add_argument("--phase1-db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_phase1.db")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"],
                        default="balanced")
    parser.add_argument("--train-start", type=str, default="2023-01-03",
                        help="훈련 시작일 (기본: 2023-01-03, 3y 데이터 자동 선택)")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--oos-start", type=str, default="2026-01-01")
    args = parser.parse_args()

    opt = V4Study3Optimizer(
        phase1_db=args.phase1_db,
        objective_mode=args.objective,
        train_start=args.train_start,
    )

    print("=" * 70)
    print("  PTJ v4 — Study 3: CB 필터 + 진입 트리거 최적화")
    print("  탐색: CB/VIX 4개 + 진입 트리거 3개")
    print("  고정: Study 1 (3y) + Study 2 swing 확정값")
    print(f"  objective={args.objective} | n-trials={args.n_trials} | n-jobs={args.n_jobs}")
    print(f"  train: {args.train_start} ~ {args.train_end or '엔진 기본'}")
    print(f"  OOS  : {args.oos_start} ~")
    print("=" * 70)

    if args.stage == 1:
        result, params = opt.run_stage1()
        print(f"\n  Study 2 swing 확정값 적용됨:")
        for k, v in _STUDY2_BEST_SWING.items():
            print(f"    {k} = {v}")

    elif args.stage == 2:
        _run_parallel_stage2(args, opt)

    else:
        result, params = opt.run_stage1()
        print()
        _run_parallel_stage2(args, opt, baseline=result.to_dict(), baseline_params=params)

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

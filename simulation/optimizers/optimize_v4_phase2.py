#!/usr/bin/env python3
"""
PTJ v4 — Phase 2: Narrow Search (확정 파라미터 주변 촘촘한 탐색)
================================================================
목적:
  Study 1~5 확정값을 중심으로 좁은 범위 + 작은 step으로 정밀 탐색.
  Phase 1(Study 1~5)에서 발견한 최적점 주변의 지형을 촘촘히 스캔하여
  글로벌 최적에 더 가까운 파라미터 조합을 찾는다.

배경:
  - Study 2 확정: SWING_TRIGGER=27.5, DRAWDOWN=-11.0, ATR_MULT=2.5, HOLD=21
  - Study 5 확정: BTC_SURGE=13.5, BTC_CRASH=-6.0, FIXED_TP=6.5, GAP_SELL=9.0
  - Study 1 확정: STOP_LOSS=-4.25, DCA_DROP=-1.35, CONL_SELL=4.5
  - Study 4 핵심: SIDEWAYS_ATR_DECLINE(fANOVA=0.87) — 좁은 범위 재탐색

탐색 파라미터 (13개, 좁은 범위):
  Swing (4개)
  1. V4_SWING_TRIGGER_PCT          [24.0, 31.0, step=0.5]    확정=27.5
  2. V4_SWING_STAGE1_DRAWDOWN_PCT  [-13.0, -9.0, step=0.5]   확정=-11.0
  3. V4_SWING_STAGE1_ATR_MULT      [1.75, 3.25, step=0.25]   확정=2.5
  4. V4_SWING_STAGE1_HOLD_DAYS     [14, 28, step=7]          확정=21

  매도/BTC CB (5개)
  5. V4_CB_BTC_SURGE_PCT           [11.0, 16.0, step=0.5]    확정=13.5
  6. V4_CB_BTC_CRASH_PCT           [-7.5, -4.5, step=0.5]    확정=-6.0
  7. V4_PAIR_FIXED_TP_PCT          [5.0, 8.5, step=0.5]      확정=6.5
  8. V4_PAIR_IMMEDIATE_SELL_PCT    [0.25, 0.55, step=0.05]   확정=0.4
  9. PAIR_GAP_SELL_THRESHOLD_V2    [7.5, 11.0, step=0.5]     확정=9.0

  공유 매도/손절 (3개)
  10. STOP_LOSS_PCT                [-5.5, -3.0, step=0.25]    확정=-4.25
  11. CONL_SELL_PROFIT_PCT         [3.0, 6.0, step=0.5]       확정=4.5
  12. DCA_DROP_PCT                 [-1.8, -0.9, step=0.05]    확정=-1.35

  횡보장 핵심 (1개)
  13. V4_SIDEWAYS_ATR_DECLINE_PCT  [5.0, 15.0, step=1.0]     현재=10.0

고정값: 위 13개 외 전체 — Study 1+2+5 확정값

목적함수: balanced (phase=2)
데이터: backtest_1min_3y.parquet (start_date=2023-01-03 ~ )
OOS 평가: 2026-01-01 이후

Usage:
    # 연속 실행 (Stage 1 → 2)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_phase2.py \\
      --n-trials 250 --n-jobs 8 \\
      --train-start 2023-01-03 --oos-start 2026-01-01

    # Stage 2만
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_phase2.py --stage 2 \\
      --n-trials 250 --n-jobs 8 \\
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
from simulation.optimizers.optimize_v4_study2 import V4Study2Optimizer, _run_oos_eval
from simulation.optimizers.optimize_v4_study5 import _STUDY2_BEST_SWING
from simulation.optimizers.optimizer_base import TrialResult


# ============================================================
# Phase 2 확정 파라미터 (Study 1+2+5 통합)
# ============================================================
_PHASE2_CONFIRMED: dict[str, Any] = {
    # Study 2 Swing Best
    **_STUDY2_BEST_SWING,
    # Study 5 매도/BTC CB Best (config.py 반영됨)
    "V4_PAIR_FIXED_TP_PCT":        config.V4_PAIR_FIXED_TP_PCT,        # 6.5
    "V4_PAIR_IMMEDIATE_SELL_PCT":  config.V4_PAIR_IMMEDIATE_SELL_PCT,  # 0.40
    "PAIR_GAP_SELL_THRESHOLD_V2":  config.PAIR_GAP_SELL_THRESHOLD_V2,  # 9.0
    "V4_CB_BTC_CRASH_PCT":         config.V4_CB_BTC_CRASH_PCT,         # -6.0
    "V4_CB_BTC_SURGE_PCT":         config.V4_CB_BTC_SURGE_PCT,         # 13.5
    # Study 1 확정
    "STOP_LOSS_PCT":               -4.25,
    "CONL_SELL_PROFIT_PCT":        4.5,
    "DCA_DROP_PCT":                -1.35,
    # Study 4 핵심 (현재 config 값)
    "V4_SIDEWAYS_ATR_DECLINE_PCT": config.V4_SIDEWAYS_ATR_DECLINE_PCT,  # 10.0 (20.0→config는 다를 수 있음)
}

# ============================================================
# Phase 2 탐색 범위 (좁은 범위 + 작은 step)
# ============================================================
# 형식: (low, high, step, center_value)
_NARROW_FLOAT_PARAMS: dict[str, tuple[float, float, float, float]] = {
    # Swing (Study 2 확정 주변)
    "V4_SWING_TRIGGER_PCT":         (24.0,  31.0,  0.5,  27.5),
    "V4_SWING_STAGE1_DRAWDOWN_PCT": (-13.0, -9.0,  0.5,  -11.0),
    "V4_SWING_STAGE1_ATR_MULT":     (1.75,  3.25,  0.25, 2.5),
    # 매도/BTC CB (Study 5 확정 주변)
    "V4_CB_BTC_SURGE_PCT":          (11.0,  16.0,  0.5,  13.5),
    "V4_CB_BTC_CRASH_PCT":          (-7.5,  -4.5,  0.5,  -6.0),
    "V4_PAIR_FIXED_TP_PCT":         (5.0,   8.5,   0.5,  6.5),
    "V4_PAIR_IMMEDIATE_SELL_PCT":   (0.25,  0.55,  0.05, 0.4),
    "PAIR_GAP_SELL_THRESHOLD_V2":   (7.5,   11.0,  0.5,  9.0),
    # 공유 매도/손절 (Study 1 확정 주변)
    "STOP_LOSS_PCT":                (-5.5,  -3.0,  0.25, -4.25),
    "CONL_SELL_PROFIT_PCT":         (3.0,   6.0,   0.5,  4.5),
    "DCA_DROP_PCT":                 (-1.8,  -0.9,  0.05, -1.35),
    # 횡보장 핵심 (Study 4 fANOVA=0.87)
    "V4_SIDEWAYS_ATR_DECLINE_PCT":  (5.0,   15.0,  1.0,  10.0),
}
_NARROW_INT_PARAMS: dict[str, tuple[int, int, int, int]] = {
    # Swing hold days (Study 2 확정 주변)
    "V4_SWING_STAGE1_HOLD_DAYS":    (14, 28, 7, 21),
}


# ============================================================
# V4Phase2Optimizer
# ============================================================


class V4Phase2Optimizer(V4Study2Optimizer):
    """Phase 2: 확정 파라미터 주변 Narrow Search (13개 파라미터).

    Study 1+2+5 확정값을 중심으로 좁은 범위 + 작은 step으로 정밀 탐색.
    """

    version = "v4"

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

    # ── Baseline: Study 1+2+5 확정값 전체 ─────────────────────

    def get_baseline_params(self) -> dict:
        """Study 1+2+5 확정값 전체를 baseline으로 사용."""
        base = super().get_baseline_params()
        base.update(_STUDY2_BEST_SWING)
        base.update(_PHASE2_CONFIRMED)
        return base

    # ── Stage 1 ──────────────────────────────────────────────

    def run_stage1(self) -> tuple[TrialResult, dict]:
        """Phase 2 baseline: Study 1+2+5 확정값 기준."""
        import time as _time

        print("\n" + "=" * 70)
        print("  [Stage 1] Baseline — Phase 2 기준 (Study 1+2+5 확정값)")
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
        print(f"    CB 차단   : {result.cb_buy_blocks}회")
        print(f"    횡보장 일수: {result.sideways_days}일 / {result.total_trading_days}일")

        self.save_baseline_json(result, params)

        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_baseline_report(result, params)
        report_path = self._optuna_reports_dir / "v4_phase2_baseline_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"  Baseline JSON : {self._baseline_json}")
        print(f"  Baseline 리포트: {report_path}")

        return result, params

    # ── 탐색 공간: 13개 좁은 범위 ─────────────────────────────

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Phase 2: Study 1+2+5 확정값 고정 + 13개 파라미터 좁은 범위 탐색."""
        params = self.get_baseline_params()

        for name, (low, high, step, _) in _NARROW_FLOAT_PARAMS.items():
            params[name] = trial.suggest_float(name, low, high, step=step)

        for name, (low, high, step, _) in _NARROW_INT_PARAMS.items():
            params[name] = trial.suggest_int(name, low, high, step=step)

        return params

    # ── 목적함수: balanced (phase=2) ─────────────────────────

    def calc_score(self, result: TrialResult) -> float:
        """Phase 2: balanced (phase=2)."""
        return _calc_score(result.to_dict(), self.objective_mode, phase=2)

    # ── warm start ───────────────────────────────────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            print(f"  기존 완료 trial {already_done}개 발견 — warm start skip")
            return already_done

        # warm start #0: 확정 center 값
        warm_params = dict(baseline_params)
        for name, (_, _, _, center) in _NARROW_FLOAT_PARAMS.items():
            warm_params[name] = center
        for name, (_, _, _, center) in _NARROW_INT_PARAMS.items():
            warm_params[name] = center
        study.enqueue_trial(warm_params)
        print("  warm start #0: 확정 center 값")

        # warm start #1: baseline (config 현재값)
        study.enqueue_trial(baseline_params)
        print("  warm start #1: baseline")

        # warm-up 순차 실행
        print("  warm-up 2개 trial 순차 실행 중...")
        obj = self._make_objective(**kwargs)
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  warm-up 완료 (총 완료: {already_done}회)")
        return already_done

    # ── 리포트 ───────────────────────────────────────────────

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._optuna_reports_dir / "v4_phase2_optuna_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Phase 2 리포트: {report_path}")

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return
        best = study.best_trial
        print("\n" + "=" * 65)
        print(f"  [Phase 2 결과] Narrow Search Best Trial #{best.number}")
        print("=" * 65)
        for name in list(_NARROW_FLOAT_PARAMS.keys()) + list(_NARROW_INT_PARAMS.keys()):
            val = best.params.get(name, "N/A")
            center = _NARROW_FLOAT_PARAMS.get(name, _NARROW_INT_PARAMS.get(name, (0, 0, 0, "N/A")))[3]
            print(f"  {name:<44s} = {val}  (확정: {center})")
        print("=" * 65)


# ============================================================
# 병렬 워커
# ============================================================


def _phase2_worker(
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
    opt = V4Phase2Optimizer(phase1_db=phase1_db, objective_mode=objective)
    obj_kwargs: dict = {}
    if start_date:
        obj_kwargs["start_date"] = start_date
    if train_end:
        obj_kwargs["end_date"] = train_end
    obj = opt._make_objective(**obj_kwargs)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)


# ============================================================
# 병렬 Stage 2 실행기
# ============================================================


def _run_parallel_stage2(
    args: argparse.Namespace,
    opt: "V4Phase2Optimizer",
    baseline: dict | None = None,
    baseline_params: dict | None = None,
) -> None:
    train_start: str | None = getattr(args, "train_start", None)
    train_end: str | None = getattr(args, "train_end", None)
    oos_start: str | None = getattr(args, "oos_start", None)

    if baseline is None or baseline_params is None:
        baseline, baseline_params = opt.load_baseline_json()
        print(f"  Baseline 로드: {opt._baseline_json}")
    print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")
    if train_start:
        print(f"  훈련 시작: {train_start}  (3y 데이터 자동 선택)")

    n_params = len(_NARROW_FLOAT_PARAMS) + len(_NARROW_INT_PARAMS)
    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Phase 2 Narrow Search ({args.n_trials} trials, {args.n_jobs} workers)")
    print(f"  탐색: 확정값 주변 {n_params}개 파라미터 (좁은 범위 + 작은 step)")
    print(f"  고정: Study 1+2+5 확정값")
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
            train_start,
            train_end,
        )
        for i in range(n_workers)
        if base_n + (1 if i < extra else 0) > 0
    ]

    print(f"  워커 {len(worker_args)}개 시작 (각 {base_n}~{base_n + (1 if extra else 0)}개 trial) ...")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        pool.starmap(_phase2_worker, worker_args)
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
    print(f"  스코어    : {best.value:+.4f}  (baseline 대비 {diff:+.2f})")
    print(f"  MDD       : -{best.user_attrs.get('mdd', 0):.2f}%")
    print(f"  Sharpe    : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  승률      : {best.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  CB 차단   : {best.user_attrs.get('cb_buy_blocks', 0)}회")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    header = f"  {'#':>4s}  {'스코어':>8s}  {'MDD':>7s}  {'WR':>6s}  {'Sharpe':>8s}"
    print(header)
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+8.4f}"
            f"  -{t.user_attrs.get('mdd', 0):5.2f}%"
            f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
        )

    if oos_start:
        _run_oos_eval(opt, study, oos_start, top_n=5)

    opt._post_optimize(study, baseline, baseline_params, elapsed, n_workers)


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="PTJ v4 Phase 2 — Narrow Search (확정 파라미터 주변 촘촘한 탐색, 13개)"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None)
    parser.add_argument("--n-trials", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="ptj_v4_phase2")
    parser.add_argument("--db", type=str, default="sqlite:///data/optuna/optuna_v4_phase2.db")
    parser.add_argument("--phase1-db", type=str, default="sqlite:///data/optuna/optuna_v4_phase1.db")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"], default="balanced")
    parser.add_argument("--train-start", type=str, default="2023-01-03")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--oos-start", type=str, default="2026-01-01")
    args = parser.parse_args()

    opt = V4Phase2Optimizer(
        phase1_db=args.phase1_db,
        objective_mode=args.objective,
        train_start=args.train_start,
    )

    n_params = len(_NARROW_FLOAT_PARAMS) + len(_NARROW_INT_PARAMS)
    print("=" * 70)
    print("  PTJ v4 — Phase 2: Narrow Search (확정 파라미터 주변 촘촘한 탐색)")
    print(f"  탐색: {n_params}개 파라미터 (좁은 범위 + 작은 step)")
    print(f"  고정: Study 1+2+5 확정값")
    print(f"  objective={args.objective} | n-trials={args.n_trials} | n-jobs={args.n_jobs}")
    print(f"  train: {args.train_start} ~ {args.train_end or '엔진 기본'}")
    print(f"  OOS  : {args.oos_start} ~")
    print("=" * 70)

    if args.stage == 1:
        opt.run_stage1()
    elif args.stage == 2:
        _run_parallel_stage2(args, opt)
    else:
        result, params = opt.run_stage1()
        print()
        _run_parallel_stage2(args, opt, baseline=result.to_dict(), baseline_params=params)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

#!/usr/bin/env python3
"""
PTJ v4 — Study 1: Phase 1 기반 재설정 (좁은 탐색)
====================================================
목적:
  Phase 1 Best #388 파라미터를 고정 베이스로 두고,
  수익 극대화와 직결된 7개 파라미터만 좁은 범위로 재확정한다.
  확정된 값은 Study 2 (swing_stage1 수술)의 고정 기준값이 된다.

탐색 파라미터 (7개):
  1. PAIR_GAP_SELL_THRESHOLD_V2  [4.0, 9.0]   Phase1 Best=6.6
  2. V4_PAIR_FIXED_TP_PCT        [5.5, 10.0]  Phase1 Best=7.5
  3. V4_PAIR_IMMEDIATE_SELL_PCT  [0.15, 0.35] Phase1 Best=0.20
  4. V4_DCA_MAX_COUNT            [1, 2]       Phase1 Best=1
  5. DCA_DROP_PCT                [-1.8, -0.8] Phase1 Best=-1.35
  6. CONL_SELL_PROFIT_PCT        [3.5, 6.0]   Phase1 Best=4.5
  7. STOP_LOSS_BULLISH_PCT       [-18.0, -12.0] Phase1 Best=-16.0

나머지 파라미터: Phase 1 Best #388 고정값 사용

Usage:
    # Stage 1: baseline (Phase 1 Best 기준으로 재실행)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study1.py --stage 1

    # Stage 2: Study 1 탐색 (150 trials)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study1.py --stage 2 \\
      --n-trials 150 --n-jobs 8 \\
      --study-name ptj_v4_study1 \\
      --db sqlite:///data/optuna/optuna_v4_study1.db

    # 연속 실행 (Stage 1 → 2)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study1.py \\
      --n-trials 150 --n-jobs 8 \\
      --db sqlite:///data/optuna/optuna_v4_study1.db
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
from simulation.optimizers.optimize_v4_optuna import V4Optimizer, _calc_score
from simulation.optimizers.optimizer_base import TrialResult


# ============================================================
# Phase 1 Best #388 고정값
# ============================================================
# optuna_param_comparison.md 기준 Best #388 주요 파라미터
# 탐색 대상 7개는 warm start 기본값으로도 사용됨
_PHASE1_BEST_OVERRIDES: dict[str, Any] = {
    # ── Group A: 공유 파라미터 ──────────────────────────────
    "STOP_LOSS_PCT":             -4.25,
    "STOP_LOSS_BULLISH_PCT":     -16.0,   # [탐색 대상]
    "CONL_SELL_PROFIT_PCT":       4.5,    # [탐색 대상]
    "DCA_DROP_PCT":              -1.35,   # [탐색 대상]
    "PAIR_GAP_SELL_THRESHOLD_V2": 6.6,   # [탐색 대상]
    # ── Group B: 횡보장 신규 ────────────────────────────────
    "V4_SIDEWAYS_ATR_DECLINE_PCT":    10.0,
    "V4_SIDEWAYS_VOLUME_DECLINE_PCT": 15.0,
    "V4_SIDEWAYS_RSI_LOW":            35.0,
    "V4_SIDEWAYS_RSI_HIGH":           65.0,
    # ── Group C: CB / 고변동성 ──────────────────────────────
    "V4_CB_VIX_SPIKE_PCT":       3.0,
    "V4_CB_VIX_COOLDOWN_DAYS":  13,
    "V4_CB_BTC_CRASH_PCT":      -3.5,
    "V4_CB_BTC_SURGE_PCT":       8.5,
    "V4_CB_GLD_COOLDOWN_DAYS":   1,
    # ── Group D: CONL 필터 / 분할매도 ───────────────────────
    "V4_CONL_ADX_MIN":               10.0,
    "V4_CONL_EMA_SLOPE_MIN_PCT":      0.5,
    "V4_PAIR_IMMEDIATE_SELL_PCT":     0.2,   # [탐색 대상]
    "V4_PAIR_FIXED_TP_PCT":           7.5,   # [탐색 대상]
    # ── Group E: 진입 타이밍 / 자금 ─────────────────────────
    "V4_PAIR_GAP_ENTRY_THRESHOLD":    2.0,
    "V4_DCA_MAX_COUNT":               1,     # [탐색 대상]
    "V4_COIN_TRIGGER_PCT":            3.0,
    "V4_CONL_TRIGGER_PCT":            3.0,
    "V4_SPLIT_BUY_INTERVAL_MIN":     10,
}


def _load_phase1_best(
    phase1_db: str = "sqlite:///data/optuna/optuna_v4_phase1.db",
    study_name: str = "ptj_v4_phase1",
) -> dict:
    """Phase 1 DB에서 Best Trial 파라미터를 로드한다. 실패 시 빈 dict 반환."""
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name=study_name, storage=phase1_db)
        best = study.best_trial
        print(f"  [Phase 1 Best] Trial #{best.number} 로드 완료 (score={best.value:+.2f})")
        return dict(best.params)
    except Exception as exc:
        print(f"  [WARN] Phase 1 DB 로드 실패: {exc}")
        print("  → _PHASE1_BEST_OVERRIDES 하드코딩 값으로 대체합니다.")
        return {}


# ============================================================
# V4Study1Optimizer
# ============================================================


class V4Study1Optimizer(V4Optimizer):
    """Study 1: Phase 1 Best 고정 + 7개 파라미터만 좁은 범위 탐색."""

    version = "v4"

    # Study 1 탐색 파라미터 정의 (이름: (low, high, step, best_value))
    SEARCH_PARAMS = {
        "PAIR_GAP_SELL_THRESHOLD_V2":  (4.0,   9.0,  0.2,  6.6),
        "V4_PAIR_FIXED_TP_PCT":        (5.5,  10.0,  0.5,  7.5),
        "V4_PAIR_IMMEDIATE_SELL_PCT":  (0.15,  0.35, 0.05, 0.20),
        "DCA_DROP_PCT":                (-1.8, -0.8,  0.05, -1.35),
        "CONL_SELL_PROFIT_PCT":        (3.5,   6.0,  0.5,  4.5),
        "STOP_LOSS_BULLISH_PCT":       (-18.0, -12.0, 0.5, -16.0),
    }
    SEARCH_INT_PARAMS = {
        "V4_DCA_MAX_COUNT": (1, 2, 1),  # (low, high, best_value)
    }

    STUDY1_STUDY_NAME = "ptj_v4_study1"
    PHASE1_DB = "sqlite:///data/optuna/optuna_v4_phase1.db"
    PHASE1_STUDY = "ptj_v4_phase1"

    def __init__(
        self,
        phase1_db: str = "sqlite:///data/optuna/optuna_v4_phase1.db",
        objective_mode: str = "balanced",
    ):
        super().__init__(gap_max=4.0, objective_mode=objective_mode, phase=1)
        # Phase 1 Best를 DB에서 로드 (실패 시 하드코딩 값 사용)
        loaded = _load_phase1_best(phase1_db)
        self._fixed_params = dict(_PHASE1_BEST_OVERRIDES)
        if loaded:
            self._fixed_params.update(loaded)   # DB 값이 우선
        # 결과 경로 재지정 (study1 전용)
        self._optuna_reports_dir = _PROJECT_ROOT / "docs" / "reports" / "optuna"

    # ── 베이스라인: Phase 1 Best 기준으로 재실행 ─────────────

    def get_baseline_params(self) -> dict:
        """Phase 1 Best 고정값 + config 기본값으로 전체 파라미터를 구성한다."""
        base = super().get_baseline_params()   # config 현재값 전체
        base.update(self._fixed_params)         # Phase 1 Best 값으로 덮어쓰기
        return base

    # ── 탐색 공간: 7개만 좁은 범위 ───────────────────────────

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Phase 1 Best 고정 + 7개 파라미터만 탐색한다."""
        # 베이스: Phase 1 Best 전체 고정
        params = self.get_baseline_params()

        # 탐색 대상 6개 float 파라미터 (좁은 범위)
        for name, (low, high, step, _) in self.SEARCH_PARAMS.items():
            params[name] = trial.suggest_float(name, low, high, step=step)

        # 탐색 대상 1개 int 파라미터
        for name, (low, high, _) in self.SEARCH_INT_PARAMS.items():
            params[name] = trial.suggest_int(name, low, high)

        return params

    # ── 목적함수 ────────────────────────────────────────────

    def calc_score(self, result: TrialResult) -> float:
        return _calc_score(result.to_dict(), self.objective_mode, phase=1)

    # ── warm start: Phase 1 Best 값으로 enqueue ─────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            print(f"  기존 완료 trial {already_done}개 발견 — warm start skip")
            return already_done

        # warm start: Phase 1 Best 7개 파라미터 값으로 enqueue
        warm_params = dict(baseline_params)
        for name, (_, _, _, best_val) in self.SEARCH_PARAMS.items():
            warm_params[name] = best_val
        for name, (_, _, best_val) in self.SEARCH_INT_PARAMS.items():
            warm_params[name] = best_val

        study.enqueue_trial(warm_params)
        print("  Phase 1 Best warm start enqueue 완료 (Trial #0)")

        # baseline(config 기본값)도 비교용 enqueue
        study.enqueue_trial(baseline_params)
        print("  Phase 1 baseline enqueue 완료 (Trial #1)")

        # warm-up 2개 순차 실행
        print("  warm-up 2개 trial 순차 실행 중...")
        obj = self._make_objective()
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  warm-up 완료 (총 완료: {already_done}회)")
        return already_done

    # ── 리포트 경로: study1 전용 ─────────────────────────────

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        self._optuna_reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._optuna_reports_dir / "v4_study1_optuna_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Study 1 리포트: {report_path}")

        # 콘솔에 Study 1 결과 요약 출력
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best = study.best_trial
            print("\n" + "=" * 60)
            print("  [Study 1 결과] 확정 파라미터 (Study 2에 고정할 값)")
            print("=" * 60)
            for name in list(self.SEARCH_PARAMS.keys()) + list(self.SEARCH_INT_PARAMS.keys()):
                val = best.params.get(name, "N/A")
                _, _, _, phase1_val = self.SEARCH_PARAMS.get(
                    name, (None, None, None, self.SEARCH_INT_PARAMS.get(name, (None, None, None))[2])
                )
                print(f"  {name:<40s} = {val}  (Phase1 Best: {phase1_val})")
            print("=" * 60)


# ============================================================
# 병렬 워커
# ============================================================


def _study1_worker(
    study_name: str,
    storage_url: str,
    n_trials: int,
    phase1_db: str,
    objective: str,
    start_date: str | None = None,
    train_end: str | None = None,
) -> None:
    """병렬 워커 프로세스에서 독립적으로 study.optimize()를 실행한다."""
    import optuna as _optuna
    from optuna.samplers import TPESampler as _TPE

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    sampler = _TPE(n_startup_trials=15, seed=None)  # 각 워커 seed 다르게
    study = _optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )
    opt = V4Study1Optimizer(phase1_db=phase1_db, objective_mode=objective)
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
    opt: "V4Study1Optimizer",
    baseline: dict | None = None,
    baseline_params: dict | None = None,
) -> None:
    """N개 독립 프로세스가 각자 study.optimize()를 실행한다.

    각 워커가 SQLite를 통해 공유 study에 trial을 기록하므로
    진짜 병렬 최적화가 가능하다. optimizer_base.run_stage2()의
    Pool-but-not-used 버그를 우회한다.
    """
    # train_start / train_end 처리
    train_start: str | None = getattr(args, "train_start", None)
    train_end: str | None = getattr(args, "train_end", None)
    oos_start: str | None = getattr(args, "oos_start", None)

    # 1. baseline 로드
    if baseline is None or baseline_params is None:
        baseline, baseline_params = opt.load_baseline_json()
        print(f"  Baseline 로드: {opt._baseline_json}")
    print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")
    if train_start:
        print(f"  훈련 시작: {train_start}  (3y 데이터 사용)")
    if train_end:
        print(f"  훈련 종료: {train_end}")

    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Optuna 최적화 ({args.n_trials} trials, {args.n_jobs} 병렬 프로세스)")
    print(f"{'=' * 70}")

    # 2. study 생성 + warm start (메인 프로세스에서만 수행)
    sampler = TPESampler(seed=42, n_startup_trials=15)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=args.db,
        load_if_exists=True,
    )

    already_done = opt._pre_optimize_setup(study, baseline_params)
    remaining = args.n_trials - already_done

    if remaining <= 0:
        print(f"  목표 trial {args.n_trials}회 이미 완료됨 (완료: {already_done}회)")
        return

    # 3. 병렬 워커 spawn: 각 프로세스가 독립적으로 study.optimize() 실행
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
        pool.starmap(_study1_worker, worker_args)
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
    print(f"  수익률  : {best.value:+.2f}%  (baseline 대비 {diff:+.2f}%)")
    print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
    print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    print(f"  {'#':>4s}  {'수익률':>8s}  {'MDD':>8s}  {'Sharpe':>8s}  {'승률':>6s}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+7.2f}%"
            f"  -{t.user_attrs.get('mdd', 0):6.2f}%"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
            f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
        )

    # 5. OOS 평가 (--oos-start 지정 시)
    if oos_start:
        _run_oos_eval(opt, study, oos_start, top_n=5)

    # 6. 리포트 생성
    opt._post_optimize(study, baseline, baseline_params, elapsed, n_workers)


def _run_oos_eval(
    opt: "V4Study1Optimizer",
    study: "optuna.Study",
    oos_start: str,
    oos_end: str | None = None,
    top_n: int = 5,
) -> None:
    """Top-N trial을 OOS 기간(2026-01 이후)으로 평가한다."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]

    print(f"\n{'=' * 70}")
    print(f"  [OOS 평가] {oos_start} ~ {oos_end or '최종'} (Top {top_n})")
    print(f"{'=' * 70}")

    for rank, t in enumerate(top_trials, 1):
        print(f"\n  [{rank}/{top_n}] Trial #{t.number}  (train score={t.value:+.2f})")
        try:
            res = opt.run_single_trial(
                t.params,
                start_date=oos_start,
                end_date=oos_end,
            )
            print(f"  OOS 수익률: {res.total_return_pct:+.2f}%")
            print(f"  OOS MDD   : -{res.mdd:.2f}%")
            print(f"  OOS 승률  : {res.win_rate:.1f}%  ({res.win_count}W / {res.loss_count}L)")
            print(f"  OOS 매수  : {res.total_buys}회 / 매도: {res.total_sells}회")
        except Exception as exc:
            print(f"  [WARN] OOS 평가 실패: {exc}")


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="PTJ v4 Study 1 — Phase 1 기반 재설정 (7개 파라미터)")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=150, help="Optuna trial 수 (기본: 150)")
    parser.add_argument("--n-jobs", type=int, default=8,  help="병렬 프로세스 수 (기본: 8)")
    parser.add_argument("--timeout", type=int, default=None, help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v4_study1")
    parser.add_argument("--db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_study1.db",
                        help="Optuna DB URL")
    parser.add_argument("--phase1-db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_phase1.db",
                        help="Phase 1 DB URL (warm start용)")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"],
                        default="balanced")
    parser.add_argument("--train-start", type=str, default=None,
                        help="훈련 시작일 (예: 2023-01-01, 미지정=엔진 기본값). "
                             "2025-01-01 이전이면 backtest_1min_3y.parquet 자동 선택")
    parser.add_argument("--train-end", type=str, default=None,
                        help="훈련 종료일 (예: 2025-12-31, 미지정=엔진 기본값)")
    parser.add_argument("--oos-start", type=str, default=None,
                        help="OOS 평가 시작일 (예: 2026-01-01). 지정 시 Top-5 OOS 평가 수행")
    args = parser.parse_args()

    opt = V4Study1Optimizer(phase1_db=args.phase1_db, objective_mode=args.objective)

    print("=" * 70)
    print("  PTJ v4 — Study 1: Phase 1 기반 재설정")
    print("  탐색 파라미터: 7개 (좁은 범위)")
    print(f"  고정 기준: Phase 1 Best #388")
    print(f"  objective={args.objective} | n-trials={args.n_trials} | n-jobs={args.n_jobs}")
    print("=" * 70)

    if args.stage == 1:
        opt.run_stage1()

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

#!/usr/bin/env python3
"""
PTJ v3 — Optuna 기반 파라미터 최적화
=====================================
Stage 1: 현재 config 가중치로 baseline 실행 → 리포트 저장
Stage 2: Optuna TPE sampler로 최적 파라미터 탐색 → 리포트 저장

Usage:
    pyenv shell ptj_stock_lab && python optimizers/optimize_v3_optuna.py --stage 1
    pyenv shell ptj_stock_lab && python optimizers/optimize_v3_optuna.py --stage 2 [--n-trials 20] [--n-jobs 6]
    pyenv shell ptj_stock_lab && python optimizers/optimize_v3_optuna.py              # 1 → 2 연속 실행
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import optuna

import config
from optimizers.optimizer_base import BaseOptimizer, TrialResult


# ============================================================
# V3Optimizer
# ============================================================


class V3Optimizer(BaseOptimizer):
    """v3 Optuna 최적화."""

    version = "v3"

    def __init__(self, gap_max: float = 4.0):
        super().__init__()
        self.gap_max = gap_max

    def get_baseline_params(self) -> dict:
        """현재 config.py에 설정된 v3 파라미터를 반환한다."""
        return {
            # v3 고유
            "V3_PAIR_GAP_ENTRY_THRESHOLD": config.V3_PAIR_GAP_ENTRY_THRESHOLD,
            "V3_DCA_MAX_COUNT": config.V3_DCA_MAX_COUNT,
            "V3_MAX_PER_STOCK_KRW": config.V3_MAX_PER_STOCK_KRW,
            "V3_COIN_TRIGGER_PCT": config.V3_COIN_TRIGGER_PCT,
            "V3_CONL_TRIGGER_PCT": config.V3_CONL_TRIGGER_PCT,
            "V3_SPLIT_BUY_INTERVAL_MIN": config.V3_SPLIT_BUY_INTERVAL_MIN,
            "V3_ENTRY_CUTOFF_HOUR": config.V3_ENTRY_CUTOFF_HOUR,
            "V3_ENTRY_CUTOFF_MINUTE": config.V3_ENTRY_CUTOFF_MINUTE,
            # v3 횡보장
            "V3_SIDEWAYS_MIN_SIGNALS": config.V3_SIDEWAYS_MIN_SIGNALS,
            "V3_SIDEWAYS_POLY_LOW": config.V3_SIDEWAYS_POLY_LOW,
            "V3_SIDEWAYS_POLY_HIGH": config.V3_SIDEWAYS_POLY_HIGH,
            "V3_SIDEWAYS_GLD_THRESHOLD": config.V3_SIDEWAYS_GLD_THRESHOLD,
            "V3_SIDEWAYS_GAP_FAIL_COUNT": config.V3_SIDEWAYS_GAP_FAIL_COUNT,
            "V3_SIDEWAYS_TRIGGER_FAIL_COUNT": config.V3_SIDEWAYS_TRIGGER_FAIL_COUNT,
            "V3_SIDEWAYS_INDEX_THRESHOLD": config.V3_SIDEWAYS_INDEX_THRESHOLD,
            # v2 공유
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
            "INITIAL_BUY_KRW": config.INITIAL_BUY_KRW,
            "DCA_BUY_KRW": config.DCA_BUY_KRW,
        }

    def create_engine(self, params: dict, **kwargs) -> Any:
        """v3 백테스트 엔진 인스턴스를 생성한다."""
        from backtest_v3 import BacktestEngineV3
        return BacktestEngineV3(**kwargs)

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """v3 Optuna 탐색 공간을 정의한다 (Phase 2: 확장된 미세 튜닝)."""
        params = {
            # 핵심 변동 파라미터
            "V3_PAIR_GAP_ENTRY_THRESHOLD": trial.suggest_float(
                "V3_PAIR_GAP_ENTRY_THRESHOLD", 7.0, 10.0, step=0.2
            ),
            "V3_DCA_MAX_COUNT": trial.suggest_int("V3_DCA_MAX_COUNT", 5, 9),
            "V3_MAX_PER_STOCK_KRW": trial.suggest_int(
                "V3_MAX_PER_STOCK_KRW", 6_000_000, 8_000_000, step=500_000
            ),
            "V3_COIN_TRIGGER_PCT": trial.suggest_float("V3_COIN_TRIGGER_PCT", 4.5, 6.5, step=0.5),
            "V3_CONL_TRIGGER_PCT": trial.suggest_float("V3_CONL_TRIGGER_PCT", 5.5, 7.5, step=0.5),
            "V3_SPLIT_BUY_INTERVAL_MIN": trial.suggest_int(
                "V3_SPLIT_BUY_INTERVAL_MIN", 25, 30, step=5
            ),
            "V3_ENTRY_CUTOFF_HOUR": 12,  # 고정
            "V3_ENTRY_CUTOFF_MINUTE": 30,  # 고정
            # v3 횡보장
            "V3_SIDEWAYS_MIN_SIGNALS": trial.suggest_int("V3_SIDEWAYS_MIN_SIGNALS", 2, 3),
            "V3_SIDEWAYS_POLY_LOW": trial.suggest_float("V3_SIDEWAYS_POLY_LOW", 0.30, 0.40, step=0.05),
            "V3_SIDEWAYS_POLY_HIGH": trial.suggest_float("V3_SIDEWAYS_POLY_HIGH", 0.45, 0.55, step=0.05),
            "V3_SIDEWAYS_GLD_THRESHOLD": trial.suggest_float("V3_SIDEWAYS_GLD_THRESHOLD", 0.2, 0.5, step=0.1),
            "V3_SIDEWAYS_INDEX_THRESHOLD": trial.suggest_float("V3_SIDEWAYS_INDEX_THRESHOLD", 0.8, 1.0, step=0.1),
            # v2 공유 (확장 범위)
            "STOP_LOSS_PCT": trial.suggest_float("STOP_LOSS_PCT", -5.5, -2.5, step=0.25),
            "STOP_LOSS_BULLISH_PCT": trial.suggest_float("STOP_LOSS_BULLISH_PCT", -14.0, -9.0, step=0.5),
            "COIN_SELL_PROFIT_PCT": trial.suggest_float("COIN_SELL_PROFIT_PCT", 4.0, 6.0, step=0.5),
            "CONL_SELL_PROFIT_PCT": trial.suggest_float("CONL_SELL_PROFIT_PCT", 2.5, 4.5, step=0.5),
            "DCA_DROP_PCT": trial.suggest_float("DCA_DROP_PCT", -1.0, -0.3, step=0.05),
            "MAX_HOLD_HOURS": trial.suggest_int("MAX_HOLD_HOURS", 2, 6),
            "TAKE_PROFIT_PCT": trial.suggest_float("TAKE_PROFIT_PCT", 3.0, 5.0, step=0.5),
            "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float("PAIR_GAP_SELL_THRESHOLD_V2", 5.0, 10.0, step=0.2),
            "PAIR_SELL_FIRST_PCT": trial.suggest_float("PAIR_SELL_FIRST_PCT", 0.85, 1.0, step=0.05),
        }
        return params

    def run_stage2(
        self,
        n_trials: int = 20,
        n_jobs: int = 6,
        timeout: int | None = None,
        study_name: str | None = None,
        db: str | None = None,
        baseline: dict | None = None,
        baseline_params: dict | None = None,
        **kwargs,
    ) -> None:
        """Stage 2: v3 Optuna 최적화 (subprocess 병렬 지원)."""
        if study_name is None:
            study_name = "ptj_v3_opt"

        # baseline 로드
        if baseline is None or baseline_params is None:
            baseline, baseline_params = self.load_baseline_json()
            print(f"  Baseline 로드: {self._baseline_json}")
            print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")

        print(f"\n{'=' * 70}")
        print(f"  [Stage 2] Optuna 최적화 ({n_trials} trials, {n_jobs} workers, GAP max {self.gap_max}%)")
        print(f"{'=' * 70}")

        from optuna.samplers import TPESampler
        import time

        sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials))
        storage = db if db else None

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        study.enqueue_trial(baseline_params)

        t0 = time.time()

        objective = self._make_objective(**kwargs)

        def _log_progress(study, trial):
            if trial.number % 10 == 0 or trial.number < 5:
                best = study.best_trial
                print(f"  [{trial.number:3d}/{n_trials}] Best: Trial #{best.number} ({best.value:+.2f}%)")

        # v3는 subprocess 기반 병렬화를 사용
        if n_jobs > 1 and storage:
            print(f"  병렬 실행: {n_jobs} 프로세스 수동 실행")
            processes = []
            for i in range(n_jobs):
                if i == 0:
                    continue
                cmd = [
                    sys.executable,
                    "-c",
                    f"""
import optuna
from optimize_v3_optuna import V3Optimizer
opt = V3Optimizer(gap_max={self.gap_max})
study = optuna.load_study(study_name='{study_name}', storage='{storage}')
objective = opt._make_objective()
study.optimize(objective, n_trials={n_trials // n_jobs + 1}, show_progress_bar=False)
"""
                ]
                p = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
                processes.append(p)

            study.optimize(
                objective,
                n_trials=n_trials // n_jobs + 1,
                timeout=timeout,
                show_progress_bar=False,
                callbacks=[_log_progress],
            )

            for p in processes:
                p.wait()
        else:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                callbacks=[_log_progress],
            )

        elapsed = time.time() - t0

        # 콘솔 요약
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best = study.best_trial
            diff = best.value - baseline["total_return_pct"]
            print(f"\n  실행 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
            print(f"  Trial당 평균: {elapsed / len(study.trials):.1f}초")
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
        else:
            print("\n  완료된 trial 없음")
            return

        # 마크다운 리포트
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_optuna_report(study, baseline, baseline_params, elapsed, n_jobs)
        self._optuna_report.write_text(md, encoding="utf-8")
        print(f"\n  Optuna 리포트: {self._optuna_report}")


# ── 워커 함수 (mp.Pool 호환) ──────────────────────────────────


def _worker(params: dict) -> dict:
    """mp.Pool에서 호출되는 최상위 함수."""
    opt = V3Optimizer()
    result = opt.run_single_trial(params)
    return result.to_dict()


# ── 메인 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PTJ v3 Optuna 최적화")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trial 수 (기본: 20)")
    parser.add_argument("--n-jobs", type=int, default=6, help="병렬 프로세스 수 (기본: 6)")
    parser.add_argument("--timeout", type=int, default=None, help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v3_opt", help="study 이름")
    parser.add_argument("--db", type=str, default=None, help="Optuna DB URL (기본: in-memory)")
    parser.add_argument("--gap-max", type=float, default=4.0,
                        help="V3_PAIR_GAP_ENTRY_THRESHOLD 탐색 상한 (%%, 기본: 4.0)")
    args = parser.parse_args()

    opt = V3Optimizer(gap_max=args.gap_max)

    print("=" * 70)
    print("  PTJ v3 — Optuna 파라미터 최적화")
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
        )

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

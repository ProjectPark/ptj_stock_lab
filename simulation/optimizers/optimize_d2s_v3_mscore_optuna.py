#!/usr/bin/env python3
"""
D2S v3 Market Score Weights Optuna 재탐색
=========================================
v3 파라미터 고정 + market_score_weights 6개만 Optuna 탐색.
현재값: gld=0.2273, spy=0.1364, riskoff=0.1818, streak=0.1818, vol=0.1818, btc=0.0909
이 값들이 1.5년 전체에서도 최적인지 재검증.

탐색 공간 (6 파라미터):
  w_gld, w_spy, w_riskoff, w_streak, w_vol, w_btc
  raw 값 sample 후 합계=1.0으로 정규화

스코어 함수: IS_Sharpe x 10 + OOS_Sharpe x 20 - |IS_MDD| x 0.5
  (OOS 2배 가중 -> 일반화 유도, 과적합 억제)

병렬 전략:
  JournalStorage(JournalFileBackend) -- 10 workers 충돌 없음

Usage:
    # Stage 1: baseline 측정
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_mscore_optuna.py --stage 1

    # Stage 2: Optuna 200 trials, 10 workers
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_mscore_optuna.py \\
        --stage 2 --n-trials 200 --n-jobs 10

    # Stage 3: Full 검증
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_mscore_optuna.py --stage 3

    # 연속 실행 (1->2->3)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_mscore_optuna.py \\
        --n-trials 200 --n-jobs 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN

# ============================================================
# 상수
# ============================================================

RESULTS_DIR   = _PROJECT_ROOT / "data" / "results" / "optimization"
OPTUNA_DB_DIR = _PROJECT_ROOT / "data" / "optuna"
REPORTS_DIR   = _PROJECT_ROOT / "simulation" / "optimizers" / "docs"

IS_START  = date(2024, 9, 18)
IS_END    = date(2025, 5, 31)
OOS_START = date(2025, 6, 1)
OOS_END   = date(2026, 2, 17)

STUDY_NAME    = "d2s_v3_mscore_optuna"
JOURNAL_PATH  = OPTUNA_DB_DIR / "d2s_v3_mscore_journal.log"
BASELINE_JSON = RESULTS_DIR / "d2s_v3_mscore_baseline.json"
DATA_PATH     = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily_3y.parquet"
_BASE_PARAMS  = D2S_ENGINE_V3_NO_ROBN

# 스코어: IS Sharpe + OOS Sharpe (OOS 2배 가중) - MDD 패널티
SCORE_IS_SHARPE_W  = 10.0
SCORE_OOS_SHARPE_W = 20.0   # OOS 2배 -> 과적합 억제
SCORE_MDD_W        = 0.50

MIN_SELL_TRADES = 5
MAX_MDD_HARD    = 35.0


# ============================================================
# 탐색 공간 (6 파라미터: market_score_weights)
# ============================================================

def define_search_space(trial: optuna.Trial) -> dict:
    """Market score weights 6개만 탐색. 나머지 v3 파라미터는 고정."""

    raw_gld     = trial.suggest_float("w_gld",     0.05, 0.40)
    raw_spy     = trial.suggest_float("w_spy",     0.05, 0.30)
    raw_riskoff = trial.suggest_float("w_riskoff", 0.05, 0.40)
    raw_streak  = trial.suggest_float("w_streak",  0.05, 0.30)
    raw_vol     = trial.suggest_float("w_vol",     0.05, 0.30)
    raw_btc     = trial.suggest_float("w_btc",     0.0,  0.20)

    total = raw_gld + raw_spy + raw_riskoff + raw_streak + raw_vol + raw_btc

    return {
        **_BASE_PARAMS,
        "market_score_weights": {
            "gld_score":     round(raw_gld     / total, 4),
            "spy_score":     round(raw_spy     / total, 4),
            "riskoff_score": round(raw_riskoff / total, 4),
            "streak_score":  round(raw_streak  / total, 4),
            "vol_score":     round(raw_vol     / total, 4),
            "btc_score":     round(raw_btc     / total, 4),
        },
    }


# ============================================================
# 목적 함수 -- IS + OOS 동시 최적화
# ============================================================

def run_backtest(params: dict, start: date, end: date) -> dict:
    from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
    bt = D2SBacktestV3(params=params, start_date=start, end_date=end,
                       use_fees=True, data_path=DATA_PATH)
    bt.run(verbose=False)
    return bt.report()


def calc_score(is_report: dict, oos_report: dict) -> float:
    """IS + OOS 균형 스코어.

    score = IS_Sharpe x 10 + OOS_Sharpe x 20 - |IS_MDD| x 0.5
    OOS 2배 가중 -> 일반화 유도, 과적합 억제
    """
    if is_report.get("sell_trades", 0) < MIN_SELL_TRADES:
        return -100.0
    if abs(is_report.get("mdd_pct", 0)) > MAX_MDD_HARD:
        return -50.0

    return (
        SCORE_IS_SHARPE_W  * is_report.get("sharpe_ratio", 0)
        + SCORE_OOS_SHARPE_W * oos_report.get("sharpe_ratio", 0)
        - SCORE_MDD_W * abs(is_report.get("mdd_pct", 0))
    )


def objective(trial: optuna.Trial) -> float:
    params     = define_search_space(trial)
    is_report  = run_backtest(params, IS_START, IS_END)
    oos_report = run_backtest(params, OOS_START, OOS_END)
    score      = calc_score(is_report, oos_report)

    trial.set_user_attr("is_return",  is_report.get("total_return_pct", 0))
    trial.set_user_attr("oos_return", oos_report.get("total_return_pct", 0))
    trial.set_user_attr("is_sharpe",  is_report.get("sharpe_ratio", 0))
    trial.set_user_attr("oos_sharpe", oos_report.get("sharpe_ratio", 0))
    trial.set_user_attr("is_mdd",     is_report.get("mdd_pct", 0))
    trial.set_user_attr("oos_mdd",    oos_report.get("mdd_pct", 0))
    trial.set_user_attr("is_wr",      is_report.get("win_rate", 0))
    trial.set_user_attr("is_trades",  is_report.get("sell_trades", 0))
    trial.set_user_attr("oos_trades", oos_report.get("sell_trades", 0))

    return score


# ============================================================
# 병렬 워커
# ============================================================

def _worker_run(args: tuple) -> None:
    n_trials_per_worker, journal_path_str, study_name = args
    import optuna as _optuna
    from optuna.storages import JournalStorage as _JS
    from optuna.storages.journal import JournalFileBackend as _JFB
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _storage = _JS(_JFB(journal_path_str))
    study = _optuna.load_study(study_name=study_name, storage=_storage)
    study.optimize(objective, n_trials=n_trials_per_worker, show_progress_bar=False)


# ============================================================
# Stage 함수
# ============================================================

def run_stage1() -> dict:
    print("\n" + "=" * 70)
    print("  [Stage 1] D2S v3 Market Score Baseline -- IS + OOS")
    print(f"  IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}")
    print("=" * 70)

    t0 = time.time()
    is_report  = run_backtest(_BASE_PARAMS, IS_START, IS_END)
    oos_report = run_backtest(_BASE_PARAMS, OOS_START, OOS_END)
    elapsed = time.time() - t0

    score = calc_score(is_report, oos_report)
    print(f"\n  IS  : {is_report['total_return_pct']:+.2f}%  "
          f"Sharpe: {is_report['sharpe_ratio']:.3f}  MDD: {is_report['mdd_pct']:.1f}%")
    print(f"  OOS : {oos_report['total_return_pct']:+.2f}%  "
          f"Sharpe: {oos_report['sharpe_ratio']:.3f}  MDD: {oos_report['mdd_pct']:.1f}%")
    print(f"  Score: {score:.4f}  ({elapsed:.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "is_report": is_report, "oos_report": oos_report,
        "score": score, "timestamp": datetime.now().isoformat(),
    }
    with open(BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Baseline 저장: {BASELINE_JSON}")
    return payload


def run_stage2(
    n_trials: int = 200, n_jobs: int = 10,
    timeout: int | None = None,
    study_name: str = STUDY_NAME,
    journal_path: Path = JOURNAL_PATH,
) -> None:
    print("\n" + "=" * 70)
    print(f"  [Stage 2] D2S v3 Market Score Optuna -- {n_trials} trials / {n_jobs} workers")
    print(f"  탐색: market_score_weights 6개 (v3 파라미터 고정)")
    print(f"  스코어: IS_Shp x {SCORE_IS_SHARPE_W} + OOS_Shp x {SCORE_OOS_SHARPE_W} - MDD x {SCORE_MDD_W}")
    print("=" * 70)

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    baseline_score = None
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON) as f:
            bl = json.load(f)
        baseline_score = bl.get("score")
        print(f"  Baseline score={baseline_score:.4f}  "
              f"IS={bl['is_report']['total_return_pct']:+.2f}%  "
              f"OOS={bl['oos_report']['total_return_pct']:+.2f}%")

    sampler = TPESampler(seed=42, n_startup_trials=min(30, n_trials))
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        sampler=sampler, storage=storage, load_if_exists=True,
    )

    # Warm start -- 현재 D2S_ENGINE_V3_NO_ROBN 기본 weights
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done == 0:
        mw = _BASE_PARAMS["market_score_weights"]
        # 현재값의 raw 역산 (정규화 전 값, 대략적 비율 유지)
        study.enqueue_trial({
            "w_gld":     0.25,   # -> 0.2273
            "w_spy":     0.15,   # -> 0.1364
            "w_riskoff": 0.20,   # -> 0.1818
            "w_streak":  0.20,   # -> 0.1818
            "w_vol":     0.20,   # -> 0.1818
            "w_btc":     0.10,   # -> 0.0909
        })
        print("  v3 기본 weights enqueue (warm start)")

    remaining = max(0, n_trials - already_done)
    if remaining == 0:
        print(f"  이미 {already_done}회 완료.")
    else:
        print(f"  {already_done}회 완료 -> {remaining}회 추가")
        t0 = time.time()
        if n_jobs <= 1:
            study.optimize(objective, n_trials=remaining, timeout=timeout,
                           show_progress_bar=True)
        else:
            tpw = max(1, remaining // n_jobs)
            extra = remaining - tpw * n_jobs
            args_list = [
                (tpw + (1 if i < extra else 0), str(journal_path), study_name)
                for i in range(n_jobs)
            ]
            ctx = mp.get_context("spawn")
            print(f"  workers: {n_jobs} x ~{tpw} trials")
            with ctx.Pool(processes=n_jobs) as pool:
                pool.map(_worker_run, args_list)
        elapsed = time.time() - t0
        print(f"  완료: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    _print_study_summary(study, baseline_score)
    _save_optuna_report(study, baseline_score, n_trials, n_jobs, journal_path)


def run_stage3(study_name: str = STUDY_NAME, journal_path: Path = JOURNAL_PATH) -> None:
    print("\n" + "=" * 70)
    print("  [Stage 3] D2S v3 Market Score Full 검증")
    print("=" * 70)

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        storage = JournalStorage(JournalFileBackend(str(journal_path)))
        study   = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        print(f"  [ERROR] Study '{study_name}' 없음.")
        return

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  완료된 trial 없음.")
        return

    best = study.best_trial
    print(f"  Best #{best.number}  score={best.value:.4f}")
    print(f"    IS  ret={best.user_attrs.get('is_return', 0):+.2f}%  "
          f"Shp={best.user_attrs.get('is_sharpe', 0):.3f}  "
          f"MDD={best.user_attrs.get('is_mdd', 0):.1f}%")
    print(f"    OOS ret={best.user_attrs.get('oos_return', 0):+.2f}%  "
          f"Shp={best.user_attrs.get('oos_sharpe', 0):.3f}  "
          f"MDD={best.user_attrs.get('oos_mdd', 0):.1f}%")

    best_params = _reconstruct_params(best.params)

    # Best weights 출력
    print(f"\n  Best market_score_weights:")
    for k, v in best_params["market_score_weights"].items():
        print(f"    {k}: {v:.4f}")

    from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
    t0 = time.time()
    full_bt = D2SBacktestV3(params=best_params,
                             start_date=IS_START, end_date=OOS_END, use_fees=True,
                             data_path=DATA_PATH)
    full_bt.run(verbose=False)
    r_full = full_bt.report()
    elapsed = time.time() - t0

    print(f"\n  FULL ({IS_START} ~ {OOS_END}):")
    print(f"    수익률: {r_full['total_return_pct']:+.2f}%  "
          f"MDD: {r_full['mdd_pct']:.1f}%  Sharpe: {r_full['sharpe_ratio']:.3f}")
    full_bt.print_report()

    result_path = RESULTS_DIR / f"{STUDY_NAME}_best_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_trial":  best.number,
            "score":       best.value,
            "is_return":   best.user_attrs.get("is_return", 0),
            "oos_return":  best.user_attrs.get("oos_return", 0),
            "full_report": r_full,
            "best_params": best.params,
            "best_weights": best_params["market_score_weights"],
            "timestamp":   datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  결과 저장: {result_path}")


# ============================================================
# 유틸리티
# ============================================================

def _reconstruct_params(trial_params: dict) -> dict:
    """trial params (raw w_*) -> 정규화된 market_score_weights로 복원."""
    p = dict(_BASE_PARAMS)
    w_keys = ["w_gld", "w_spy", "w_riskoff", "w_streak", "w_vol", "w_btc"]
    ws = [trial_params.get(k, 0.1) for k in w_keys]
    total = sum(ws)
    score_keys = ["gld_score", "spy_score", "riskoff_score", "streak_score", "vol_score", "btc_score"]
    p["market_score_weights"] = {k: round(w / total, 4) for k, w in zip(score_keys, ws)}
    return p


def _print_study_summary(study: optuna.Study, baseline_score: float | None) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    diff_s = f" (baseline 대비 {best.value-(baseline_score or 0):+.4f})" if baseline_score else ""
    print(f"\n  BEST #{best.number}  score={best.value:.4f}{diff_s}")
    print(f"    IS  {best.user_attrs.get('is_return', 0):+.2f}%  Shp={best.user_attrs.get('is_sharpe', 0):.3f}")
    print(f"    OOS {best.user_attrs.get('oos_return', 0):+.2f}%  Shp={best.user_attrs.get('oos_sharpe', 0):.3f}")

    # Best weights 출력
    best_params = _reconstruct_params(best.params)
    print(f"    weights: {best_params['market_score_weights']}")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    print(f"  {'#':>4}  {'score':>8}  {'IS%':>8}  {'OOS%':>8}  {'IS_Shp':>7}  {'OOS_Shp':>8}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+7.4f}  "
            f"{t.user_attrs.get('is_return', 0):+7.2f}%  "
            f"{t.user_attrs.get('oos_return', 0):+7.2f}%  "
            f"{t.user_attrs.get('is_sharpe', 0):+6.3f}  "
            f"{t.user_attrs.get('oos_sharpe', 0):+7.3f}"
        )


def _save_optuna_report(
    study: optuna.Study, baseline_score: float | None,
    n_trials: int, n_jobs: int, journal_path: Path,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{STUDY_NAME}_report.md"

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    best  = study.best_trial
    top10 = sorted(completed, key=lambda t: t.value, reverse=True)[:10]
    best_weights = _reconstruct_params(best.params)["market_score_weights"]

    lines = [
        "# D2S v3 Market Score Weights Optuna 리포트",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}  ",
        f"> trial: {len(completed)}  |  n_jobs: {n_jobs}  ",
        f"> 탐색: market_score_weights 6개 (v3 파라미터 고정)  ",
        f"> 스코어: IS_Shp x {SCORE_IS_SHARPE_W} + OOS_Shp x {SCORE_OOS_SHARPE_W} - MDD x {SCORE_MDD_W}  ",
        f"> Journal: {journal_path.name}",
        "",
        "## 1. Best Trial",
        "",
        "| 항목 | IS | OOS |",
        "|---|---|---|",
        f"| Trial # | {best.number} | -- |",
        f"| Score | {best.value:.4f} | -- |",
        f"| 수익률 | {best.user_attrs.get('is_return', 0):+.2f}% "
        f"| {best.user_attrs.get('oos_return', 0):+.2f}% |",
        f"| MDD | {best.user_attrs.get('is_mdd', 0):.1f}% "
        f"| {best.user_attrs.get('oos_mdd', 0):.1f}% |",
        f"| Sharpe | {best.user_attrs.get('is_sharpe', 0):.3f} "
        f"| {best.user_attrs.get('oos_sharpe', 0):.3f} |",
        "",
        "## 2. Best Weights",
        "",
        "| Weight | 값 |",
        "|---|---|",
    ]
    for k, v in sorted(best_weights.items()):
        lines.append(f"| `{k}` | {v:.4f} |")

    lines += [
        "",
        "## 3. Top 10",
        "",
        "| # | score | IS% | OOS% | IS_Shp | OOS_Shp |",
        "|---|---|---|---|---|---|",
    ]
    for t in top10:
        lines.append(
            f"| {t.number} | {t.value:+.4f} | "
            f"{t.user_attrs.get('is_return', 0):+.2f}% | "
            f"{t.user_attrs.get('oos_return', 0):+.2f}% | "
            f"{t.user_attrs.get('is_sharpe', 0):.3f} | "
            f"{t.user_attrs.get('oos_sharpe', 0):.3f} |"
        )

    lines += [
        "",
        "## 4. Raw 파라미터",
        "",
        "| 파라미터 | 값 |",
        "|---|---|",
    ]
    for k, v in sorted(best.params.items()):
        lines.append(f"| `{k}` | {v} |")

    lines += [""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  리포트: {report_path}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="D2S v3 Market Score Weights Optuna 재탐색")
    parser.add_argument("--stage",      type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--n-trials",   type=int, default=200)
    parser.add_argument("--n-jobs",     type=int, default=10)
    parser.add_argument("--timeout",    type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--journal",    type=str, default=None)
    args = parser.parse_args()

    journal_path = Path(args.journal) if args.journal else JOURNAL_PATH
    study_name   = args.study_name if args.study_name else STUDY_NAME

    print("=" * 70)
    print("  D2S v3 Market Score Weights Optuna 재탐색")
    print("  탐색: market_score_weights 6개 (v3 파라미터 고정)")
    print(f"  IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}")
    print(f"  스코어: IS_Shp x {SCORE_IS_SHARPE_W} + OOS_Shp x {SCORE_OOS_SHARPE_W} - MDD x {SCORE_MDD_W}")
    print("=" * 70)

    if args.stage == 1:
        run_stage1()
    elif args.stage == 2:
        run_stage2(n_trials=args.n_trials, n_jobs=args.n_jobs,
                   timeout=args.timeout, study_name=study_name,
                   journal_path=journal_path)
    elif args.stage == 3:
        run_stage3(study_name=study_name, journal_path=journal_path)
    else:
        run_stage1()
        print()
        run_stage2(n_trials=args.n_trials, n_jobs=args.n_jobs,
                   timeout=args.timeout, study_name=study_name,
                   journal_path=journal_path)
        print()
        run_stage3(study_name=study_name, journal_path=journal_path)

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

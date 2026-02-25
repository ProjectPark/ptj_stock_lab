#!/usr/bin/env python3
"""
D2S Optuna v3 — Walk-Forward + Extended IS 최적화
===================================================
3년치 데이터(market_daily_3y.parquet)를 활용한 두 가지 검증 모드:

  --mode wf       : Walk-Forward 검증 (3개 창)
                    W1: IS=2025-03-03~2025-06-30  / OOS=2025-07-01~2025-09-30
                    W2: IS=2025-03-03~2025-09-30  / OOS=2025-10-01~2025-12-31
                    W3: IS=2025-03-03~2025-12-31  / OOS=2026-01-01~2026-02-17

  --mode ext_is   : Extended IS (Option B)
                    IS=2025-03-03~2025-12-31 / OOS=2026-01-01~2026-02-17

각 창(window)마다 독립 Optuna study + JournalStorage 사용.
병렬: 20 workers × mp.Pool (v2와 동일한 구조).

Usage:
    # Walk-Forward (3 windows, 200 trials each)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_optuna.py \\
        --mode wf --n-trials 200 --n-jobs 20

    # Extended IS
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_optuna.py \\
        --mode ext_is --n-trials 200 --n-jobs 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE

# ============================================================
# 데이터 경로
# ============================================================
DATA_PATH_3Y = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily_3y.parquet"

# ============================================================
# Walk-Forward 창 정의
# ============================================================
WF_WINDOWS = [
    # (window_id, is_start, is_end, oos_start, oos_end)
    ("W1", date(2025, 3, 3),  date(2025, 6, 30), date(2025, 7,  1), date(2025, 9, 30)),
    ("W2", date(2025, 3, 3),  date(2025, 9, 30), date(2025, 10, 1), date(2025, 12, 31)),
    ("W3", date(2025, 3, 3),  date(2025, 12, 31), date(2026, 1,  1), date(2026, 2, 17)),
]

# Extended IS (Option B)
EXT_IS_START  = date(2025, 3, 3)
EXT_IS_END    = date(2025, 12, 31)
EXT_OOS_START = date(2026, 1, 1)
EXT_OOS_END   = date(2026, 2, 17)

# ============================================================
# 결과 저장 경로
# ============================================================
RESULTS_DIR   = _PROJECT_ROOT / "data" / "results" / "optimization"
OPTUNA_DB_DIR = _PROJECT_ROOT / "data" / "optuna"

# ============================================================
# 스코어 함수 (v2와 동일)
# ============================================================
SCORE_WIN_RATE_W = 0.40
SCORE_SHARPE_W   = 15.0
SCORE_MDD_W      = 0.30
MIN_SELL_TRADES  = 5
MAX_MDD_HARD     = 30.0


# ============================================================
# 공통 유틸
# ============================================================

def run_backtest(params: dict, start: date, end: date) -> dict:
    """D2SBacktest를 market_daily_3y.parquet으로 실행."""
    from simulation.backtests.backtest_d2s import D2SBacktest
    bt = D2SBacktest(
        params=params,
        start_date=start,
        end_date=end,
        use_fees=True,
        data_path=DATA_PATH_3Y,
    )
    bt.run(verbose=False)
    return bt.report()


def calc_score(report: dict) -> float:
    """IS 스코어 계산 (v2 동일)."""
    n_sells = report.get("sell_trades", 0)
    if n_sells < MIN_SELL_TRADES:
        return -100.0
    mdd = abs(report.get("mdd_pct", 0))
    if mdd > MAX_MDD_HARD:
        return -50.0
    win_rate     = report.get("win_rate", 0)
    sharpe_ratio = report.get("sharpe_ratio", 0)
    return SCORE_WIN_RATE_W * win_rate + SCORE_SHARPE_W * sharpe_ratio - SCORE_MDD_W * mdd


# ============================================================
# Optuna 탐색 공간 (v2와 동일)
# ============================================================

def define_search_space(trial: optuna.Trial) -> dict:
    suppress   = trial.suggest_float("market_score_suppress", 0.15, 0.45, step=0.05)
    entry_b_low = round(suppress + 0.05, 2)
    entry_b    = trial.suggest_float("market_score_entry_b", entry_b_low, 0.70, step=0.05)
    entry_a_low = round(entry_b + 0.05, 2)
    entry_a    = trial.suggest_float("market_score_entry_a", entry_a_low, 0.80, step=0.05)

    w_gld     = trial.suggest_float("w_gld",     0.05, 0.35, step=0.05)
    w_spy     = trial.suggest_float("w_spy",     0.05, 0.30, step=0.05)
    w_riskoff = trial.suggest_float("w_riskoff", 0.05, 0.30, step=0.05)
    w_streak  = trial.suggest_float("w_streak",  0.05, 0.30, step=0.05)
    w_vol     = trial.suggest_float("w_vol",     0.05, 0.30, step=0.05)
    w_btc     = trial.suggest_float("w_btc",     0.00, 0.15, step=0.05)
    w_total   = w_gld + w_spy + w_riskoff + w_streak + w_vol + w_btc
    market_score_weights = {
        "gld_score":     round(w_gld     / w_total, 4),
        "spy_score":     round(w_spy     / w_total, 4),
        "riskoff_score": round(w_riskoff / w_total, 4),
        "streak_score":  round(w_streak  / w_total, 4),
        "vol_score":     round(w_vol     / w_total, 4),
        "btc_score":     round(w_btc     / w_total, 4),
    }

    spy_min_th   = trial.suggest_float("riskoff_spy_min_threshold", -3.0, -0.8, step=0.2)
    gld_opt_min  = trial.suggest_float("riskoff_gld_optimal_min",    0.2,  1.2, step=0.1)
    spy_opt_max  = trial.suggest_float("riskoff_spy_optimal_max",   -1.0, -0.2, step=0.1)
    consec_boost = trial.suggest_int("riskoff_consecutive_boost",   2, 5)
    panic_factor = trial.suggest_float("riskoff_panic_size_factor",  0.3, 0.7, step=0.1)

    gld_suppress   = trial.suggest_float("gld_suppress_threshold", 0.5, 2.0, step=0.25)
    btc_up_max     = trial.suggest_float("btc_up_max",             0.60, 0.90, step=0.05)
    btc_up_min     = trial.suggest_float("btc_up_min",             0.20, 0.55, step=0.05)
    spy_streak_max = trial.suggest_int("spy_streak_max",           2, 5)
    spy_bearish_th = trial.suggest_float("spy_bearish_threshold",  -2.0, -0.5, step=0.25)

    rsi_entry_min = trial.suggest_int("rsi_entry_min",   25, 55)
    rsi_entry_max = trial.suggest_int("rsi_entry_max",   50, 80)
    rsi_danger    = trial.suggest_int("rsi_danger_zone", 70, 90)
    bb_entry_max  = trial.suggest_float("bb_entry_max",   0.3, 1.0, step=0.1)
    bb_danger     = trial.suggest_float("bb_danger_zone", 0.9, 1.2, step=0.1)
    atr_quantile  = trial.suggest_float("atr_high_quantile", 0.50, 0.85, step=0.05)
    vol_entry_min = trial.suggest_float("vol_entry_min",  0.5, 1.5, step=0.1)
    vol_entry_max = trial.suggest_float("vol_entry_max",  1.5, 5.0, step=0.5)

    contrarian_th = trial.suggest_float("contrarian_entry_threshold",       -2.0, 0.5, step=0.5)
    amdl_th       = trial.suggest_float("amdl_friday_contrarian_threshold", -3.0, -0.5, step=0.5)

    take_profit   = trial.suggest_float("take_profit_pct",       3.0, 10.0, step=0.5)
    hold_days_max = trial.suggest_int("optimal_hold_days_max",   4, 14)
    dca_max       = trial.suggest_int("dca_max_daily",           2, 8)

    buy_large = trial.suggest_float("buy_size_large",      0.10, 0.25, step=0.05)
    buy_small = trial.suggest_float("buy_size_small",      0.03, 0.10, step=0.01)
    entry_cap = trial.suggest_float("daily_new_entry_cap", 0.15, 0.50, step=0.05)

    return {
        **D2S_ENGINE,
        "market_score_suppress":  suppress,
        "market_score_entry_b":   entry_b,
        "market_score_entry_a":   entry_a,
        "market_score_weights":   market_score_weights,
        "riskoff_spy_min_threshold":   spy_min_th,
        "riskoff_gld_optimal_min":     gld_opt_min,
        "riskoff_spy_optimal_max":     spy_opt_max,
        "riskoff_consecutive_boost":   consec_boost,
        "riskoff_panic_size_factor":   panic_factor,
        "gld_suppress_threshold":  gld_suppress,
        "btc_up_max":              btc_up_max,
        "btc_up_min":              btc_up_min,
        "spy_streak_max":          spy_streak_max,
        "spy_bearish_threshold":   spy_bearish_th,
        "rsi_entry_min":     rsi_entry_min,
        "rsi_entry_max":     rsi_entry_max,
        "rsi_danger_zone":   rsi_danger,
        "bb_entry_max":      bb_entry_max,
        "bb_danger_zone":    bb_danger,
        "atr_high_quantile": atr_quantile,
        "vol_entry_min":     vol_entry_min,
        "vol_entry_max":     vol_entry_max,
        "contrarian_entry_threshold":       contrarian_th,
        "amdl_friday_contrarian_threshold": amdl_th,
        "take_profit_pct":        take_profit,
        "optimal_hold_days_max":  hold_days_max,
        "dca_max_daily":          dca_max,
        "buy_size_large":         buy_large,
        "buy_size_small":         buy_small,
        "daily_new_entry_cap":    entry_cap,
    }


# ============================================================
# 목적함수 팩토리 (IS 구간을 클로저로 캡처)
# ============================================================

def make_objective(is_start: date, is_end: date):
    """is_start/is_end를 캡처한 objective 함수를 반환."""
    def objective(trial: optuna.Trial) -> float:
        params = define_search_space(trial)
        report = run_backtest(params, is_start, is_end)
        score  = calc_score(report)
        trial.set_user_attr("win_rate",         report.get("win_rate", 0))
        trial.set_user_attr("total_return_pct", report.get("total_return_pct", 0))
        trial.set_user_attr("mdd_pct",          report.get("mdd_pct", 0))
        trial.set_user_attr("sharpe_ratio",     report.get("sharpe_ratio", 0))
        trial.set_user_attr("sell_trades",      report.get("sell_trades", 0))
        trial.set_user_attr("buy_trades",       report.get("buy_trades", 0))
        return score
    return objective


# ============================================================
# 병렬 워커 (mp.Pool 호환 — 최상위 함수)
# ============================================================

# 전역 변수: 워커 내부에서 참조 (spawn safe: pickle 불가한 클로저 대신)
_WORKER_IS_START: date | None = None
_WORKER_IS_END:   date | None = None


def _worker_run(args: tuple) -> None:
    """mp.Pool 워커. (n_trials, journal_path_str, study_name, is_start_str, is_end_str)"""
    n_trials_per_worker, journal_path_str, study_name, is_start_str, is_end_str = args

    import optuna as _optuna
    from optuna.storages import JournalStorage as _JS
    from optuna.storages.journal import JournalFileBackend as _JFB
    from datetime import date as _date
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    _is_start = _date.fromisoformat(is_start_str)
    _is_end   = _date.fromisoformat(is_end_str)
    _obj = make_objective(_is_start, _is_end)

    _storage = _JS(_JFB(journal_path_str))
    study = _optuna.load_study(study_name=study_name, storage=_storage)
    study.optimize(_obj, n_trials=n_trials_per_worker, show_progress_bar=False)


# ============================================================
# 창 단위 최적화 실행
# ============================================================

def _optimize_window(
    win_id: str,
    is_start: date,
    is_end: date,
    oos_start: date,
    oos_end: date,
    n_trials: int,
    n_jobs: int,
) -> dict:
    """하나의 WF 창 또는 Extended IS 최적화 + OOS 검증."""
    study_name   = f"d2s_v3_{win_id}"
    journal_path = OPTUNA_DB_DIR / f"d2s_v3_{win_id}.log"
    result_path  = RESULTS_DIR / f"d2s_v3_{win_id}_result.json"

    print(f"\n{'=' * 70}")
    print(f"  [{win_id}] IS: {is_start} ~ {is_end}  |  OOS: {oos_start} ~ {oos_end}")
    print(f"  Trials: {n_trials} / Workers: {n_jobs}")
    print(f"  Journal: {journal_path.name}")
    print("=" * 70)

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # JournalStorage 생성
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    # Sampler
    sampler = TPESampler(seed=42, n_startup_trials=min(20, n_trials))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Warm start (baseline 파라미터)
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done == 0:
        study.enqueue_trial({
            "market_score_suppress":           D2S_ENGINE["market_score_suppress"],
            "market_score_entry_b":            D2S_ENGINE["market_score_entry_b"],
            "market_score_entry_a":            D2S_ENGINE["market_score_entry_a"],
            "w_gld":     D2S_ENGINE["market_score_weights"]["gld_score"],
            "w_spy":     D2S_ENGINE["market_score_weights"]["spy_score"],
            "w_riskoff": D2S_ENGINE["market_score_weights"]["riskoff_score"],
            "w_streak":  D2S_ENGINE["market_score_weights"]["streak_score"],
            "w_vol":     D2S_ENGINE["market_score_weights"]["vol_score"],
            "w_btc":     D2S_ENGINE["market_score_weights"]["btc_score"],
            "riskoff_spy_min_threshold":   -1.6,
            "riskoff_gld_optimal_min":     D2S_ENGINE["riskoff_gld_optimal_min"],
            "riskoff_spy_optimal_max":     D2S_ENGINE["riskoff_spy_optimal_max"],
            "riskoff_consecutive_boost":   D2S_ENGINE["riskoff_consecutive_boost"],
            "riskoff_panic_size_factor":   D2S_ENGINE["riskoff_panic_size_factor"],
            "gld_suppress_threshold":      D2S_ENGINE["gld_suppress_threshold"],
            "btc_up_max":                  D2S_ENGINE["btc_up_max"],
            "btc_up_min":                  D2S_ENGINE["btc_up_min"],
            "spy_streak_max":              D2S_ENGINE["spy_streak_max"],
            "spy_bearish_threshold":       D2S_ENGINE["spy_bearish_threshold"],
            "rsi_entry_min":               D2S_ENGINE["rsi_entry_min"],
            "rsi_entry_max":               D2S_ENGINE["rsi_entry_max"],
            "rsi_danger_zone":             D2S_ENGINE["rsi_danger_zone"],
            "bb_entry_max":                D2S_ENGINE["bb_entry_max"],
            "bb_danger_zone":              D2S_ENGINE["bb_danger_zone"],
            "atr_high_quantile":           D2S_ENGINE["atr_high_quantile"],
            "vol_entry_min":               D2S_ENGINE["vol_entry_min"],
            "vol_entry_max":               D2S_ENGINE["vol_entry_max"],
            "contrarian_entry_threshold":            D2S_ENGINE["contrarian_entry_threshold"],
            "amdl_friday_contrarian_threshold":      D2S_ENGINE["amdl_friday_contrarian_threshold"],
            "take_profit_pct":        6.0,
            "optimal_hold_days_max":  D2S_ENGINE["optimal_hold_days_max"],
            "dca_max_daily":          D2S_ENGINE["dca_max_daily"],
            "buy_size_large":         D2S_ENGINE["buy_size_large"],
            "buy_size_small":         D2S_ENGINE["buy_size_small"],
            "daily_new_entry_cap":    D2S_ENGINE["daily_new_entry_cap"],
        })

    # 최적화 실행
    remaining = max(0, n_trials - already_done)
    if remaining > 0:
        t0 = time.time()
        if n_jobs <= 1:
            obj = make_objective(is_start, is_end)
            study.optimize(obj, n_trials=remaining, show_progress_bar=True)
        else:
            trials_per_worker = max(1, remaining // n_jobs)
            extra = remaining - trials_per_worker * n_jobs
            worker_args = []
            for i in range(n_jobs):
                n = trials_per_worker + (1 if i < extra else 0)
                worker_args.append((n, str(journal_path), study_name,
                                    is_start.isoformat(), is_end.isoformat()))
            ctx = mp.get_context("spawn")
            print(f"  workers: {n_jobs} × ~{trials_per_worker} trials/worker")
            with ctx.Pool(processes=n_jobs) as pool:
                pool.map(_worker_run, worker_args)
        print(f"  완료: {time.time() - t0:.1f}초")
    else:
        print(f"  이미 {already_done}회 완료. OOS 검증으로 진행합니다.")

    # 결과 수집
    best = study.best_trial
    best_params = define_search_space(best)  # 재구성 (저장용)
    # 실제로는 best.params에서 재구성
    best_full_params = {
        **D2S_ENGINE,
        **{k: v for k, v in best.params.items() if k in D2S_ENGINE},
    }

    print(f"\n  Best Trial #{best.number}  IS score={best.value:.4f}")
    print(f"    win_rate={best.user_attrs.get('win_rate', 0):.1f}%  "
          f"return={best.user_attrs.get('total_return_pct', 0):+.2f}%  "
          f"MDD={best.user_attrs.get('mdd_pct', 0):.2f}%  "
          f"sharpe={best.user_attrs.get('sharpe_ratio', 0):.3f}")

    # OOS 검증
    print(f"\n  OOS 검증: {oos_start} ~ {oos_end}")

    # best.params를 D2S_ENGINE에 오버레이하여 완전한 파라미터 구성
    best_params_for_run = dict(D2S_ENGINE)

    # market_score_weights 재구성
    w = best.params
    w_gld = w.get("w_gld", 0.2)
    w_spy = w.get("w_spy", 0.1)
    w_riskoff = w.get("w_riskoff", 0.25)
    w_streak  = w.get("w_streak", 0.15)
    w_vol     = w.get("w_vol", 0.15)
    w_btc     = w.get("w_btc", 0.10)
    w_total   = w_gld + w_spy + w_riskoff + w_streak + w_vol + w_btc
    best_params_for_run["market_score_weights"] = {
        "gld_score":     round(w_gld     / w_total, 4),
        "spy_score":     round(w_spy     / w_total, 4),
        "riskoff_score": round(w_riskoff / w_total, 4),
        "streak_score":  round(w_streak  / w_total, 4),
        "vol_score":     round(w_vol     / w_total, 4),
        "btc_score":     round(w_btc     / w_total, 4),
    }

    # 나머지 best params 오버레이 (w_* 제외)
    for key, val in best.params.items():
        if not key.startswith("w_"):
            best_params_for_run[key] = val

    oos_report = run_backtest(best_params_for_run, oos_start, oos_end)
    oos_score  = calc_score(oos_report)
    print(f"    OOS score={oos_score:.4f}  "
          f"win_rate={oos_report.get('win_rate', 0):.1f}%  "
          f"return={oos_report.get('total_return_pct', 0):+.2f}%  "
          f"MDD={oos_report.get('mdd_pct', 0):.2f}%  "
          f"sharpe={oos_report.get('sharpe_ratio', 0):.3f}")

    # 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "window":      win_id,
        "is_start":    str(is_start),
        "is_end":      str(is_end),
        "oos_start":   str(oos_start),
        "oos_end":     str(oos_end),
        "best_trial":  best.number,
        "is_score":    best.value,
        "is_report": {
            "win_rate":         best.user_attrs.get("win_rate", 0),
            "total_return_pct": best.user_attrs.get("total_return_pct", 0),
            "mdd_pct":          best.user_attrs.get("mdd_pct", 0),
            "sharpe_ratio":     best.user_attrs.get("sharpe_ratio", 0),
            "sell_trades":      best.user_attrs.get("sell_trades", 0),
        },
        "oos_score":   oos_score,
        "oos_report":  oos_report,
        "best_params": {k: v for k, v in best_params_for_run.items()
                        if not isinstance(v, (dict, list))},
        "timestamp":   datetime.now().isoformat(),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  결과 저장: {result_path.name}")

    return result


# ============================================================
# 요약 출력
# ============================================================

def _print_wf_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print("  Walk-Forward 요약")
    print("=" * 70)
    print(f"  {'창':4s}  {'IS 기간':23s}  {'IS wr':>6s}  {'IS ret':>8s}  "
          f"{'OOS wr':>7s}  {'OOS ret':>8s}  {'OOS MDD':>8s}  {'OOS sh':>7s}")
    print(f"  {'-' * 4}  {'-' * 23}  {'-' * 6}  {'-' * 8}  "
          f"{'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 7}")
    for r in results:
        ir = r["is_report"]
        or_ = r["oos_report"]
        print(f"  {r['window']:4s}  "
              f"{r['is_start']}~{r['is_end'][5:]}  "
              f"{ir.get('win_rate', 0):6.1f}%  "
              f"{ir.get('total_return_pct', 0):+8.2f}%  "
              f"{or_.get('win_rate', 0):7.1f}%  "
              f"{or_.get('total_return_pct', 0):+8.2f}%  "
              f"{or_.get('mdd_pct', 0):8.2f}%  "
              f"{or_.get('sharpe_ratio', 0):7.3f}")
    print()


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="D2S Optuna v3")
    parser.add_argument("--mode",     choices=["wf", "ext_is"], default="wf",
                        help="wf=Walk-Forward, ext_is=Extended IS (Option B)")
    parser.add_argument("--window",   type=str, default=None,
                        help="개별 창 실행 (W1, W2, W3). 미지정 시 전체 순차 실행")
    parser.add_argument("--n-trials", type=int, default=600,
                        help="창당 Optuna trial 수")
    parser.add_argument("--n-jobs",   type=int, default=20,
                        help="병렬 워커 수")
    args = parser.parse_args()

    print(f"\nD2S Optuna v3 — mode={args.mode}, window={args.window}, "
          f"trials={args.n_trials}, jobs={args.n_jobs}")
    print(f"데이터: {DATA_PATH_3Y}")

    if not DATA_PATH_3Y.exists():
        print(f"[ERROR] {DATA_PATH_3Y} 없음. fetch_d2s_daily_3y.py를 먼저 실행하세요.")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.mode == "wf":
        # --window 지정 시 해당 창만 실행 (병렬 SLURM 제출용)
        if args.window:
            win_map = {w[0]: w for w in WF_WINDOWS}
            if args.window not in win_map:
                print(f"[ERROR] --window {args.window} 없음. 사용 가능: {list(win_map.keys())}")
                sys.exit(1)
            win_id, is_start, is_end, oos_start, oos_end = win_map[args.window]
            result = _optimize_window(
                win_id, is_start, is_end, oos_start, oos_end,
                n_trials=args.n_trials, n_jobs=args.n_jobs,
            )
            print(f"\n[{win_id} 완료]")
            print(f"  IS:  wr={result['is_report']['win_rate']:.1f}%  "
                  f"ret={result['is_report']['total_return_pct']:+.2f}%  "
                  f"sharpe={result['is_report']['sharpe_ratio']:.3f}")
            print(f"  OOS: wr={result['oos_report'].get('win_rate', 0):.1f}%  "
                  f"ret={result['oos_report'].get('total_return_pct', 0):+.2f}%  "
                  f"MDD={result['oos_report'].get('mdd_pct', 0):.2f}%")
        else:
            # 전체 순차 실행 (레거시)
            results = []
            for win_id, is_start, is_end, oos_start, oos_end in WF_WINDOWS:
                result = _optimize_window(
                    win_id, is_start, is_end, oos_start, oos_end,
                    n_trials=args.n_trials, n_jobs=args.n_jobs,
                )
                results.append(result)
            _print_wf_summary(results)

            summary_path = RESULTS_DIR / "d2s_v3_wf_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"WF 통합 결과 저장: {summary_path.name}")

    else:  # ext_is
        result = _optimize_window(
            "EXT_IS", EXT_IS_START, EXT_IS_END, EXT_OOS_START, EXT_OOS_END,
            n_trials=args.n_trials, n_jobs=args.n_jobs,
        )
        print(f"\n[Extended IS 완료]")
        print(f"  IS:  wr={result['is_report']['win_rate']:.1f}%  "
              f"ret={result['is_report']['total_return_pct']:+.2f}%  "
              f"sharpe={result['is_report']['sharpe_ratio']:.3f}")
        print(f"  OOS: wr={result['oos_report'].get('win_rate', 0):.1f}%  "
              f"ret={result['oos_report'].get('total_return_pct', 0):+.2f}%  "
              f"MDD={result['oos_report'].get('mdd_pct', 0):.2f}%")


if __name__ == "__main__":
    main()

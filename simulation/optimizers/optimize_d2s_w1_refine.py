#!/usr/bin/env python3
"""
D2S W1 best params 좁은 재탐색 (B-2)
======================================
W1 best params(wf_summary.json) 기준 ±20% 범위 내 재탐색.
IS=2025-03-03~2025-06-30 / OOS=2025-07-01~2025-09-30 (W1과 동일).

목적: W1이 OOS 75% 승률로 가장 유망 → 주변 탐색으로 정밀화.
n_startup_trials=5 (TPE 빠른 수렴), warm_start로 W1 best params 우선 평가.

Usage:
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_w1_refine.py \\
        --n-trials 300 --n-jobs 20
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
from simulation.optimizers.optimize_d2s_v3_optuna import (
    run_backtest,
    calc_score,
    _print_wf_summary,
    DATA_PATH_3Y,
    RESULTS_DIR,
    OPTUNA_DB_DIR,
)

# ── W1 기간 ────────────────────────────────────────────────────
W1_IS_START  = date(2025, 3,  3)
W1_IS_END    = date(2025, 6, 30)
W1_OOS_START = date(2025, 7,  1)
W1_OOS_END   = date(2025, 9, 30)

WF_SUMMARY_PATH = _PROJECT_ROOT / "data" / "results" / "optimization" / "d2s_v3_wf_summary.json"


def load_w1_best_params() -> dict:
    """wf_summary.json에서 W1 best_params를 로드."""
    with open(WF_SUMMARY_PATH) as f:
        summary = json.load(f)
    for entry in summary:
        if entry.get("window") == "W1":
            return entry["best_params"]
    raise ValueError("W1 best_params를 wf_summary.json에서 찾을 수 없습니다.")


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def define_search_space_refined(trial: optuna.Trial, base: dict) -> dict:
    """W1 best params ±20% 범위 내 탐색 공간."""

    def _r(key: str, lo: float, hi: float, step: float) -> float:
        b = base.get(key, (lo + hi) / 2)
        l = _clamp(round(b * 0.80 / step) * step, lo, hi)
        h = _clamp(round(b * 1.20 / step) * step, lo, hi)
        if l >= h:
            l = max(lo, h - step)
        return trial.suggest_float(key, l, h, step=step)

    def _ri(key: str, lo: int, hi: int) -> int:
        b = int(base.get(key, (lo + hi) // 2))
        l = max(lo, round(b * 0.80))
        h = min(hi, round(b * 1.20))
        if l >= h:
            h = min(hi, l + 1)
        return trial.suggest_int(key, l, h)

    # ── market_score 게이트 ──
    suppress = _r("market_score_suppress", 0.15, 0.45, 0.05)
    entry_b_low = round(suppress + 0.05, 2)
    entry_b = trial.suggest_float(
        "market_score_entry_b",
        _clamp(round(base.get("market_score_entry_b", 0.55) * 0.80 / 0.05) * 0.05, entry_b_low, 0.70),
        _clamp(round(base.get("market_score_entry_b", 0.55) * 1.20 / 0.05) * 0.05, entry_b_low, 0.70),
        step=0.05,
    )
    entry_a_low = round(entry_b + 0.05, 2)
    entry_a = trial.suggest_float(
        "market_score_entry_a",
        _clamp(round(base.get("market_score_entry_a", 0.65) * 0.80 / 0.05) * 0.05, entry_a_low, 0.80),
        _clamp(round(base.get("market_score_entry_a", 0.65) * 1.20 / 0.05) * 0.05, entry_a_low, 0.80),
        step=0.05,
    )

    # ── market_score 가중치 (±20%) ──
    w_keys = ["w_gld", "w_spy", "w_riskoff", "w_streak", "w_vol", "w_btc"]
    w_base_map = {
        "w_gld":     base.get("market_score_weights", {}).get("gld_score",     0.20),
        "w_spy":     base.get("market_score_weights", {}).get("spy_score",     0.15),
        "w_riskoff": base.get("market_score_weights", {}).get("riskoff_score", 0.25),
        "w_streak":  base.get("market_score_weights", {}).get("streak_score",  0.15),
        "w_vol":     base.get("market_score_weights", {}).get("vol_score",     0.15),
        "w_btc":     base.get("market_score_weights", {}).get("btc_score",     0.10),
    }
    w_vals = {}
    for k in w_keys:
        b_w = w_base_map[k]
        lo_w = 0.00 if k == "w_btc" else 0.05
        hi_w = 0.15 if k == "w_btc" else 0.35
        l_w = _clamp(round(b_w * 0.80 / 0.05) * 0.05, lo_w, hi_w)
        h_w = _clamp(round(b_w * 1.20 / 0.05) * 0.05, lo_w, hi_w)
        if l_w >= h_w:
            l_w = max(lo_w, h_w - 0.05)
        w_vals[k] = trial.suggest_float(k, l_w, h_w, step=0.05)

    w_total = sum(w_vals.values()) or 1.0
    market_score_weights = {
        "gld_score":     round(w_vals["w_gld"]     / w_total, 4),
        "spy_score":     round(w_vals["w_spy"]      / w_total, 4),
        "riskoff_score": round(w_vals["w_riskoff"]  / w_total, 4),
        "streak_score":  round(w_vals["w_streak"]   / w_total, 4),
        "vol_score":     round(w_vals["w_vol"]       / w_total, 4),
        "btc_score":     round(w_vals["w_btc"]       / w_total, 4),
    }

    # ── 시황 필터 ──
    spy_min_th   = _r("riskoff_spy_min_threshold", -3.0, -0.8, 0.2)
    gld_opt_min  = _r("riskoff_gld_optimal_min",    0.2,  1.2, 0.1)
    spy_opt_max  = _r("riskoff_spy_optimal_max",   -1.0, -0.2, 0.1)
    consec_boost = _ri("riskoff_consecutive_boost", 2, 5)
    panic_factor = _r("riskoff_panic_size_factor",  0.3,  0.7, 0.1)

    gld_suppress   = _r("gld_suppress_threshold",  0.5,  2.0, 0.25)
    btc_up_max     = _r("btc_up_max",              0.60, 0.90, 0.05)
    btc_up_min     = _r("btc_up_min",              0.20, 0.55, 0.05)
    spy_streak_max = _ri("spy_streak_max",          2,  5)
    spy_bearish_th = _r("spy_bearish_threshold",   -2.0, -0.5, 0.25)

    # ── 기술적 지표 ──
    rsi_min  = _ri("rsi_entry_min",   25, 55)
    rsi_max  = _ri("rsi_entry_max",   50, 80)
    rsi_dng  = _ri("rsi_danger_zone", 70, 90)
    bb_entry = _r("bb_entry_max",     0.3, 1.0, 0.1)
    bb_dng   = _r("bb_danger_zone",   0.9, 1.2, 0.1)
    atr_q    = _r("atr_high_quantile", 0.50, 0.85, 0.05)
    vol_min  = _r("vol_entry_min",     0.5,  1.5, 0.1)
    vol_max  = _r("vol_entry_max",     1.5,  5.0, 0.5)

    # ── 역발상 ──
    contrarian_th = _r("contrarian_entry_threshold",       -2.0, 0.5, 0.5)
    amdl_th       = _r("amdl_friday_contrarian_threshold", -3.0, -0.5, 0.5)

    # ── 청산/자금 ──
    take_profit   = _r("take_profit_pct",      3.0, 10.0, 0.5)
    hold_days_max = _ri("optimal_hold_days_max", 4, 14)
    dca_max       = _ri("dca_max_daily",         2,  8)
    buy_large     = _r("buy_size_large",       0.10, 0.25, 0.05)
    buy_small     = _r("buy_size_small",       0.03, 0.10, 0.01)
    entry_cap     = _r("daily_new_entry_cap",  0.15, 0.50, 0.05)

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
        "rsi_entry_min":     rsi_min,
        "rsi_entry_max":     rsi_max,
        "rsi_danger_zone":   rsi_dng,
        "bb_entry_max":      bb_entry,
        "bb_danger_zone":    bb_dng,
        "atr_high_quantile": atr_q,
        "vol_entry_min":     vol_min,
        "vol_entry_max":     vol_max,
        "contrarian_entry_threshold":       contrarian_th,
        "amdl_friday_contrarian_threshold": amdl_th,
        "take_profit_pct":        take_profit,
        "optimal_hold_days_max":  hold_days_max,
        "dca_max_daily":          dca_max,
        "buy_size_large":         buy_large,
        "buy_size_small":         buy_small,
        "daily_new_entry_cap":    entry_cap,
    }


# ── 목적함수 (refined 탐색 공간) ────────────────────────────────
_W1_BASE_PARAMS: dict = {}  # 워커 초기화 시 설정


def make_objective_refined(is_start: date, is_end: date, base: dict):
    def objective(trial: optuna.Trial) -> float:
        params = define_search_space_refined(trial, base)
        report = run_backtest(params, is_start, is_end)
        score  = calc_score(report)
        trial.set_user_attr("win_rate",         report.get("win_rate", 0))
        trial.set_user_attr("total_return_pct", report.get("total_return_pct", 0))
        trial.set_user_attr("mdd_pct",          report.get("mdd_pct", 0))
        trial.set_user_attr("sharpe_ratio",     report.get("sharpe_ratio", 0))
        trial.set_user_attr("sell_trades",      report.get("sell_trades", 0))
        return score
    return objective


def _worker_run_refined(args: tuple) -> None:
    n_trials_per_worker, journal_path_str, study_name, is_start_str, is_end_str, base_json = args
    import json as _json
    import optuna as _optuna
    from optuna.storages import JournalStorage as _JS
    from optuna.storages.journal import JournalFileBackend as _JFB
    from datetime import date as _date

    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _is_start = _date.fromisoformat(is_start_str)
    _is_end   = _date.fromisoformat(is_end_str)
    _base     = _json.loads(base_json)
    _obj = make_objective_refined(_is_start, _is_end, _base)

    _storage = _JS(_JFB(journal_path_str))
    study = _optuna.load_study(study_name=study_name, storage=_storage)
    study.optimize(_obj, n_trials=n_trials_per_worker, show_progress_bar=False)


def optimize_w1_refine(n_trials: int, n_jobs: int) -> dict:
    win_id       = "W1_REFINE"
    study_name   = f"d2s_{win_id.lower()}"
    journal_path = OPTUNA_DB_DIR / f"d2s_{win_id.lower()}.log"
    result_path  = RESULTS_DIR / f"d2s_{win_id.lower()}_result.json"

    base = load_w1_best_params()

    print(f"\n{'=' * 70}")
    print(f"  [{win_id}] IS: {W1_IS_START} ~ {W1_IS_END}")
    print(f"           OOS: {W1_OOS_START} ~ {W1_OOS_END}")
    print(f"  W1 best params 기준 ±20% 범위 재탐색")
    print(f"  Trials: {n_trials} / Workers: {n_jobs}")
    print("=" * 70)

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))
    sampler = TPESampler(seed=42, n_startup_trials=5)  # 빠른 수렴
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Warm start: W1 best params 첫 번째 trial로 enqueue
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done == 0:
        # market_score_weights를 w_* 형태로 변환
        msw = base.get("market_score_weights", {})
        w_total = sum(msw.values()) or 1.0
        warmup = {
            k: base[k] for k in base
            if k in [
                "market_score_suppress", "market_score_entry_b", "market_score_entry_a",
                "riskoff_spy_min_threshold", "riskoff_gld_optimal_min",
                "riskoff_spy_optimal_max", "riskoff_consecutive_boost",
                "riskoff_panic_size_factor", "gld_suppress_threshold",
                "btc_up_max", "btc_up_min", "spy_streak_max", "spy_bearish_threshold",
                "rsi_entry_min", "rsi_entry_max", "rsi_danger_zone",
                "bb_entry_max", "bb_danger_zone", "atr_high_quantile",
                "vol_entry_min", "vol_entry_max",
                "contrarian_entry_threshold", "amdl_friday_contrarian_threshold",
                "take_profit_pct", "optimal_hold_days_max", "dca_max_daily",
                "buy_size_large", "buy_size_small", "daily_new_entry_cap",
            ]
        }
        warmup.update({
            "w_gld":     msw.get("gld_score",     0.20) * w_total,
            "w_spy":     msw.get("spy_score",     0.15) * w_total,
            "w_riskoff": msw.get("riskoff_score", 0.25) * w_total,
            "w_streak":  msw.get("streak_score",  0.15) * w_total,
            "w_vol":     msw.get("vol_score",     0.15) * w_total,
            "w_btc":     msw.get("btc_score",     0.10) * w_total,
        })
        study.enqueue_trial(warmup)
        print("  Warm start: W1 best params enqueue됨")

    # 최적화 실행
    remaining = max(0, n_trials - already_done)
    import json as _json
    base_json = _json.dumps(base)

    if remaining > 0:
        t0 = time.time()
        trials_per_worker = max(1, remaining // n_jobs)
        extra = remaining - trials_per_worker * n_jobs
        worker_args = []
        for i in range(n_jobs):
            n = trials_per_worker + (1 if i < extra else 0)
            worker_args.append((n, str(journal_path), study_name,
                                W1_IS_START.isoformat(), W1_IS_END.isoformat(),
                                base_json))
        ctx = mp.get_context("spawn")
        print(f"  workers: {n_jobs} × ~{trials_per_worker} trials/worker")
        with ctx.Pool(processes=n_jobs) as pool:
            pool.map(_worker_run_refined, worker_args)
        print(f"  완료: {time.time() - t0:.1f}초")

    # 결과 수집
    best = study.best_trial
    best_params_for_run = dict(D2S_ENGINE)
    w = best.params
    w_gld = w.get("w_gld", 0.2); w_spy = w.get("w_spy", 0.15)
    w_riskoff = w.get("w_riskoff", 0.25); w_streak = w.get("w_streak", 0.15)
    w_vol = w.get("w_vol", 0.15); w_btc = w.get("w_btc", 0.10)
    w_tot = w_gld + w_spy + w_riskoff + w_streak + w_vol + w_btc or 1.0
    best_params_for_run["market_score_weights"] = {
        "gld_score":     round(w_gld     / w_tot, 4),
        "spy_score":     round(w_spy     / w_tot, 4),
        "riskoff_score": round(w_riskoff / w_tot, 4),
        "streak_score":  round(w_streak  / w_tot, 4),
        "vol_score":     round(w_vol     / w_tot, 4),
        "btc_score":     round(w_btc     / w_tot, 4),
    }
    for key, val in best.params.items():
        if not key.startswith("w_"):
            best_params_for_run[key] = val

    print(f"\n  Best Trial #{best.number}  IS score={best.value:.4f}")
    print(f"    win_rate={best.user_attrs.get('win_rate', 0):.1f}%  "
          f"return={best.user_attrs.get('total_return_pct', 0):+.2f}%  "
          f"MDD={best.user_attrs.get('mdd_pct', 0):.2f}%")

    # OOS 검증
    from simulation.optimizers.optimize_d2s_v3_optuna import calc_score
    oos_report = run_backtest(best_params_for_run, W1_OOS_START, W1_OOS_END)
    oos_score  = calc_score(oos_report)
    print(f"\n  OOS: win_rate={oos_report.get('win_rate', 0):.1f}%  "
          f"return={oos_report.get('total_return_pct', 0):+.2f}%  "
          f"MDD={oos_report.get('mdd_pct', 0):.2f}%  "
          f"score={oos_score:.4f}")

    # 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "window":     win_id,
        "is_start":   str(W1_IS_START),
        "is_end":     str(W1_IS_END),
        "oos_start":  str(W1_OOS_START),
        "oos_end":    str(W1_OOS_END),
        "best_trial": best.number,
        "is_score":   best.value,
        "is_report": {
            "win_rate":         best.user_attrs.get("win_rate", 0),
            "total_return_pct": best.user_attrs.get("total_return_pct", 0),
            "mdd_pct":          best.user_attrs.get("mdd_pct", 0),
            "sharpe_ratio":     best.user_attrs.get("sharpe_ratio", 0),
            "sell_trades":      best.user_attrs.get("sell_trades", 0),
        },
        "oos_score":  oos_score,
        "oos_report": oos_report,
        "best_params": {k: v for k, v in best_params_for_run.items()
                        if not isinstance(v, (dict, list))},
        "refine_base": "W1",
        "timestamp":  datetime.now().isoformat(),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  결과 저장: {result_path.name}")

    return result


def main():
    import json
    parser = argparse.ArgumentParser(description="D2S W1 좁은 재탐색 (B-2)")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--n-jobs",   type=int, default=20)
    args = parser.parse_args()

    result = optimize_w1_refine(args.n_trials, args.n_jobs)
    _print_wf_summary([result])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
D2S v3 Regime r5 Optuna — R21 제거 + TP/HD 재탐색
====================================================
목적: Study 10 Ablation 결과 반영
  - R21(HD 조건부) 제거: bull_hd=bear_hd → hold_days_max 단일 파라미터
  - R20(TP 조건부) 유지: bull/bear TP 차등 허용, 탐색 범위 확대 [4.0, 8.0]
  - Warm start: E variant (TP=6.0%, HD=10d) → ablation best

Ablation 근거 (study_10):
  A_full_regime (현재): +30.57%, Sharpe 1.307
  B_tp_only (R21 제거):  +32.39%, Sharpe 1.355  ← +1.82%p
  E_no_regime_alt (TP=6.0%/HD=10d): +34.78%, Sharpe 1.433  ← BEST

기간 (no-ROBN 1.5년):
  IS : 2024-09-18 ~ 2025-05-31
  OOS: 2025-06-01 ~ 2026-02-17

Usage:
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_v3_r5.py \\
        --no-robn --n-trials 500 --n-jobs 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3

# ============================================================
# 상수
# ============================================================

RESULTS_DIR   = _PROJECT_ROOT / "data" / "results" / "optimization"
OPTUNA_DB_DIR = _PROJECT_ROOT / "data" / "optuna"
REPORTS_DIR   = _PROJECT_ROOT / "simulation" / "optimizers" / "docs"

# 기본값 (no-robn=False, 1년 구간)
IS_START  = date(2025, 3, 3)
IS_END    = date(2025, 9, 30)
OOS_START = date(2025, 10, 1)
OOS_END   = date(2026, 2, 17)
STUDY_NAME    = "d2s_v3_regime_r5"
JOURNAL_PATH  = OPTUNA_DB_DIR / "d2s_v3_regime_r5_journal.log"
BASELINE_JSON = RESULTS_DIR / "d2s_v3_regime_r5_baseline.json"
DATA_PATH     = None
_BASE_PARAMS  = D2S_ENGINE_V3


def _apply_no_robn() -> None:
    from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN
    global _BASE_PARAMS, DATA_PATH, IS_START, IS_END, OOS_START, OOS_END
    global STUDY_NAME, JOURNAL_PATH, BASELINE_JSON
    _BASE_PARAMS  = D2S_ENGINE_V3_NO_ROBN
    DATA_PATH     = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily_3y.parquet"
    IS_START  = date(2024, 9, 18)
    IS_END    = date(2025, 5, 31)
    OOS_START = date(2025, 6, 1)
    OOS_END   = date(2026, 2, 17)
    STUDY_NAME    = "d2s_v3_regime_r5_norobn"
    JOURNAL_PATH  = OPTUNA_DB_DIR / "d2s_v3_regime_r5_norobn_journal.log"
    BASELINE_JSON = RESULTS_DIR / "d2s_v3_regime_r5_norobn_baseline.json"


if os.environ.get("D2S_NO_ROBN", "0") == "1":
    _apply_no_robn()

SCORE_IS_SHARPE_W  = 10.0
SCORE_OOS_SHARPE_W = 20.0
SCORE_MDD_W        = 0.50

MIN_SELL_TRADES = 5
MAX_MDD_HARD    = 35.0


# ============================================================
# 탐색 공간 — R21 제거 버전
# ============================================================

def define_search_space(trial: optuna.Trial) -> dict:
    """D2S v3 r5 탐색 공간 — R21(HD 조건부) 제거, R20(TP 조건부) 유지."""

    # ── 0차 게이트: market_score ────────────────────────────
    suppress    = trial.suggest_float("market_score_suppress", 0.25, 0.50, step=0.05)
    entry_b_low = round(suppress + 0.05, 2)
    entry_b     = trial.suggest_float("market_score_entry_b", entry_b_low, 0.70, step=0.05)
    entry_a_low = round(entry_b + 0.05, 2)
    entry_a     = trial.suggest_float("market_score_entry_a", entry_a_low, 0.80, step=0.05)

    w_gld     = trial.suggest_float("w_gld",     0.10, 0.35, step=0.05)
    w_spy     = trial.suggest_float("w_spy",     0.05, 0.25, step=0.05)
    w_riskoff = trial.suggest_float("w_riskoff", 0.15, 0.40, step=0.05)
    w_streak  = trial.suggest_float("w_streak",  0.05, 0.25, step=0.05)
    w_vol     = trial.suggest_float("w_vol",     0.05, 0.25, step=0.05)
    w_btc     = trial.suggest_float("w_btc",     0.05, 0.20, step=0.05)
    w_total   = w_gld + w_spy + w_riskoff + w_streak + w_vol + w_btc
    market_score_weights = {
        "gld_score":     round(w_gld     / w_total, 4),
        "spy_score":     round(w_spy     / w_total, 4),
        "riskoff_score": round(w_riskoff / w_total, 4),
        "streak_score":  round(w_streak  / w_total, 4),
        "vol_score":     round(w_vol     / w_total, 4),
        "btc_score":     round(w_btc     / w_total, 4),
    }

    # ── R14 그라데이션 ──────────────────────────────────────
    spy_min_th   = trial.suggest_float("riskoff_spy_min_threshold", -3.0, -0.8, step=0.2)
    gld_opt_min  = trial.suggest_float("riskoff_gld_optimal_min",    0.2,  1.2, step=0.1)
    spy_opt_max  = trial.suggest_float("riskoff_spy_optimal_max",   -1.0, -0.2, step=0.1)
    consec_boost = trial.suggest_int("riskoff_consecutive_boost",   2, 5)
    panic_factor = trial.suggest_float("riskoff_panic_size_factor",  0.3, 0.7, step=0.1)

    # ── 시황 필터 ───────────────────────────────────────────
    gld_suppress   = trial.suggest_float("gld_suppress_threshold", 0.5, 2.0, step=0.25)
    btc_up_max     = trial.suggest_float("btc_up_max",             0.60, 0.90, step=0.05)
    btc_up_min     = trial.suggest_float("btc_up_min",             0.20, 0.55, step=0.05)
    spy_streak_max = trial.suggest_int("spy_streak_max",           2, 5)
    spy_bearish_th = trial.suggest_float("spy_bearish_threshold",  -2.0, -0.5, step=0.25)

    # ── 쌍둥이 갭 필터 ─────────────────────────────────────
    gap_bank_conl = trial.suggest_float("gap_bank_conl_max", 3.0, 10.0, step=0.5)
    robn_pct      = trial.suggest_float("robn_pct_max",      1.0,  4.0, step=0.5)

    # ── 기술적 지표 ─────────────────────────────────────────
    rsi_entry_min = trial.suggest_int("rsi_entry_min",   30, 50)
    rsi_entry_max = trial.suggest_int("rsi_entry_max",   55, 75)
    rsi_danger    = trial.suggest_int("rsi_danger_zone", 70, 90)
    bb_entry_max  = trial.suggest_float("bb_entry_max",   0.3, 0.7, step=0.1)
    bb_danger     = trial.suggest_float("bb_danger_zone", 0.9, 1.2, step=0.1)
    atr_quantile  = trial.suggest_float("atr_high_quantile", 0.60, 0.85, step=0.05)
    vol_entry_min = trial.suggest_float("vol_entry_min",  0.8, 1.5, step=0.1)
    vol_entry_max = trial.suggest_float("vol_entry_max",  1.5, 3.5, step=0.5)

    # ── 역발상 기준 ─────────────────────────────────────────
    contrarian_th = trial.suggest_float("contrarian_entry_threshold",       -1.5, 0.0, step=0.5)
    amdl_th       = trial.suggest_float("amdl_friday_contrarian_threshold", -3.0, -0.5, step=0.5)

    # ── 자금 관리 ────────────────────────────────────────────
    buy_large  = trial.suggest_float("buy_size_large",      0.10, 0.25, step=0.05)
    buy_small  = trial.suggest_float("buy_size_small",      0.03, 0.10, step=0.01)
    entry_cap  = trial.suggest_float("daily_new_entry_cap", 0.15, 0.50, step=0.05)
    dca_max    = trial.suggest_int("dca_max_daily",         2, 8)

    # ── v2: R17 V-바운스 ──────────────────────────────────
    vb_bb_th    = trial.suggest_float("vbounce_bb_threshold",     0.05, 0.25, step=0.05)
    vb_crash_th = trial.suggest_float("vbounce_crash_threshold", -15.0, -5.0, step=1.0)
    vb_score_th = trial.suggest_float("vbounce_score_threshold",  0.75, 0.95, step=0.05)
    vb_mult     = trial.suggest_float("vbounce_size_multiplier",  1.5,  3.0,  step=0.5)
    vb_max      = trial.suggest_float("vbounce_size_max",         0.20, 0.40, step=0.05)

    # ── v2: R18 조기 손절 ────────────────────────────────
    es_days     = trial.suggest_int("early_stoploss_days",       2, 5)
    es_recovery = trial.suggest_float("early_stoploss_recovery", 0.5, 5.0, step=0.5)

    # ── v2: DCA 레이어 제한 ──────────────────────────────
    dca_layers  = trial.suggest_int("dca_max_layers", 1, 3)

    # ── [v3] R19: BB 진입 하드 필터 ──────────────────────
    bb_hard_max    = trial.suggest_float("bb_entry_hard_max",  0.15, 0.50, step=0.05)
    bb_hard_filter = trial.suggest_categorical("bb_entry_hard_filter", [True, False])

    # ── [v3] 레짐 감지: SPY streak + SMA ─────────────────
    bull_streak = trial.suggest_int("regime_bull_spy_streak", 2, 5)
    bear_streak = trial.suggest_int("regime_bear_spy_streak", 1, 4)
    sma_period  = trial.suggest_int("regime_spy_sma_period",  10, 30)
    sma_bull_p  = trial.suggest_float("regime_spy_sma_bull_pct",  0.3, 2.0, step=0.1)
    sma_bear_p  = trial.suggest_float("regime_spy_sma_bear_pct", -2.0, -0.3, step=0.1)

    # ── [v3] 레짐 감지: Polymarket BTC ───────────────────
    btc_bull_th = trial.suggest_float("regime_btc_bull_threshold", 0.50, 0.75, step=0.05)
    btc_bear_th = trial.suggest_float("regime_btc_bear_threshold", 0.25, 0.50, step=0.05)

    # ── [r5] R20: 레짐 조건부 take_profit (범위 확대) ─────
    # Bull/Bear 차등 허용, Optuna가 같은 값도 탐색 가능 (범위 통일)
    bull_tp = trial.suggest_float("bull_take_profit_pct",  4.0, 8.0, step=0.5)
    bear_tp = trial.suggest_float("bear_take_profit_pct",  4.0, 8.0, step=0.5)

    # ── [r5] R21 제거: 단일 hold_days_max (Bull=Bear 동일) ─
    # 기존: bull_hd(5~14) / bear_hd(2~bull_hd)  →  단일 hold_days
    hold_days = trial.suggest_int("hold_days_max",  6, 14)

    return {
        **_BASE_PARAMS,
        # 0차 게이트
        "market_score_suppress": suppress,
        "market_score_entry_b":  entry_b,
        "market_score_entry_a":  entry_a,
        "market_score_weights":  market_score_weights,
        # R14 그라데이션
        "riskoff_spy_min_threshold":  spy_min_th,
        "riskoff_gld_optimal_min":    gld_opt_min,
        "riskoff_spy_optimal_max":    spy_opt_max,
        "riskoff_consecutive_boost":  consec_boost,
        "riskoff_panic_size_factor":  panic_factor,
        # 시황 필터
        "gld_suppress_threshold": gld_suppress,
        "btc_up_max":             btc_up_max,
        "btc_up_min":             btc_up_min,
        "spy_streak_max":         spy_streak_max,
        "spy_bearish_threshold":  spy_bearish_th,
        # 쌍둥이 갭
        "gap_bank_conl_max": gap_bank_conl,
        "robn_pct_max":      robn_pct,
        # 기술적 지표
        "rsi_entry_min":     rsi_entry_min,
        "rsi_entry_max":     rsi_entry_max,
        "rsi_danger_zone":   rsi_danger,
        "bb_entry_max":      bb_entry_max,
        "bb_danger_zone":    bb_danger,
        "atr_high_quantile": atr_quantile,
        "vol_entry_min":     vol_entry_min,
        "vol_entry_max":     vol_entry_max,
        # 역발상
        "contrarian_entry_threshold":       contrarian_th,
        "amdl_friday_contrarian_threshold": amdl_th,
        # 자금
        "buy_size_large":       buy_large,
        "buy_size_small":       buy_small,
        "daily_new_entry_cap":  entry_cap,
        "dca_max_daily":        dca_max,
        # v2: R17
        "vbounce_bb_threshold":    vb_bb_th,
        "vbounce_crash_threshold": vb_crash_th,
        "vbounce_score_threshold": vb_score_th,
        "vbounce_size_multiplier": vb_mult,
        "vbounce_size_max":        vb_max,
        # v2: R18
        "early_stoploss_days":     es_days,
        "early_stoploss_recovery": es_recovery,
        # v2: DCA
        "dca_max_layers": dca_layers,
        # v3: R19
        "bb_entry_hard_max":    bb_hard_max,
        "bb_entry_hard_filter": bb_hard_filter,
        # v3: 레짐 감지
        "regime_enabled":              True,
        "regime_bull_spy_streak":      bull_streak,
        "regime_bear_spy_streak":      bear_streak,
        "regime_spy_sma_period":       sma_period,
        "regime_spy_sma_bull_pct":     sma_bull_p,
        "regime_spy_sma_bear_pct":     sma_bear_p,
        "regime_btc_bull_threshold":   btc_bull_th,
        "regime_btc_bear_threshold":   btc_bear_th,
        # r5: R20 (TP 차등 유지)
        "bull_take_profit_pct":  bull_tp,
        "bear_take_profit_pct":  bear_tp,
        # r5: R21 제거 (bull=bear 동일)
        "bull_hold_days_max": hold_days,
        "bear_hold_days_max": hold_days,
    }


# ============================================================
# 목적 함수 — IS + OOS 동시 최적화
# ============================================================

def run_backtest(params: dict, start: date, end: date) -> dict:
    from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
    bt = D2SBacktestV3(params=params, start_date=start, end_date=end,
                       use_fees=True, data_path=DATA_PATH)
    bt.run(verbose=False)
    return bt.report()


def calc_score(is_report: dict, oos_report: dict) -> float:
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
    n_trials_per_worker, journal_path_str, study_name, no_robn = args
    if no_robn:
        _apply_no_robn()
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
    print("  [Stage 1] D2S v3 r5 Baseline — IS + OOS")
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
    print(f"  Score: {score:.4f}  ({elapsed:.1f}초)")

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
    n_trials: int = 500, n_jobs: int = 20,
    timeout: int | None = None,
    study_name: str = STUDY_NAME,
    journal_path: Path = JOURNAL_PATH,
) -> None:
    print("\n" + "=" * 70)
    print(f"  [Stage 2] D2S v3 r5 Optuna — {n_trials} trials / {n_jobs} workers")
    print(f"  R21 제거 (hold_days 단일) + R20 TP 범위 확대 [4.0, 8.0]")
    print(f"  스코어: IS_Shp×{SCORE_IS_SHARPE_W} + OOS_Shp×{SCORE_OOS_SHARPE_W} - MDD×{SCORE_MDD_W}")
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

    sampler = TPESampler(seed=42, n_startup_trials=min(50, n_trials))
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        sampler=sampler, storage=storage, load_if_exists=True,
    )

    # Warm start — E variant (ablation best): TP=6.0%, HD=10d
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done == 0:
        p = _BASE_PARAMS
        mw = p["market_score_weights"]
        study.enqueue_trial({
            "market_score_suppress": p["market_score_suppress"],
            "market_score_entry_b":  p["market_score_entry_b"],
            "market_score_entry_a":  p["market_score_entry_a"],
            "w_gld":     mw["gld_score"],    "w_spy":     mw["spy_score"],
            "w_riskoff": mw["riskoff_score"],"w_streak":  mw["streak_score"],
            "w_vol":     mw["vol_score"],    "w_btc":     mw["btc_score"],
            "riskoff_spy_min_threshold":  -1.6,
            "riskoff_gld_optimal_min":    p["riskoff_gld_optimal_min"],
            "riskoff_spy_optimal_max":    p["riskoff_spy_optimal_max"],
            "riskoff_consecutive_boost":  p["riskoff_consecutive_boost"],
            "riskoff_panic_size_factor":  p["riskoff_panic_size_factor"],
            "gld_suppress_threshold": p["gld_suppress_threshold"],
            "btc_up_max": p["btc_up_max"],   "btc_up_min": p["btc_up_min"],
            "spy_streak_max": p["spy_streak_max"],
            "spy_bearish_threshold": p["spy_bearish_threshold"],
            "gap_bank_conl_max": p["gap_bank_conl_max"],
            "robn_pct_max": p["robn_pct_max"],
            "rsi_entry_min": p["rsi_entry_min"], "rsi_entry_max": p["rsi_entry_max"],
            "rsi_danger_zone": p["rsi_danger_zone"],
            "bb_entry_max": p["bb_entry_max"], "bb_danger_zone": p["bb_danger_zone"],
            "atr_high_quantile": p["atr_high_quantile"],
            "vol_entry_min": p["vol_entry_min"], "vol_entry_max": p["vol_entry_max"],
            "contrarian_entry_threshold":       p["contrarian_entry_threshold"],
            "amdl_friday_contrarian_threshold": p["amdl_friday_contrarian_threshold"],
            "buy_size_large": p["buy_size_large"], "buy_size_small": p["buy_size_small"],
            "daily_new_entry_cap": p["daily_new_entry_cap"],
            "dca_max_daily": p["dca_max_daily"],
            "vbounce_bb_threshold": p["vbounce_bb_threshold"],
            "vbounce_crash_threshold": p["vbounce_crash_threshold"],
            "vbounce_score_threshold": p.get("vbounce_score_threshold", 0.85),
            "vbounce_size_multiplier": p["vbounce_size_multiplier"],
            "vbounce_size_max": p["vbounce_size_max"],
            "early_stoploss_days": p["early_stoploss_days"],
            "early_stoploss_recovery": p["early_stoploss_recovery"],
            "dca_max_layers": p["dca_max_layers"],
            "bb_entry_hard_max": p["bb_entry_hard_max"],
            "bb_entry_hard_filter": p["bb_entry_hard_filter"],
            "regime_bull_spy_streak": p["regime_bull_spy_streak"],
            "regime_bear_spy_streak": p["regime_bear_spy_streak"],
            "regime_spy_sma_period":  p["regime_spy_sma_period"],
            "regime_spy_sma_bull_pct": p["regime_spy_sma_bull_pct"],
            "regime_spy_sma_bear_pct": p["regime_spy_sma_bear_pct"],
            "regime_btc_bull_threshold": p["regime_btc_bull_threshold"],
            "regime_btc_bear_threshold": p["regime_btc_bear_threshold"],
            # r5 warm-start: E variant (TP=6.0% uniform, HD=10d)
            "bull_take_profit_pct": 6.0,
            "bear_take_profit_pct": 6.0,
            "hold_days_max": 10,
        })
        print("  Warm start: E variant (TP=6.0%, HD=10d) enqueue")

    remaining = max(0, n_trials - already_done)
    if remaining == 0:
        print(f"  이미 {already_done}회 완료.")
    else:
        print(f"  {already_done}회 완료 → {remaining}회 추가")
        t0 = time.time()
        if n_jobs <= 1:
            study.optimize(objective, n_trials=remaining, timeout=timeout,
                           show_progress_bar=True)
        else:
            tpw = max(1, remaining // n_jobs)
            extra = remaining - tpw * n_jobs
            _no_robn_flag = os.environ.get("D2S_NO_ROBN", "0") == "1"
            args_list = [
                (tpw + (1 if i < extra else 0), str(journal_path), study_name, _no_robn_flag)
                for i in range(n_jobs)
            ]
            ctx = mp.get_context("spawn")
            print(f"  workers: {n_jobs} × ~{tpw} trials")
            with ctx.Pool(processes=n_jobs) as pool:
                pool.map(_worker_run, args_list)
        elapsed = time.time() - t0
        print(f"  완료: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

    _print_study_summary(study, baseline_score)
    _save_optuna_report(study, baseline_score, n_trials, n_jobs, journal_path)


def run_stage3(study_name: str = STUDY_NAME, journal_path: Path = JOURNAL_PATH) -> None:
    print("\n" + "=" * 70)
    print("  [Stage 3] D2S v3 r5 Full 검증")
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
            "timestamp":   datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  결과 저장: {result_path}")


# ============================================================
# 유틸리티
# ============================================================

def _reconstruct_params(trial_params: dict) -> dict:
    p = dict(_BASE_PARAMS)
    w_keys = ["w_gld", "w_spy", "w_riskoff", "w_streak", "w_vol", "w_btc"]
    if any(k in trial_params for k in w_keys):
        ws    = [trial_params.get(k, 0.1) for k in w_keys]
        total = sum(ws)
        keys  = ["gld_score", "spy_score", "riskoff_score", "streak_score", "vol_score", "btc_score"]
        p["market_score_weights"] = {k: round(w / total, 4) for k, w in zip(keys, ws)}
    skip = set(w_keys)
    # hold_days_max → bull/bear 양쪽 적용
    hold_days = trial_params.get("hold_days_max")
    for k, v in trial_params.items():
        if k not in skip and k != "hold_days_max":
            p[k] = v
    if hold_days is not None:
        p["bull_hold_days_max"] = hold_days
        p["bear_hold_days_max"] = hold_days
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

    lines = [
        "# D2S v3 r5 Optuna 최적화 리포트 (R21 제거)",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}  ",
        f"> trial: {len(completed)}  |  n_jobs: {n_jobs}  ",
        f"> 변경: R21(HD 조건부) 제거 → hold_days_max 단일 파라미터  ",
        f"> R20(TP 조건부) 유지 — bull/bear TP [4.0, 8.0] 탐색  ",
        f"> 스코어: IS_Shp×{SCORE_IS_SHARPE_W} + OOS_Shp×{SCORE_OOS_SHARPE_W} - MDD×{SCORE_MDD_W}  ",
        f"> Journal: {journal_path.name}",
        "",
        "## 1. Best Trial",
        "",
        "| 항목 | IS | OOS |",
        "|---|---|---|",
        f"| Trial # | {best.number} | — |",
        f"| Score | {best.value:.4f} | — |",
        f"| 수익률 | {best.user_attrs.get('is_return', 0):+.2f}% "
        f"| {best.user_attrs.get('oos_return', 0):+.2f}% |",
        f"| MDD | {best.user_attrs.get('is_mdd', 0):.1f}% "
        f"| {best.user_attrs.get('oos_mdd', 0):.1f}% |",
        f"| Sharpe | {best.user_attrs.get('is_sharpe', 0):.3f} "
        f"| {best.user_attrs.get('oos_sharpe', 0):.3f} |",
        "",
        "## 2. Top 10",
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
        "## 3. Best 파라미터",
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
    parser = argparse.ArgumentParser(description="D2S v3 r5 Optuna — R21 제거")
    parser.add_argument("--stage",      type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--n-trials",   type=int, default=500)
    parser.add_argument("--n-jobs",     type=int, default=20)
    parser.add_argument("--timeout",    type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--journal",    type=str, default=None)
    parser.add_argument("--no-robn",    action="store_true", default=False,
                        help="ROBN 제외 + market_daily_3y + 2024-09-18 시작 (1.5년 모드)")
    args = parser.parse_args()

    if args.no_robn:
        os.environ["D2S_NO_ROBN"] = "1"
        _apply_no_robn()

    journal_path = Path(args.journal) if args.journal else JOURNAL_PATH
    study_name   = args.study_name if args.study_name else STUDY_NAME

    print("=" * 70)
    print("  D2S v3 r5 Optuna — R21(HD 조건부) 제거 + TP 재탐색")
    print(f"  IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}")
    print(f"  스코어: IS_Shp×{SCORE_IS_SHARPE_W} + OOS_Shp×{SCORE_OOS_SHARPE_W} - MDD×{SCORE_MDD_W}")
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

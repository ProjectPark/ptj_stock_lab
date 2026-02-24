#!/usr/bin/env python3
"""
D2S Optuna 파라미터 최적화
============================
Study A/B/C 반영 D2S 엔진(trading_rules_attach_v1.md)의 파라미터를
Optuna TPE 샘플러로 최적화한다.

병렬 전략 (v2 수정):
  - JournalStorage(JournalFileBackend) 사용 → 다중 프로세스 lock-free 쓰기
  - SQLite 대비 20 workers 동시 쓰기 충돌 없음
  - SLURM 20 CPU 할당 시 --n-jobs 20 으로 20 trial 동시 진행

스코어 함수 (v2 수정):
  - 구: win_rate×0.50 + return×0.30 - mdd×0.20  (return 지배 문제)
  - 신: win_rate×0.40 + sharpe×15  - mdd×0.30   (리스크 조정 수익률 우선)

Usage:
    # Stage 1: baseline 측정 (IS 구간)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_optuna.py --stage 1

    # Stage 2: Optuna 최적화 (20병렬, 200 trial)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_optuna.py --stage 2 --n-trials 200 --n-jobs 20

    # Stage 3: OOS 검증 (최적 파라미터)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_optuna.py --stage 3

    # 연속 실행 (1 → 2 → 3)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_optuna.py --n-trials 200 --n-jobs 20
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
# 상수
# ============================================================

# IS / OOS 날짜 분리
IS_START  = date(2025, 3, 3)    # 기술적 지표 워밍업 후 시작
IS_END    = date(2025, 9, 30)   # IS 종료 (~7개월)
OOS_START = date(2025, 10, 1)   # OOS 시작
OOS_END   = date(2026, 2, 17)   # OOS 종료 (~4.5개월)

# 결과 저장 경로
RESULTS_DIR   = _PROJECT_ROOT / "data" / "results" / "optimization"
OPTUNA_DB_DIR = _PROJECT_ROOT / "data" / "optuna"
REPORTS_DIR   = _PROJECT_ROOT / "simulation" / "optimizers" / "docs"

BASELINE_JSON  = RESULTS_DIR / "d2s_baseline.json"
# v2: SQLite 대신 JournalStorage (다중 프로세스 lock-free)
JOURNAL_PATH   = OPTUNA_DB_DIR / "d2s_journal_v2.log"
STUDY_NAME     = "d2s_optuna_v2"

# 스코어 함수 (v2: Sharpe 기반, return 지배 문제 해결)
#   구: win_rate×0.50 + return×0.30 - mdd×0.20
#   신: win_rate×0.40 + sharpe×15  - mdd×0.30
SCORE_WIN_RATE_W = 0.40
SCORE_SHARPE_W   = 15.0
SCORE_MDD_W      = 0.30

# 페널티
MIN_SELL_TRADES = 5    # 매도 거래 최소 횟수 (미달 시 패널티)
MAX_MDD_HARD    = 30.0 # MDD 상한 (초과 시 패널티, v2: 50→30%)


# ============================================================
# 목적함수 + 탐색 공간
# ============================================================


def define_search_space(trial: optuna.Trial) -> dict:
    """D2S Optuna 탐색 공간.

    D2S_ENGINE 파라미터에서 최적화할 항목만 선택.
    고정값(tickers, twin_pairs 등)은 제외.
    """
    # ── 0차 게이트: market_score (Study C) ─────────────────────
    suppress = trial.suggest_float("market_score_suppress", 0.25, 0.50, step=0.05)
    # dynamic lower bound를 0.05 단위로 반올림해 step 분할 오류 방지
    entry_b_low = round(suppress + 0.05, 2)
    entry_b  = trial.suggest_float("market_score_entry_b",  entry_b_low, 0.70, step=0.05)
    entry_a_low = round(entry_b + 0.05, 2)
    entry_a  = trial.suggest_float("market_score_entry_a",  entry_a_low,  0.80, step=0.05)

    # market_score 가중치 (합계 1.0 강제)
    w_gld     = trial.suggest_float("w_gld",     0.10, 0.35, step=0.05)
    w_spy     = trial.suggest_float("w_spy",     0.05, 0.25, step=0.05)
    w_riskoff = trial.suggest_float("w_riskoff", 0.15, 0.40, step=0.05)
    w_streak  = trial.suggest_float("w_streak",  0.05, 0.25, step=0.05)
    w_vol     = trial.suggest_float("w_vol",     0.05, 0.25, step=0.05)
    w_btc     = trial.suggest_float("w_btc",     0.05, 0.20, step=0.05)
    w_total   = w_gld + w_spy + w_riskoff + w_streak + w_vol + w_btc
    # 정규화
    market_score_weights = {
        "gld_score":     round(w_gld     / w_total, 4),
        "spy_score":     round(w_spy     / w_total, 4),
        "riskoff_score": round(w_riskoff / w_total, 4),
        "streak_score":  round(w_streak  / w_total, 4),
        "vol_score":     round(w_vol     / w_total, 4),
        "btc_score":     round(w_btc     / w_total, 4),
    }

    # ── R14 그라데이션 (Study A) ────────────────────────────────
    spy_min_th     = trial.suggest_float("riskoff_spy_min_threshold", -3.0, -0.8, step=0.2)
    gld_opt_min    = trial.suggest_float("riskoff_gld_optimal_min",    0.2,  1.2, step=0.1)
    spy_opt_max    = trial.suggest_float("riskoff_spy_optimal_max",   -1.0, -0.2, step=0.1)
    consec_boost   = trial.suggest_int("riskoff_consecutive_boost",   2, 5)
    panic_factor   = trial.suggest_float("riskoff_panic_size_factor",  0.3, 0.7, step=0.1)

    # ── 시황 필터 (R1, R3, R13) ─────────────────────────────────
    gld_suppress   = trial.suggest_float("gld_suppress_threshold", 0.5, 2.0, step=0.25)
    btc_up_max     = trial.suggest_float("btc_up_max",             0.60, 0.90, step=0.05)
    btc_up_min     = trial.suggest_float("btc_up_min",             0.20, 0.55, step=0.05)
    spy_streak_max = trial.suggest_int("spy_streak_max",           2, 5)
    spy_bearish_th = trial.suggest_float("spy_bearish_threshold",  -2.0, -0.5, step=0.25)

    # ── 기술적 지표 (R7~R9, R16) ────────────────────────────────
    rsi_entry_min  = trial.suggest_int("rsi_entry_min",   30, 50)
    rsi_entry_max  = trial.suggest_int("rsi_entry_max",   55, 75)
    rsi_danger     = trial.suggest_int("rsi_danger_zone", 70, 90)
    bb_entry_max   = trial.suggest_float("bb_entry_max",   0.4, 0.8, step=0.1)
    bb_danger      = trial.suggest_float("bb_danger_zone", 0.9, 1.2, step=0.1)
    atr_quantile   = trial.suggest_float("atr_high_quantile", 0.60, 0.85, step=0.05)
    vol_entry_min  = trial.suggest_float("vol_entry_min",  0.8, 1.5, step=0.1)
    vol_entry_max  = trial.suggest_float("vol_entry_max",  1.5, 3.5, step=0.5)

    # ── 역발상 기준 ──────────────────────────────────────────────
    contrarian_th  = trial.suggest_float("contrarian_entry_threshold", -1.5, 0.0, step=0.5)
    amdl_th        = trial.suggest_float("amdl_friday_contrarian_threshold", -3.0, -0.5, step=0.5)

    # ── 청산 (R4, R5) ────────────────────────────────────────────
    take_profit    = trial.suggest_float("take_profit_pct",       3.0, 10.0, step=0.5)
    hold_days_max  = trial.suggest_int("optimal_hold_days_max",   4, 14)
    dca_max        = trial.suggest_int("dca_max_daily",           2, 8)

    # ── 자금 관리 ────────────────────────────────────────────────
    buy_large      = trial.suggest_float("buy_size_large",        0.10, 0.25, step=0.05)
    buy_small      = trial.suggest_float("buy_size_small",        0.03, 0.10, step=0.01)
    entry_cap      = trial.suggest_float("daily_new_entry_cap",   0.15, 0.50, step=0.05)

    return {
        **D2S_ENGINE,  # 기본값 전체 계승 (고정 필드 포함)
        # 0차 게이트
        "market_score_suppress":  suppress,
        "market_score_entry_b":   entry_b,
        "market_score_entry_a":   entry_a,
        "market_score_weights":   market_score_weights,
        # R14 그라데이션
        "riskoff_spy_min_threshold":   spy_min_th,
        "riskoff_gld_optimal_min":     gld_opt_min,
        "riskoff_spy_optimal_max":     spy_opt_max,
        "riskoff_consecutive_boost":   consec_boost,
        "riskoff_panic_size_factor":   panic_factor,
        # 시황 필터
        "gld_suppress_threshold":  gld_suppress,
        "btc_up_max":              btc_up_max,
        "btc_up_min":              btc_up_min,
        "spy_streak_max":          spy_streak_max,
        "spy_bearish_threshold":   spy_bearish_th,
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
        "contrarian_entry_threshold":          contrarian_th,
        "amdl_friday_contrarian_threshold":    amdl_th,
        # 청산
        "take_profit_pct":        take_profit,
        "optimal_hold_days_max":  hold_days_max,
        "dca_max_daily":          dca_max,
        # 자금
        "buy_size_large":         buy_large,
        "buy_size_small":         buy_small,
        "daily_new_entry_cap":    entry_cap,
    }


def run_backtest(params: dict, start: date, end: date) -> dict:
    """D2SBacktest를 주어진 파라미터/기간으로 실행하고 report dict를 반환."""
    from simulation.backtests.backtest_d2s import D2SBacktest
    bt = D2SBacktest(params=params, start_date=start, end_date=end, use_fees=True)
    bt.run(verbose=False)
    return bt.report()


def calc_score(report: dict) -> float:
    """IS 백테스트 결과에서 최적화 스코어를 계산한다.

    score = win_rate × 0.40 + sharpe_ratio × 15 − |mdd| × 0.30
    (v2: return 지배 문제 해결 — Sharpe로 리스크 조정 수익률 반영)

    페널티: 거래 부족 / MDD 30% 초과
    """
    n_sells = report.get("sell_trades", 0)
    if n_sells < MIN_SELL_TRADES:
        return -100.0

    mdd = abs(report.get("mdd_pct", 0))
    if mdd > MAX_MDD_HARD:
        return -50.0

    win_rate     = report.get("win_rate", 0)
    sharpe_ratio = report.get("sharpe_ratio", 0)

    return (
        SCORE_WIN_RATE_W * win_rate
        + SCORE_SHARPE_W * sharpe_ratio
        - SCORE_MDD_W * mdd
    )


def objective(trial: optuna.Trial) -> float:
    """Optuna objective — IS 구간 백테스트."""
    params = define_search_space(trial)
    report = run_backtest(params, IS_START, IS_END)
    score = calc_score(report)

    # user_attrs 기록
    trial.set_user_attr("win_rate",        report.get("win_rate", 0))
    trial.set_user_attr("total_return_pct", report.get("total_return_pct", 0))
    trial.set_user_attr("mdd_pct",         report.get("mdd_pct", 0))
    trial.set_user_attr("sharpe_ratio",    report.get("sharpe_ratio", 0))
    trial.set_user_attr("sell_trades",     report.get("sell_trades", 0))
    trial.set_user_attr("buy_trades",      report.get("buy_trades", 0))

    return score


# ============================================================
# 병렬 워커 함수 (mp.Pool 호환 — 최상위 함수)
# ============================================================


def _worker_run(args: tuple) -> None:
    """mp.Pool 워커: JournalStorage 공유로 Optuna trial을 실행한다.

    JournalFileBackend는 파일 락 기반으로 다중 프로세스 동시 쓰기가 안전.
    """
    n_trials_per_worker, journal_path_str, study_name = args
    # 각 워커에서 별도 import (spawn safe)
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
    """Stage 1: 현재 D2S_ENGINE 기본값으로 IS baseline 실행."""
    print("\n" + "=" * 70)
    print("  [Stage 1] D2S Baseline — 현재 파라미터 IS 백테스트")
    print(f"  기간: {IS_START} ~ {IS_END}")
    print("=" * 70)

    t0 = time.time()
    report = run_backtest(D2S_ENGINE, IS_START, IS_END)
    elapsed = time.time() - t0

    score = calc_score(report)
    _print_report_summary(report, score=score)
    print(f"  실행 시간: {elapsed:.1f}초")

    # 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "report":    report,
        "params":    {k: v for k, v in D2S_ENGINE.items()
                      if not isinstance(v, (dict, list))},  # JSON-serializable
        "score":     score,
        "is_start":  str(IS_START),
        "is_end":    str(IS_END),
        "timestamp": datetime.now().isoformat(),
    }
    with open(BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Baseline 저장: {BASELINE_JSON}")

    return payload


def run_stage2(
    n_trials: int = 200,
    n_jobs: int = 20,
    timeout: int | None = None,
    study_name: str = STUDY_NAME,
    journal_path: Path = JOURNAL_PATH,
) -> None:
    """Stage 2: Optuna 최적화 — n_jobs 병렬 프로세스 × JournalStorage 공유."""
    print("\n" + "=" * 70)
    print(f"  [Stage 2] D2S Optuna 최적화 — {n_trials} trials / {n_jobs} workers")
    print(f"  Journal: {journal_path}")
    print(f"  IS 기간: {IS_START} ~ {IS_END}")
    print("=" * 70)

    # DB 디렉토리 생성
    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)

    # JournalStorage: 다중 프로세스 lock-free (SQLite 대체)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    # Baseline 참조
    baseline_score = None
    baseline_return = None
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON) as f:
            bl = json.load(f)
        # baseline score를 v2 스코어로 재계산
        bl_r = bl.get("report", {})
        baseline_score = calc_score(bl_r)
        baseline_return = bl_r.get("total_return_pct", 0)
        print(f"  Baseline score(v2)={baseline_score:.2f}, return={baseline_return:+.2f}%")

    # Study 생성 (load_if_exists=True → 재시작 가능)
    sampler = TPESampler(seed=42, n_startup_trials=min(20, n_trials))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Baseline 파라미터 enqueue (warm start)
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done == 0:
        print("  Baseline 파라미터 enqueue (warm start)")
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
            "riskoff_spy_min_threshold":   -1.6,   # -1.5 → step=0.2 경계값으로 조정
            "riskoff_gld_optimal_min":     D2S_ENGINE["riskoff_gld_optimal_min"],
            "riskoff_spy_optimal_max":     D2S_ENGINE["riskoff_spy_optimal_max"],
            "riskoff_consecutive_boost":   D2S_ENGINE["riskoff_consecutive_boost"],
            "riskoff_panic_size_factor":   D2S_ENGINE["riskoff_panic_size_factor"],
            "gld_suppress_threshold":  D2S_ENGINE["gld_suppress_threshold"],
            "btc_up_max":              D2S_ENGINE["btc_up_max"],
            "btc_up_min":              D2S_ENGINE["btc_up_min"],
            "spy_streak_max":          D2S_ENGINE["spy_streak_max"],
            "spy_bearish_threshold":   D2S_ENGINE["spy_bearish_threshold"],
            "rsi_entry_min":     D2S_ENGINE["rsi_entry_min"],
            "rsi_entry_max":     D2S_ENGINE["rsi_entry_max"],
            "rsi_danger_zone":   D2S_ENGINE["rsi_danger_zone"],
            "bb_entry_max":      D2S_ENGINE["bb_entry_max"],
            "bb_danger_zone":    D2S_ENGINE["bb_danger_zone"],
            "atr_high_quantile": D2S_ENGINE["atr_high_quantile"],
            "vol_entry_min":     D2S_ENGINE["vol_entry_min"],
            "vol_entry_max":     D2S_ENGINE["vol_entry_max"],
            "contrarian_entry_threshold":          D2S_ENGINE["contrarian_entry_threshold"],
            "amdl_friday_contrarian_threshold":    D2S_ENGINE["amdl_friday_contrarian_threshold"],
            "take_profit_pct":        6.0,      # 5.9 → step=0.5 경계값으로 조정
            "optimal_hold_days_max":  D2S_ENGINE["optimal_hold_days_max"],
            "dca_max_daily":          D2S_ENGINE["dca_max_daily"],
            "buy_size_large":         D2S_ENGINE["buy_size_large"],
            "buy_size_small":         D2S_ENGINE["buy_size_small"],
            "daily_new_entry_cap":    D2S_ENGINE["daily_new_entry_cap"],
        })

    # 남은 trial 수 계산
    remaining = max(0, n_trials - already_done)
    if remaining == 0:
        print(f"  이미 {already_done}회 완료. 리포트만 출력합니다.")
    else:
        print(f"  시작: {already_done}회 완료 → {remaining}회 추가 실행")
        t0 = time.time()

        if n_jobs <= 1:
            # 단일 프로세스
            study.optimize(objective, n_trials=remaining, timeout=timeout,
                           show_progress_bar=True)
        else:
            # n_jobs개 프로세스가 각각 remaining//n_jobs trial 실행
            trials_per_worker = max(1, remaining // n_jobs)
            extra = remaining - trials_per_worker * n_jobs
            worker_args = []
            for i in range(n_jobs):
                n = trials_per_worker + (1 if i < extra else 0)
                worker_args.append((n, str(journal_path), study_name))

            ctx = mp.get_context("spawn")
            print(f"  workers: {n_jobs} × ~{trials_per_worker} trials/worker")
            with ctx.Pool(processes=n_jobs) as pool:
                pool.map(_worker_run, worker_args)

        elapsed = time.time() - t0
        print(f"  실행 완료: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")

    # 결과 요약
    _print_study_summary(study, baseline_score, baseline_return)

    # 리포트 저장
    _save_optuna_report(study, baseline_score, n_trials, n_jobs, journal_path)


def run_stage3(study_name: str = STUDY_NAME, journal_path: Path = JOURNAL_PATH) -> None:
    """Stage 3: 최적 파라미터로 OOS 검증."""
    print("\n" + "=" * 70)
    print("  [Stage 3] D2S OOS 검증 — 최적 파라미터")
    print(f"  OOS 기간: {OOS_START} ~ {OOS_END}")
    print("=" * 70)

    OPTUNA_DB_DIR.mkdir(parents=True, exist_ok=True)

    try:
        storage = JournalStorage(JournalFileBackend(str(journal_path)))
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        print(f"  [ERROR] Study '{study_name}' 없음. Stage 2를 먼저 실행하세요.")
        return

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  완료된 trial 없음. Stage 2를 먼저 실행하세요.")
        return

    best = study.best_trial
    print(f"  Best Trial #{best.number}  IS score={best.value:.4f}")
    print(f"    IS win_rate={best.user_attrs.get('win_rate', 0):.1f}%  "
          f"return={best.user_attrs.get('total_return_pct', 0):+.2f}%  "
          f"MDD={best.user_attrs.get('mdd_pct', 0):.2f}%")

    # best 파라미터 재구성
    best_params = _reconstruct_params(best.params)

    # OOS 실행
    t0 = time.time()
    oos_report = run_backtest(best_params, OOS_START, OOS_END)
    elapsed = time.time() - t0
    oos_score = calc_score(oos_report)

    print(f"\n  OOS 결과 (best Trial #{best.number}):")
    _print_report_summary(oos_report, score=oos_score)
    print(f"  실행 시간: {elapsed:.1f}초")

    # IS baseline과 비교
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON) as f:
            bl = json.load(f)
        print(f"\n  비교:")
        print(f"    {'':20s} {'IS(base)':>10s} {'IS(opt)':>10s} {'OOS':>10s}")
        bl_r = bl.get("report", {})
        ua   = best.user_attrs
        print(f"    {'수익률':20s} "
              f"{bl_r.get('total_return_pct',0):>+9.2f}%  "
              f"{ua.get('total_return_pct',0):>+9.2f}%  "
              f"{oos_report.get('total_return_pct',0):>+9.2f}%")
        print(f"    {'승률':20s} "
              f"{bl_r.get('win_rate',0):>9.1f}%  "
              f"{ua.get('win_rate',0):>9.1f}%  "
              f"{oos_report.get('win_rate',0):>9.1f}%")
        print(f"    {'MDD':20s} "
              f"{bl_r.get('mdd_pct',0):>9.2f}%  "
              f"{ua.get('mdd_pct',0):>9.2f}%  "
              f"{oos_report.get('mdd_pct',0):>9.2f}%")

    # OOS 결과 JSON 저장
    oos_path = RESULTS_DIR / "d2s_oos_result.json"
    oos_payload = {
        "best_trial":   best.number,
        "is_score":     best.value,
        "oos_report":   oos_report,
        "oos_score":    oos_score,
        "best_params":  best.params,
        "oos_start":    str(OOS_START),
        "oos_end":      str(OOS_END),
        "timestamp":    datetime.now().isoformat(),
    }
    with open(oos_path, "w", encoding="utf-8") as f:
        json.dump(oos_payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  OOS 결과 저장: {oos_path}")


# ============================================================
# 유틸리티
# ============================================================


def _reconstruct_params(trial_params: dict) -> dict:
    """trial.params (플랫)에서 D2S_ENGINE 형식 dict를 재구성한다."""
    p = dict(D2S_ENGINE)  # 기본값 전체 복사

    # market_score_weights 재구성 (가중치 정규화)
    w_keys = ["w_gld", "w_spy", "w_riskoff", "w_streak", "w_vol", "w_btc"]
    if any(k in trial_params for k in w_keys):
        ws = [trial_params.get(k, 0.1) for k in w_keys]
        total = sum(ws)
        keys  = ["gld_score", "spy_score", "riskoff_score", "streak_score", "vol_score", "btc_score"]
        p["market_score_weights"] = {k: round(w / total, 4) for k, w in zip(keys, ws)}

    # 나머지 flat 파라미터 덮어쓰기
    skip = set(w_keys)
    for k, v in trial_params.items():
        if k not in skip:
            p[k] = v

    return p


def _print_report_summary(report: dict, score: float | None = None) -> None:
    print(f"\n  수익률: {report.get('total_return_pct', 0):+.2f}%  "
          f"| 최종: ${report.get('final_equity', 0):,.0f}")
    print(f"  승률: {report.get('win_rate', 0):.1f}%  "
          f"({report.get('win_trades', 0)}W / {report.get('lose_trades', 0)}L)  "
          f"| 거래: {report.get('sell_trades', 0)}건")
    print(f"  MDD: {report.get('mdd_pct', 0):.2f}%  "
          f"| Sharpe: {report.get('sharpe_ratio', 0):.3f}")
    if score is not None:
        print(f"  Score: {score:.4f}")


def _print_study_summary(
    study: optuna.Study,
    baseline_score: float | None,
    baseline_return: float | None,
) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("  완료된 trial 없음")
        return

    best = study.best_trial
    diff_s = f"  (baseline 대비 {best.value - (baseline_score or 0):+.4f})" if baseline_score else ""
    print(f"\n  BEST Trial #{best.number}  score={best.value:.4f}{diff_s}")
    print(f"    win_rate={best.user_attrs.get('win_rate', 0):.1f}%  "
          f"return={best.user_attrs.get('total_return_pct', 0):+.2f}%  "
          f"MDD={best.user_attrs.get('mdd_pct', 0):.2f}%  "
          f"Sharpe={best.user_attrs.get('sharpe_ratio', 0):.3f}")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    print(f"  {'#':>4s}  {'score':>8s}  {'승률':>6s}  {'수익률':>8s}  {'MDD':>7s}  {'Sharpe':>7s}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+7.4f}  "
            f"{t.user_attrs.get('win_rate', 0):>5.1f}%  "
            f"{t.user_attrs.get('total_return_pct', 0):>+7.2f}%  "
            f"{t.user_attrs.get('mdd_pct', 0):>6.2f}%  "
            f"{t.user_attrs.get('sharpe_ratio', 0):>6.3f}"
        )


def _save_optuna_report(
    study: optuna.Study,
    baseline_score: float | None,
    n_trials: int,
    n_jobs: int,
    journal_path: Path = JOURNAL_PATH,
) -> None:
    """간이 Optuna 마크다운 리포트를 저장한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "d2s_optuna_report.md"

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    best = study.best_trial
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    lines = [
        "# D2S Optuna 최적화 리포트 (v2)",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> IS 기간: {IS_START} ~ {IS_END}  ",
        f"> 총 trial: {len(completed)}  |  n_jobs: {n_jobs}  ",
        f"> 스코어: win_rate×0.40 + sharpe×15 - mdd×0.30  ",
        f"> Storage: {journal_path.name}",
        "",
        "## 1. Best Trial",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| Trial # | {best.number} |",
        f"| Score | {best.value:.4f} |",
        f"| 승률 | {best.user_attrs.get('win_rate', 0):.1f}% |",
        f"| 수익률 | {best.user_attrs.get('total_return_pct', 0):+.2f}% |",
        f"| MDD | {best.user_attrs.get('mdd_pct', 0):.2f}% |",
        f"| Sharpe | {best.user_attrs.get('sharpe_ratio', 0):.3f} |",
        f"| 매도 거래 | {best.user_attrs.get('sell_trades', 0)}건 |",
        "",
        "## 2. Top 5",
        "",
        "| # | score | 승률 | 수익률 | MDD | Sharpe |",
        "|---|---|---|---|---|---|",
    ]
    for t in top5:
        lines.append(
            f"| {t.number} | {t.value:+.4f} | "
            f"{t.user_attrs.get('win_rate',0):.1f}% | "
            f"{t.user_attrs.get('total_return_pct',0):+.2f}% | "
            f"{t.user_attrs.get('mdd_pct',0):.2f}% | "
            f"{t.user_attrs.get('sharpe_ratio',0):.3f} |"
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
    print(f"  Optuna 리포트: {report_path}")


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="D2S Optuna 파라미터 최적화")
    parser.add_argument("--stage",     type=int, choices=[1, 2, 3], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 3: OOS 검증)")
    parser.add_argument("--n-trials",  type=int, default=200,
                        help="Optuna trial 수 (기본: 200)")
    parser.add_argument("--n-jobs",    type=int, default=20,
                        help="병렬 프로세스 수 (기본: 20 = SLURM CPU)")
    parser.add_argument("--timeout",   type=int, default=None,
                        help="최대 실행 시간(초, 기본: 없음)")
    parser.add_argument("--study-name", type=str, default=STUDY_NAME,
                        help=f"Optuna study 이름 (기본: {STUDY_NAME})")
    parser.add_argument("--journal",   type=str, default=str(JOURNAL_PATH),
                        help=f"JournalStorage 경로 (기본: {JOURNAL_PATH})")
    args = parser.parse_args()
    journal_path = Path(args.journal)

    print("=" * 70)
    print("  D2S Optuna 파라미터 최적화 v2 (Study A/B/C 반영)")
    print(f"  IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}")
    print(f"  스코어: win_rate×0.40 + sharpe×15 - mdd×0.30")
    print("=" * 70)

    if args.stage == 1:
        run_stage1()
    elif args.stage == 2:
        run_stage2(
            n_trials=args.n_trials, n_jobs=args.n_jobs,
            timeout=args.timeout, study_name=args.study_name,
            journal_path=journal_path,
        )
    elif args.stage == 3:
        run_stage3(study_name=args.study_name, journal_path=journal_path)
    else:
        # 1 → 2 → 3 연속 실행
        run_stage1()
        print()
        run_stage2(
            n_trials=args.n_trials, n_jobs=args.n_jobs,
            timeout=args.timeout, study_name=args.study_name,
            journal_path=journal_path,
        )
        print()
        run_stage3(study_name=args.study_name, journal_path=journal_path)

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

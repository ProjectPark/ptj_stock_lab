#!/usr/bin/env python3
"""
Study 15 — recent_is r2 vs r3 OOS 검증 (bias 수정 후)
======================================================
목적: WF r2 (OOS +2.40%, Sharpe 1.427) vs r3 (OOS -0.04%, MDD -3.80%) 최종 선택.
     기존 Optuna 저널 best params → corrected engine (T+1 시가 체결)으로 재검증.

r2: study d2s_v3_recent_is_r2  (journal: data/optuna/d2s_v3_recent_is_r2.log)
r3: study d2s_v3_recent_is_r3  (journal: data/optuna/d2s_v3_recent_is_r3.log)

검증 기간:
  IS   : 2025-10-01 ~ 2026-01-31
  OOS  : 2026-02-01 ~ 2026-02-27

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_15_recent_is_compare.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN

OPTUNA_DIR  = _PROJECT_ROOT / "data" / "optuna"
RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START  = date(2025, 10, 1)
IS_END    = date(2026, 1, 31)
OOS_START = date(2026, 2, 1)
OOS_END   = date(2026, 2, 27)

PERIODS = [
    ("IS",  IS_START,  IS_END),
    ("OOS", OOS_START, OOS_END),
]

VARIANTS = [
    ("r2", "d2s_v3_recent_is_r2", OPTUNA_DIR / "d2s_v3_recent_is_r2.log"),
    ("r3", "d2s_v3_recent_is_r3", OPTUNA_DIR / "d2s_v3_recent_is_r3.log"),
]


def load_best_params(study_name: str, journal_path: Path) -> dict | None:
    """Optuna journal에서 best trial params 추출."""
    if not journal_path.exists():
        print(f"  [WARN] Journal 없음: {journal_path}")
        return None
    try:
        storage = JournalStorage(JournalFileBackend(str(journal_path)))
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"  [WARN] {study_name}: 완료된 trial 없음")
            return None
        best = study.best_trial
        print(f"    best #{best.number}  score={best.value:.4f}  "
              f"IS={best.user_attrs.get('is_return', 0):+.2f}%  "
              f"OOS={best.user_attrs.get('oos_return', 0):+.2f}%")
        return best.params
    except Exception as e:
        print(f"  [ERROR] {study_name} 로드 실패: {e}")
        return None


def reconstruct_params(trial_params: dict) -> dict:
    """trial params + base params → 전체 파라미터 딕셔너리."""
    p = dict(D2S_ENGINE_V3_NO_ROBN)
    # 가중치 파라미터 처리 (정규화)
    w_keys = ["w_gld", "w_spy", "w_riskoff", "w_streak", "w_vol", "w_btc"]
    if any(k in trial_params for k in w_keys):
        ws    = [trial_params.get(k, 0.1) for k in w_keys]
        total = sum(ws) or 1.0
        keys  = ["gld_score", "spy_score", "riskoff_score", "streak_score", "vol_score", "btc_score"]
        p["market_score_weights"] = {k: round(w / total, 4) for k, w in zip(keys, ws)}
    skip = set(w_keys)
    for k, v in trial_params.items():
        if k not in skip:
            p[k] = v
    return p


def run_backtest(params: dict, start: date, end: date) -> dict:
    bt = D2SBacktestV3(params=params, start_date=start, end_date=end)
    bt.run(verbose=False)
    return bt.report()


def main():
    print("=" * 70)
    print("  Study 15 — recent_is r2 vs r3 OOS 검증 (bias 수정 후)")
    print(f"  IS: {IS_START} ~ {IS_END}  |  OOS: {OOS_START} ~ {OOS_END}")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    all_results: list[dict] = []

    for label, study_name, journal_path in VARIANTS:
        print(f"\n  [{label}] {study_name}")
        trial_params = load_best_params(study_name, journal_path)
        if trial_params is None:
            print(f"  [{label}] SKIP")
            continue

        params = reconstruct_params(trial_params)

        for period_label, start, end in PERIODS:
            r = run_backtest(params, start, end)
            r["variant"] = label
            r["period"]  = period_label
            all_results.append(r)
            print(f"    {period_label}: {r['total_return_pct']:+.2f}%  "
                  f"Sharpe={r['sharpe_ratio']:.3f}  MDD={r['mdd_pct']:.1f}%  "
                  f"WR={r['win_rate']:.1f}%  trades={r.get('sell_trades', 0)}")

    # ── 비교 요약 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  비교 요약")
    print("=" * 70)
    header = f"  {'구분':<6} {'IS%':>8} {'OOS%':>8} {'IS_Shp':>8} {'OOS_Shp':>9} {'OOS_MDD':>9} {'OOS_WR':>8}"
    print(header)
    print("  " + "-" * 60)

    winner = None
    best_oos = -9999.0

    for lbl in ["r2", "r3"]:
        is_r  = next((r for r in all_results if r["variant"] == lbl and r["period"] == "IS"),  None)
        oos_r = next((r for r in all_results if r["variant"] == lbl and r["period"] == "OOS"), None)
        if is_r and oos_r:
            print(
                f"  {lbl:<6} "
                f"{is_r['total_return_pct']:>7.2f}% "
                f"{oos_r['total_return_pct']:>7.2f}% "
                f"{is_r['sharpe_ratio']:>8.3f} "
                f"{oos_r['sharpe_ratio']:>9.3f} "
                f"{oos_r['mdd_pct']:>8.1f}% "
                f"{oos_r['win_rate']:>7.1f}%"
            )
            if oos_r['total_return_pct'] > best_oos:
                best_oos = oos_r['total_return_pct']
                winner = lbl

    if winner:
        print(f"\n  → OOS 수익률 기준 우승: {winner}")

    # ── 저장 ────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"study_15_recent_is_compare_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

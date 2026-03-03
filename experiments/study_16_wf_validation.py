#!/usr/bin/env python3
"""
Study 16 — Study 13 Best Params Walk-Forward 검증
===================================================
목적: Study 13 #466 best params가 과적합인지 확인.
     고정 파라미터를 여러 OOS 창에 적용해 일관성 검증.

설계:
  - 파라미터: d2s_v3_norobn_s13 study best trial (#466) 고정
  - IS 창: Expanding (기준 시작일 고정)
  - OOS 창: 3개월 Rolling

창 구성 (no-ROBN, market_daily_3y.parquet):
  W1: IS 2024-09-18~2025-02-28 | OOS 2025-03-01~2025-05-31
  W2: IS 2024-09-18~2025-05-31 | OOS 2025-06-01~2025-08-31
  W3: IS 2024-09-18~2025-08-31 | OOS 2025-09-01~2025-11-30
  W4: IS 2024-09-18~2025-11-30 | OOS 2025-12-01~2026-02-28
  FULL: 2024-09-18~2026-02-28 (단일 전체 검증)

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_16_wf_validation.py
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
RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "optimization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STUDY_NAME   = "d2s_v3_norobn_s13"
JOURNAL_PATH = OPTUNA_DIR / "d2s_v3_norobn_s13.log"

DATA_PATH = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily_3y.parquet"

# ── OOS 창 정의 (Expanding IS + Rolling OOS 3개월) ─────────────
WINDOWS = [
    ("W1", date(2024, 9, 18), date(2025, 2, 28), date(2025, 3,  1), date(2025, 5, 31)),
    ("W2", date(2024, 9, 18), date(2025, 5, 31), date(2025, 6,  1), date(2025, 8, 31)),
    ("W3", date(2024, 9, 18), date(2025, 8, 31), date(2025, 9,  1), date(2025, 11,30)),
    ("W4", date(2024, 9, 18), date(2025,11, 30), date(2025, 12, 1), date(2026, 2, 28)),
]
FULL_WINDOW = ("FULL", date(2024, 9, 18), date(2026, 2, 28))


def load_best_params(study_name: str, journal_path: Path) -> tuple[int, dict] | None:
    """Optuna journal에서 best trial params + number 반환."""
    if not journal_path.exists():
        print(f"  [ERROR] Journal 없음: {journal_path}")
        return None
    try:
        storage = JournalStorage(JournalFileBackend(str(journal_path)))
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"  [ERROR] 완료된 trial 없음")
            return None
        best = study.best_trial
        print(f"  Best Trial #{best.number}  score={best.value:.4f}")
        print(f"    IS={best.user_attrs.get('is_return', 0):+.2f}%  "
              f"OOS={best.user_attrs.get('oos_return', 0):+.2f}%  "
              f"IS_Shp={best.user_attrs.get('is_sharpe', 0):.3f}  "
              f"OOS_Shp={best.user_attrs.get('oos_sharpe', 0):.3f}")
        return best.number, best.params
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def reconstruct_params(trial_params: dict) -> dict:
    """trial params + base params 병합."""
    p = dict(D2S_ENGINE_V3_NO_ROBN)
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
    bt = D2SBacktestV3(
        params=params, start_date=start, end_date=end,
        data_path=DATA_PATH,
    )
    bt.run(verbose=False)
    return bt.report()


def main():
    print("=" * 70)
    print("  Study 16 — Study 13 Best Params Walk-Forward 검증")
    print(f"  Study: {STUDY_NAME}")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # ── best params 로드 ──────────────────────────────────────
    print(f"\n  [1] {STUDY_NAME} best params 로드")
    result = load_best_params(STUDY_NAME, JOURNAL_PATH)
    if result is None:
        print("  [ABORT] params 로드 실패")
        return

    best_num, trial_params = result
    params = reconstruct_params(trial_params)

    # ── 창별 백테스트 ─────────────────────────────────────────
    print(f"\n  [2] Walk-Forward 창별 검증 (고정 파라미터 #{ best_num})")
    print(f"  {'창':>6}  {'IS 기간':>25}  {'OOS 기간':>25}  "
          f"{'IS%':>7}  {'OOS%':>7}  {'OOS_Shp':>8}  {'OOS_MDD':>8}")
    print("  " + "-" * 95)

    all_results = []

    for win_id, is_start, is_end, oos_start, oos_end in WINDOWS:
        r_is  = run_backtest(params, is_start, is_end)
        r_oos = run_backtest(params, oos_start, oos_end)

        row = {
            "window": win_id,
            "is_start": str(is_start), "is_end": str(is_end),
            "oos_start": str(oos_start), "oos_end": str(oos_end),
            "is_return": r_is["total_return_pct"],
            "oos_return": r_oos["total_return_pct"],
            "is_sharpe":  r_is["sharpe_ratio"],
            "oos_sharpe": r_oos["sharpe_ratio"],
            "oos_mdd":    r_oos["mdd_pct"],
            "oos_wr":     r_oos["win_rate"],
            "oos_trades": r_oos.get("sell_trades", 0),
        }
        all_results.append(row)

        print(f"  {win_id:>6}  {str(is_start)+'~'+str(is_end):>25}  "
              f"{str(oos_start)+'~'+str(oos_end):>25}  "
              f"{r_is['total_return_pct']:>6.1f}%  "
              f"{r_oos['total_return_pct']:>6.1f}%  "
              f"{r_oos['sharpe_ratio']:>8.3f}  "
              f"{r_oos['mdd_pct']:>7.1f}%")

    # ── FULL 기간 ─────────────────────────────────────────────
    print(f"\n  [3] FULL 기간 검증")
    win_id, full_start, full_end = FULL_WINDOW
    r_full = run_backtest(params, full_start, full_end)
    full_row = {
        "window": win_id,
        "start": str(full_start), "end": str(full_end),
        "return": r_full["total_return_pct"],
        "sharpe": r_full["sharpe_ratio"],
        "mdd": r_full["mdd_pct"],
        "win_rate": r_full["win_rate"],
        "sell_trades": r_full.get("sell_trades", 0),
    }
    print(f"  FULL {full_start}~{full_end}: "
          f"{r_full['total_return_pct']:+.2f}%  "
          f"Sharpe={r_full['sharpe_ratio']:.3f}  "
          f"MDD={r_full['mdd_pct']:.1f}%  "
          f"WR={r_full['win_rate']:.1f}%  "
          f"trades={r_full.get('sell_trades', 0)}")

    # ── 요약 통계 ─────────────────────────────────────────────
    oos_returns = [r["oos_return"] for r in all_results]
    oos_positive = sum(1 for x in oos_returns if x > 0)
    avg_oos = sum(oos_returns) / len(oos_returns) if oos_returns else 0

    print("\n" + "=" * 70)
    print("  WF 요약")
    print("=" * 70)
    print(f"  OOS 창 수익 양수: {oos_positive}/{len(all_results)}")
    print(f"  OOS 평균 수익률: {avg_oos:+.2f}%")
    print(f"  FULL 수익률: {r_full['total_return_pct']:+.2f}%  Sharpe: {r_full['sharpe_ratio']:.3f}")

    if oos_positive == len(all_results):
        verdict = "✅ 전 창 OOS 양수 — 파라미터 견고성 확인"
    elif oos_positive >= len(all_results) // 2 + 1:
        verdict = "⚠️  과반 창 OOS 양수 — 부분 견고성"
    else:
        verdict = "❌ 과반 창 OOS 음수 — 과적합 가능성"
    print(f"  판정: {verdict}")

    # ── 저장 ──────────────────────────────────────────────────
    out = {
        "study": STUDY_NAME,
        "best_trial": best_num,
        "windows": all_results,
        "full": full_row,
        "summary": {
            "oos_positive": oos_positive,
            "oos_total": len(all_results),
            "avg_oos_return": avg_oos,
            "verdict": verdict,
        },
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / f"study_16_wf_validation_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

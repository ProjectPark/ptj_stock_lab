#!/usr/bin/env python3
"""
Study 13 — v5 s2 best trial OOS 검증
======================================
목적: ptj_v5_s2 study best trial (#239, value=-2.4417)이
     실제 OOS 구간에서 baseline 대비 개선됐는지 검증.

검증 기간:
  IS   : 2025-02-18 ~ 2025-12-31  (optimizer 학습 구간)
  OOS  : 2026-01-01 ~ 2026-02-27  (미래 검증 구간)
  FULL : 2025-02-18 ~ 2026-02-27

비교 대상:
  baseline  : 현재 config.py 기본값
  s2_best   : ptj_v5_s2 study 최고 trial 파라미터

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_v5_s2_oos.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna

from simulation.optimizers.optimize_v5_optuna import V5Optimizer
from simulation.optimizers.optimizer_base import extract_metrics

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = _PROJECT_ROOT / "data" / "optuna" / "v5_opt.db"
STUDY_NAME = "ptj_v5_s2"
OUT_FILE = RESULTS_DIR / "study_13_v5_s2_oos_20260227.json"

# ── 기간 정의 ──────────────────────────────────────────────
IS_START   = date(2025, 2, 18)
IS_END     = date(2025, 12, 31)
OOS_START  = date(2026, 1, 1)
OOS_END    = date(2026, 2, 27)
FULL_START = date(2025, 2, 18)
FULL_END   = date(2026, 2, 27)

PERIODS = [
    ("IS",   IS_START,   IS_END),
    ("OOS",  OOS_START,  OOS_END),
    ("FULL", FULL_START, FULL_END),
]


def run_backtest(opt: V5Optimizer, params: dict, start: date, end: date) -> dict:
    """파라미터로 지정 기간 백테스트를 실행하고 결과 dict를 반환한다."""
    result = opt.run_single_trial(params, start_date=start, end_date=end)
    return result.to_dict()


def load_best_trial_params(db_path: Path, study_name: str) -> tuple[dict, int, float]:
    """Optuna DB에서 best trial의 params, number, value를 로드한다."""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best = study.best_trial
    print(f"  Study: {study_name}")
    print(f"  Best trial: #{best.number}  value={best.value:+.4f}")
    print(f"  총 완료 trials: {len(study.trials)}")
    return best.params, best.number, best.value


def main() -> None:
    print("=" * 70)
    print("  Study 13 — PTJ v5 s2 Best Trial OOS 검증")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n[ERROR] Optuna DB 없음: {DB_PATH}")
        sys.exit(1)

    # ── 1. Optuna DB에서 best trial 로드 ───────────────────
    print(f"\n[1] Optuna DB 로드: {DB_PATH}")
    best_params, best_number, best_value = load_best_trial_params(DB_PATH, STUDY_NAME)

    # s2는 default variant → gap_max=4.0
    opt = V5Optimizer(gap_max=4.0, variant="default")

    # baseline 파라미터 (현재 config.py 값)
    baseline_params = opt.get_baseline_params()

    # best trial 파라미터는 탐색 공간 키만 포함 → baseline으로 채우기
    full_best_params = {**baseline_params, **best_params}

    print(f"\n  [비교 파라미터 요약 — baseline vs best]")
    changed = {k: (baseline_params.get(k), v)
               for k, v in full_best_params.items()
               if baseline_params.get(k) != v}
    for k, (b, s) in changed.items():
        print(f"    {k}: {b} → {s}")

    # ── 2. 기간별 백테스트 실행 ────────────────────────────
    results = {}

    for label, params_dict, tag in [
        ("baseline", baseline_params, "baseline"),
        ("s2_best",  full_best_params, f"trial#{best_number}"),
    ]:
        print(f"\n[2] {label} ({tag}) 백테스트 실행 중...")
        period_results = {}
        for period_name, start, end in PERIODS:
            print(f"  {period_name} ({start} ~ {end}) ...", end=" ", flush=True)
            r = run_backtest(opt, params_dict, start, end)
            period_results[period_name] = r
            print(f"{r['total_return_pct']:+.2f}%  Sharpe {r['sharpe']:+.4f}  MDD -{r['mdd']:.2f}%")
        results[label] = period_results

    # ── 3. 결과 비교 출력 ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  [결과 비교]")
    print("=" * 70)
    print(f"  {'기간':<8} {'지표':<18} {'baseline':>12} {'s2_best':>12} {'Δ':>10}")
    print("  " + "-" * 62)

    for period_name in ["IS", "OOS", "FULL"]:
        bl = results["baseline"][period_name]
        sb = results["s2_best"][period_name]
        for metric, label in [
            ("total_return_pct", "수익률(%)"),
            ("mdd",              "MDD(%)"),
            ("sharpe",           "Sharpe"),
            ("win_rate",         "승률(%)"),
            ("total_sells",      "매도횟수"),
            ("sideways_blocks",  "횡보차단"),
            ("cb_buy_blocks",    "CB차단"),
        ]:
            b_val = bl.get(metric, 0)
            s_val = sb.get(metric, 0)
            delta = s_val - b_val
            sign = "+" if delta > 0 else ""
            print(f"  {period_name:<8} {label:<18} {b_val:>12.4g} {s_val:>12.4g} {sign}{delta:>9.4g}")
        print("  " + "-" * 62)

    # ── 4. 핵심 판정 ───────────────────────────────────────
    oos_bl  = results["baseline"]["OOS"]["total_return_pct"]
    oos_sb  = results["s2_best"]["OOS"]["total_return_pct"]
    oos_delta = oos_sb - oos_bl

    print(f"\n  [핵심 판정]")
    print(f"  OOS baseline : {oos_bl:+.2f}%")
    print(f"  OOS s2_best  : {oos_sb:+.2f}%")
    print(f"  개선폭       : {oos_delta:+.2f}%p")

    if oos_sb > 0:
        verdict = "✅ OOS 양전 — v5 파라미터 확정 검토"
    elif oos_delta > 3.0:
        verdict = "🟡 OOS 음전이나 baseline 대비 +3%p 이상 개선 — 추가 탐색 검토"
    elif oos_delta > 0:
        verdict = "🟠 소폭 개선 — 추가 Optuna 탐색 or 규칙 재검토 필요"
    else:
        verdict = "🔴 개선 없음 — v5 엔진 근본 문제 → 규칙 재설계 필요"

    print(f"  판정: {verdict}")

    # ── 5. JSON 저장 ───────────────────────────────────────
    output = {
        "study": STUDY_NAME,
        "best_trial_number": best_number,
        "best_trial_value": best_value,
        "changed_params": {k: {"baseline": b, "best": s} for k, (b, s) in changed.items()},
        "periods": {
            "IS":   {"start": str(IS_START),   "end": str(IS_END)},
            "OOS":  {"start": str(OOS_START),  "end": str(OOS_END)},
            "FULL": {"start": str(FULL_START),  "end": str(FULL_END)},
        },
        "results": results,
        "verdict": verdict,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  결과 저장: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()

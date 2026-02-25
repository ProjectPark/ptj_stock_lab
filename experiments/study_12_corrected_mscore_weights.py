#!/usr/bin/env python3
"""
Study 12 — market_score weights 검증 (look-ahead bias 수정 후)
===============================================================
목적: Study 10 수정 후에도 v3_current weights가 OOS 우위를 유지하는지 확인.
     유지 시 → params_d2s.py weights 확정; 역전 시 → weights 재검토.

4가지 weights 스킴 (Study 9B와 동일):
  v3_current    : 현재 D2S_ENGINE_V3_NO_ROBN 기본 weights
  v3_no_gld     : GLD 제거 후 재분배 (Study 9B trial_162 계승)
  equal_weight  : 균등 (1/6)
  v3_spy_only   : SPY + streak 만 (위험 자산 제거)

검증 기간:
  IS  : 2024-09-18 ~ 2025-05-31
  OOS : 2025-06-01 ~ 2026-02-17

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_12_corrected_mscore_weights.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 기간 정의 ──────────────────────────────────────────────
IS_START   = date(2024, 9, 18)
IS_END     = date(2025, 5, 31)
OOS_START  = date(2025, 6, 1)
OOS_END    = date(2026, 2, 17)
FULL_START = date(2024, 9, 18)
FULL_END   = date(2026, 2, 17)

PERIODS = [
    ("IS",   IS_START,   IS_END),
    ("OOS",  OOS_START,  OOS_END),
    ("FULL", FULL_START, FULL_END),
]

# ── weights 스킴 정의 ─────────────────────────────────────

V3_CURRENT_WEIGHTS = dict(D2S_ENGINE_V3_NO_ROBN["market_score_weights"])

# GLD 제거 후 나머지에 균등 재분배
_remaining_keys = ["spy_score", "riskoff_score", "streak_score", "vol_score", "btc_score"]
V3_NO_GLD_WEIGHTS = {k: 1.0 / len(_remaining_keys) for k in _remaining_keys}
V3_NO_GLD_WEIGHTS["gld_score"] = 0.0

EQUAL_WEIGHTS = {k: 1.0 / 6.0 for k in [
    "gld_score", "spy_score", "riskoff_score",
    "streak_score", "vol_score", "btc_score",
]}

# SPY + streak만 (위험 자산 노이즈 제거)
V3_SPY_ONLY_WEIGHTS = {
    "gld_score":     0.0,
    "spy_score":     0.50,
    "riskoff_score": 0.0,
    "streak_score":  0.50,
    "vol_score":     0.0,
    "btc_score":     0.0,
}

SCHEMES = [
    ("v3_current",   V3_CURRENT_WEIGHTS,  "현재 v3 weights (기준)"),
    ("v3_no_gld",    V3_NO_GLD_WEIGHTS,   "GLD 제거 후 균등 재분배"),
    ("equal_weight", EQUAL_WEIGHTS,       "모든 신호 균등 (1/6)"),
    ("v3_spy_only",  V3_SPY_ONLY_WEIGHTS, "SPY+streak만 (0.5/0.5)"),
]


def make_params(weights: dict) -> dict:
    """D2S_ENGINE_V3_NO_ROBN 기반으로 market_score_weights만 오버라이드."""
    p = dict(D2S_ENGINE_V3_NO_ROBN)
    p["market_score_weights"] = dict(weights)
    return p


def run_backtest(label: str, params: dict, start: date, end: date) -> dict:
    bt = D2SBacktestV3(params=params, start_date=start, end_date=end)
    bt.run(verbose=False)
    r = bt.report()
    r["label"] = label
    return r


def main():
    print("=" * 80)
    print("  Study 12 — market_score weights 검증 (look-ahead bias 수정 후)")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)

    # 현재 weights 출력
    print("\n  현재 v3 weights:")
    for k, v in V3_CURRENT_WEIGHTS.items():
        print(f"    {k:<16}: {v:.4f}")

    all_results: list[dict] = []
    period_results: dict[str, dict[str, dict]] = {}  # period → scheme → result

    for period_label, start, end in PERIODS:
        print(f"\n  [{period_label}] {start} ~ {end}")
        period_results[period_label] = {}

        for scheme_name, weights, desc in SCHEMES:
            r = run_backtest(scheme_name, make_params(weights), start, end)
            r["period"] = period_label
            r["desc"] = desc
            period_results[period_label][scheme_name] = r
            all_results.append(r)
            print(f"    {scheme_name:<14}: 수익률={r['total_return_pct']:+.2f}%  "
                  f"MDD={r['mdd_pct']:.2f}%  Sharpe={r['sharpe_ratio']:.3f}")

    # ── 결과 비교 테이블 ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  결과 비교 테이블 (IS / OOS / FULL)")
    print("=" * 80)

    header = f"  {'스킴':<15} {'IS 수익률':>10} {'OOS 수익률':>10} {'FULL 수익률':>11} {'IS Sharpe':>10} {'OOS Sharpe':>10}"
    print(header)
    print("  " + "-" * 70)

    for scheme_name, _, _ in SCHEMES:
        is_r   = period_results["IS"][scheme_name]
        oos_r  = period_results["OOS"][scheme_name]
        full_r = period_results["FULL"][scheme_name]
        mark = " ★" if scheme_name == "v3_current" else ""
        print(
            f"  {scheme_name:<15}"
            f"{is_r['total_return_pct']:>9.2f}% "
            f"{oos_r['total_return_pct']:>9.2f}% "
            f"{full_r['total_return_pct']:>10.2f}% "
            f"{is_r['sharpe_ratio']:>9.3f} "
            f"{oos_r['sharpe_ratio']:>9.3f}"
            f"{mark}"
        )

    # ── 결론 분석 ─────────────────────────────────────────────────
    oos_by_scheme = {s: period_results["OOS"][s]["total_return_pct"] for s, _, _ in SCHEMES}
    best_scheme = max(oos_by_scheme, key=oos_by_scheme.get)
    v3_current_rank = sorted(oos_by_scheme, key=oos_by_scheme.get, reverse=True).index("v3_current") + 1

    print(f"\n  OOS 최우수: {best_scheme} ({oos_by_scheme[best_scheme]:+.2f}%)")
    print(f"  v3_current OOS 순위: {v3_current_rank}/{len(SCHEMES)}")

    if v3_current_rank <= 2:
        print("  → ✅ v3_current OOS 우위 유지 — params_d2s.py weights 확정")
    else:
        print(f"  → ⚠️ v3_current OOS 우위 역전 (best: {best_scheme}) — weights 재검토 권장")

    # ── IS/OOS 과적합 여부 체크 ──────────────────────────────────
    is_best = max(period_results["IS"], key=lambda s: period_results["IS"][s]["total_return_pct"])
    print(f"\n  IS 최우수: {is_best}")
    if is_best != best_scheme:
        print("  → ℹ️ IS/OOS 최우수 스킴 불일치 — 견고성 확인 필요")

    # ── 저장 ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"study_12_corrected_weights_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

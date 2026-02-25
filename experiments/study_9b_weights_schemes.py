#!/usr/bin/env python3
"""
Study 9B — market_score weights 스킴 4종 비교
=============================================
목적: Study 9 Optuna trial #162 최적 weights가 실제로 더 나은지,
다양한 weights 스킴과 비교해서 견고성(robustness) 검증.
IS/OOS/전체 3개 기간으로 각각 평가.

4가지 weights 스킴:
  v3_current:    현재 D2S_ENGINE_V3_NO_ROBN 기본 weights
  trial_162:     Study 9 Optuna 최적 (trial #162)
  equal:         균등 (1/6)
  riskoff_heavy: riskoff_score 최대 가중 (0.40)
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

TRIAL_162_WEIGHTS = {
    "gld_score":     0.2057,
    "spy_score":     0.0743,
    "riskoff_score": 0.2323,
    "streak_score":  0.2369,
    "vol_score":     0.1592,
    "btc_score":     0.0916,
}

EQUAL_WEIGHTS = {k: 1.0 / 6.0 for k in [
    "gld_score", "spy_score", "riskoff_score",
    "streak_score", "vol_score", "btc_score",
]}

RISKOFF_HEAVY_WEIGHTS = {
    "gld_score":     0.15,
    "spy_score":     0.05,
    "riskoff_score": 0.40,
    "streak_score":  0.25,
    "vol_score":     0.10,
    "btc_score":     0.05,
}

SCHEMES = [
    ("v3_current",    V3_CURRENT_WEIGHTS),
    ("trial_162",     TRIAL_162_WEIGHTS),
    ("equal",         EQUAL_WEIGHTS),
    ("riskoff_heavy", RISKOFF_HEAVY_WEIGHTS),
]


def make_params(weights: dict) -> dict:
    """D2S_ENGINE_V3_NO_ROBN 기반으로 market_score_weights만 오버라이드."""
    p = dict(D2S_ENGINE_V3_NO_ROBN)
    p["market_score_weights"] = weights
    return p


def run_backtest(label: str, params: dict, start: date, end: date) -> dict:
    """단일 백테스트 실행 후 report dict 반환."""
    bt = D2SBacktestV3(
        params=params,
        start_date=start,
        end_date=end,
    )
    bt.run(verbose=False)
    return bt.report()


def main():
    print("=" * 70)
    print("  Study 9B: market_score weights 스킴 4종 비교")
    print(f"  IS:   {IS_START} ~ {IS_END}")
    print(f"  OOS:  {OOS_START} ~ {OOS_END}")
    print(f"  FULL: {FULL_START} ~ {FULL_END}")
    print("=" * 70)

    # weights 확인 출력
    for name, w in SCHEMES:
        w_str = ", ".join(f"{k.replace('_score','')}={v:.3f}" for k, v in w.items())
        print(f"  {name:16s}: {w_str}")
    print()

    # 3기간 x 4스킴 = 12회 백테스트
    all_results = {}

    for period_label, start, end in PERIODS:
        print(f"\n--- {period_label} ({start} ~ {end}) ---")
        for scheme_name, weights in SCHEMES:
            params = make_params(weights)
            rpt = run_backtest(scheme_name, params, start, end)
            key = f"{period_label}_{scheme_name}"
            all_results[key] = rpt
            print(
                f"  {scheme_name:16s}: "
                f"수익률={rpt['total_return_pct']:+7.1f}%  "
                f"Sharpe={rpt['sharpe_ratio']:.3f}  "
                f"MDD={rpt['mdd_pct']:.1f}%  "
                f"승률={rpt['win_rate']:.1f}%  "
                f"청산={rpt['sell_trades']}건"
            )

    # ── 요약 테이블 ────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Study 9B 요약 비교")
    print("=" * 90)
    header = (
        f"  {'스킴':16s}  "
        f"{'IS 수익률':>10}  {'IS Sharpe':>10}  "
        f"{'OOS 수익률':>10}  {'OOS Sharpe':>10}  "
        f"{'전체 수익률':>10}"
    )
    print(header)
    print(f"  {'-' * 80}")

    for scheme_name, _ in SCHEMES:
        is_r  = all_results.get(f"IS_{scheme_name}", {})
        oos_r = all_results.get(f"OOS_{scheme_name}", {})
        full_r = all_results.get(f"FULL_{scheme_name}", {})
        print(
            f"  {scheme_name:16s}  "
            f"{is_r.get('total_return_pct', 0):>9.1f}%  "
            f"{is_r.get('sharpe_ratio', 0):>10.3f}  "
            f"{oos_r.get('total_return_pct', 0):>9.1f}%  "
            f"{oos_r.get('sharpe_ratio', 0):>10.3f}  "
            f"{full_r.get('total_return_pct', 0):>9.1f}%"
        )
    print("=" * 90)

    # ── 결과 저장 ──────────────────────────────────────────
    output = {
        "study": "study_9b_weights_schemes",
        "run_date": datetime.now().isoformat(),
        "periods": {
            "IS":   {"start": str(IS_START),   "end": str(IS_END)},
            "OOS":  {"start": str(OOS_START),  "end": str(OOS_END)},
            "FULL": {"start": str(FULL_START), "end": str(FULL_END)},
        },
        "schemes": {name: weights for name, weights in SCHEMES},
        "results": all_results,
    }
    out_path = RESULTS_DIR / f"study_9b_weights_schemes_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()

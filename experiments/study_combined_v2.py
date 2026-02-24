#!/usr/bin/env python3
"""
Study 1+4 통합 백테스트 — 복합 효과 검증
=========================================================
Study 1 결론: score≥0.70 + bb_pct≤0.30 (F1+F2) → OOS +14.29%p 개선
Study 4 결론: hold_days=4 → OOS 최고, hold_days=5 → FULL Sharpe 최고

두 스터디를 통합한 복합 효과 검증 (상호작용 유무 확인).

시나리오 (6개 × 3 period = 18 runs):
  BASELINE  : score≥0.65, bb=None, hold_days=7 (현재 기준)
  S1_ONLY   : score≥0.70 + bb≤0.30, hold_days=7
  S4_HD4    : score≥0.65, bb=None, hold_days=4
  S4_HD5    : score≥0.65, bb=None, hold_days=5
  S14_HD4   : score≥0.70 + bb≤0.30, hold_days=4
  S14_HD5   : score≥0.70 + bb≤0.30, hold_days=5

기간: FULL(2025-03-03~2026-02-17), IS(~2025-09-30), OOS(2025-10-01~)
멀티프로세싱: Pool(N_JOBS=8) — 18 combos

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_combined_v2.py
"""
from __future__ import annotations

import csv
import os
import sys
from copy import deepcopy
from datetime import date
from multiprocessing import Pool
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# experiments/에서 StudyD2SEngine, StudyD2SBacktestV2 재사용
_EXPERIMENTS_DIR = str(_PROJECT_ROOT / "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from study_1layer_backtest import StudyD2SBacktestV2  # noqa: E402

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2


# ============================================================
# 시나리오 정의
# ============================================================

SCENARIOS: dict[str, dict] = {
    "BASELINE": {
        "label": "v2 기준 (score≥0.65, bb=None, hd=7)",
        "params": {},
    },
    "S1_ONLY": {
        "label": "S1: score≥0.70 + bb≤0.30, hd=7",
        "params": {
            "score_contrarian_min": 0.70,
            "score_riskoff_min": 0.65,
            "bb_entry_hard_max": 0.30,
        },
    },
    "S4_HD4": {
        "label": "S4: hd=4 (OOS 최고)",
        "params": {
            "optimal_hold_days_max": 4,
        },
    },
    "S4_HD5": {
        "label": "S4: hd=5 (FULL Sharpe 최고)",
        "params": {
            "optimal_hold_days_max": 5,
        },
    },
    "S14_HD4": {
        "label": "S1+S4: score≥0.70 + bb≤0.30 + hd=4",
        "params": {
            "score_contrarian_min": 0.70,
            "score_riskoff_min": 0.65,
            "bb_entry_hard_max": 0.30,
            "optimal_hold_days_max": 4,
        },
    },
    "S14_HD5": {
        "label": "S1+S4: score≥0.70 + bb≤0.30 + hd=5",
        "params": {
            "score_contrarian_min": 0.70,
            "score_riskoff_min": 0.65,
            "bb_entry_hard_max": 0.30,
            "optimal_hold_days_max": 5,
        },
    },
}


# ============================================================
# 단일 시나리오 실행
# ============================================================

def run_scenario(
    name: str,
    config: dict,
    start_date: date,
    end_date: date,
) -> dict:
    params = deepcopy(D2S_ENGINE_V2)
    params.update(config["params"])

    bt = StudyD2SBacktestV2(
        params=params,
        start_date=start_date,
        end_date=end_date,
        use_fees=True,
    )
    bt.run(verbose=False)
    r = bt.report()

    forced_liq = sum(
        1 for t in bt.trades
        if t.side == "SELL" and "hold_days" in t.reason
    )
    buy_scores = [t.score for t in bt.trades if t.side == "BUY"]
    avg_buy_score = sum(buy_scores) / len(buy_scores) if buy_scores else 0.0

    return {
        "scenario": name,
        "label": config["label"],
        "total_return": r["total_return_pct"],
        "win_rate": r["win_rate"],
        "mdd": r["mdd_pct"],
        "sharpe": r["sharpe_ratio"],
        "buy_trades": r["buy_trades"],
        "sell_trades": r["sell_trades"],
        "win_trades": r["win_trades"],
        "lose_trades": r["lose_trades"],
        "avg_pnl": r["avg_pnl_pct"],
        "total_pnl": r["total_pnl"],
        "forced_liquidations": forced_liq,
        "r17_count": bt._r17_count,
        "r18_count": bt._r18_count,
        "avg_buy_score": avg_buy_score,
    }


# ============================================================
# 기간 정의 & 병렬 실행
# ============================================================

DATE_FULL_START = date(2025, 3, 3)
DATE_FULL_END   = date(2026, 2, 17)
DATE_IS_END     = date(2025, 9, 30)
DATE_OOS_START  = date(2025, 10, 1)

N_JOBS = int(os.environ.get("N_JOBS", "20"))


def _run_task(args: tuple) -> dict:
    name, config, period_label, start, end = args
    r = run_scenario(name, config, start, end)
    r["period"] = period_label
    return r


# ============================================================
# 비교 출력
# ============================================================

def print_comparison(
    results_full: list[dict],
    results_is: list[dict],
    results_oos: list[dict],
) -> None:
    print("\n" + "=" * 90)
    print("  Study 1+4 통합 백테스트 — 복합 효과 검증")
    print("=" * 90)

    # FULL 성과 테이블
    base_full = next(r for r in results_full if r["scenario"] == "BASELINE")
    print("\n[FULL 기간 성과]")
    hdr = (
        f"  {'시나리오':<12} {'수익률':>8} {'승률':>6} {'MDD':>7}"
        f" {'Sharpe':>7} {'매수':>5} {'강제청산':>6} {'Δ수익률':>8}"
    )
    print(hdr)
    print("  " + "-" * 72)
    for r in results_full:
        delta = r["total_return"] - base_full["total_return"]
        delta_str = f"{delta:+.2f}%p" if r["scenario"] != "BASELINE" else "  —"
        print(
            f"  {r['scenario']:<12}"
            f" {r['total_return']:>+7.2f}%"
            f" {r['win_rate']:>5}%"
            f" {r['mdd']:>6.1f}%"
            f" {r['sharpe']:>7.3f}"
            f" {r['buy_trades']:>5}건"
            f" {r['forced_liquidations']:>5}건"
            f" {delta_str:>8}"
        )

    # IS / OOS 안정성 테이블
    base_is  = next(r for r in results_is  if r["scenario"] == "BASELINE")
    base_oos = next(r for r in results_oos if r["scenario"] == "BASELINE")

    print("\n[IS / OOS 안정성 분석]")
    hdr2 = (
        f"  {'시나리오':<12} {'IS 수익률':>10} {'OOS 수익률':>11}"
        f" {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Δ':>8}"
    )
    print(hdr2)
    print("  " + "-" * 68)
    for name in SCENARIOS:
        r_is  = next(r for r in results_is  if r["scenario"] == name)
        r_oos = next(r for r in results_oos if r["scenario"] == name)
        oos_delta = r_oos["total_return"] - base_oos["total_return"]
        oos_delta_str = f"{oos_delta:+.2f}%p" if name != "BASELINE" else "  —"
        print(
            f"  {name:<12}"
            f" {r_is['total_return']:>+9.2f}%"
            f" {r_oos['total_return']:>+10.2f}%"
            f" {r_is['sharpe']:>10.3f}"
            f" {r_oos['sharpe']:>10.3f}"
            f" {oos_delta_str:>8}"
        )

    # 복합 효과 검증 (상호작용)
    print("\n[복합 효과 분석 — 상호작용 유무]")
    s1_oos_d  = next(r["total_return"] for r in results_oos if r["scenario"] == "S1_ONLY")  - base_oos["total_return"]
    s4h4_oos_d = next(r["total_return"] for r in results_oos if r["scenario"] == "S4_HD4") - base_oos["total_return"]
    s4h5_oos_d = next(r["total_return"] for r in results_oos if r["scenario"] == "S4_HD5") - base_oos["total_return"]
    s14h4_oos_d = next(r["total_return"] for r in results_oos if r["scenario"] == "S14_HD4") - base_oos["total_return"]
    s14h5_oos_d = next(r["total_return"] for r in results_oos if r["scenario"] == "S14_HD5") - base_oos["total_return"]

    print(f"  S1_ONLY  OOS Δ: {s1_oos_d:+.2f}%p")
    print(f"  S4_HD4   OOS Δ: {s4h4_oos_d:+.2f}%p")
    print(f"  S4_HD5   OOS Δ: {s4h5_oos_d:+.2f}%p")
    print(f"  S14_HD4  OOS Δ: {s14h4_oos_d:+.2f}%p  (예상: {s1_oos_d + s4h4_oos_d:+.2f}%p, 실제-예상: {s14h4_oos_d - (s1_oos_d + s4h4_oos_d):+.2f}%p)")
    print(f"  S14_HD5  OOS Δ: {s14h5_oos_d:+.2f}%p  (예상: {s1_oos_d + s4h5_oos_d:+.2f}%p, 실제-예상: {s14h5_oos_d - (s1_oos_d + s4h5_oos_d):+.2f}%p)")
    print("  * 예상 = S1 + S4 단순합 (상호작용 없다고 가정)")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Study 1+4 통합 백테스트 — IS / OOS 분리 실행")
    print("=" * 70)
    print(f"  IS  기간: {DATE_FULL_START} ~ {DATE_IS_END}")
    print(f"  OOS 기간: {DATE_OOS_START} ~ {DATE_FULL_END}")
    print(f"  전체기간: {DATE_FULL_START} ~ {DATE_FULL_END}")
    print(f"  시나리오: {len(SCENARIOS)}개 × 3기간 = {len(SCENARIOS) * 3}개 병렬")
    print()

    combos = (
        [(name, cfg, "FULL", DATE_FULL_START, DATE_FULL_END) for name, cfg in SCENARIOS.items()]
        + [(name, cfg, "IS",   DATE_FULL_START, DATE_IS_END)   for name, cfg in SCENARIOS.items()]
        + [(name, cfg, "OOS",  DATE_OOS_START,  DATE_FULL_END)  for name, cfg in SCENARIOS.items()]
    )

    n_workers = min(N_JOBS, len(combos))
    print(f"▶ [{len(combos)}개 조합  {n_workers} workers 병렬 실행]")

    with Pool(n_workers) as pool:
        all_results = pool.map(_run_task, combos)

    results_full = sorted(
        [r for r in all_results if r["period"] == "FULL"],
        key=lambda r: list(SCENARIOS).index(r["scenario"]),
    )
    results_is = sorted(
        [r for r in all_results if r["period"] == "IS"],
        key=lambda r: list(SCENARIOS).index(r["scenario"]),
    )
    results_oos = sorted(
        [r for r in all_results if r["period"] == "OOS"],
        key=lambda r: list(SCENARIOS).index(r["scenario"]),
    )

    print(f"\n완료: {len(all_results)}/{len(combos)}\n")

    print_comparison(results_full, results_is, results_oos)

    # CSV 저장
    out_dir = _PROJECT_ROOT / "data" / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_v2_results.csv"

    csv_fields = [
        "period", "scenario", "label",
        "total_return", "win_rate", "mdd", "sharpe",
        "buy_trades", "sell_trades", "win_trades", "lose_trades",
        "avg_pnl", "total_pnl",
        "forced_liquidations", "r17_count", "r18_count", "avg_buy_score",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  CSV 저장: {out_path}")

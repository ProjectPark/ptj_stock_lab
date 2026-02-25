#!/usr/bin/env python3
"""
Phase 4A -- hold_days=5, Study B/R13 OFF 파라미터 검증
=======================================================
D2SBacktest v1 + D2S_ENGINE 수정 파라미터로 WARM 기간만 실행.

수정 파라미터 (D2S_ENGINE):
  optimal_hold_days_max: 7 -> 5
  mstu_riskoff_contrarian_only: True -> False
  robn_riskoff_momentum_boost: True -> False
  conl_contrarian_require_riskoff: True -> False
  spy_streak_max: 3 -> 999 (비활성화)

실거래 목표 (attach v1):
  승률: 65.5%
  평균 수익률: +7.39%
  라운드트립: 722건

Usage:
    pyenv shell ptj_stock_lab && python experiments/backtest_d2s_v1_optimized.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.backtests.backtest_d2s import D2SBacktest
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE

# -- 실거래 목표 ---------------------------------------------------------------
ACTUAL_TARGET = {
    "period":       "2025-02-19 ~ 2026-02-12",
    "trading_days": 248,
    "total_trades": 953,
    "round_trips":  722,
    "win_rate":     65.5,
    "avg_pnl_pct":  7.39,
}

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"


def run_warm() -> dict:
    label = "WARM"
    start = date(2025, 3, 3)
    end   = date(2026, 2, 12)
    note  = "Phase 4A 파라미터 검증 (기술적 지표 워밍업 확보)"

    print(f"\n{'=' * 65}")
    print(f"  [{label}] {start} ~ {end}  ({note})")
    print(f"{'=' * 65}")

    bt = D2SBacktest(
        params=D2S_ENGINE,
        start_date=start,
        end_date=end,
        use_fees=True,
    )
    bt.run(verbose=False)
    bt.print_report()

    rpt = bt.report()
    rpt["period_label"] = label
    rpt["note"] = note

    path = RESULTS_DIR / "d2s_v1_optimized_warm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rpt, f, indent=2, default=str)
    print(f"  [저장] {path.name}")

    return rpt


def print_comparison(result: dict) -> None:
    tgt = ACTUAL_TARGET
    print(f"\n{'=' * 75}")
    print("  Phase 4A -- 실거래 목표 대비 비교")
    print(f"{'=' * 75}")
    print(f"  {'구분':<12}  {'기간':30}  {'승률':>6}  {'평균PnL%':>8}  {'거래수':>6}")
    print(f"  {'-'*12}  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*6}")

    # 실거래 목표
    print(f"  {'[실거래]':<12}  {tgt['period']:30}  "
          f"{tgt['win_rate']:>6.1f}  {tgt['avg_pnl_pct']:>8.2f}  "
          f"{tgt['round_trips']:>6}")

    # Phase 4A 결과
    period_str = f"{result.get('period', '-')}"
    win_rate   = result.get("win_rate", 0)
    avg_pnl    = result.get("avg_pnl_pct", 0)
    sells      = result.get("sell_trades", 0)
    label      = result.get("period_label", "?")

    wr_gap  = win_rate - tgt["win_rate"]
    pnl_gap = avg_pnl  - tgt["avg_pnl_pct"]

    print(f"  [{label:<10}]  {period_str:30}  "
          f"{win_rate:>6.1f}  {avg_pnl:>8.2f}  {sells:>6}  "
          f"  (wr {wr_gap:+.1f}%p / avg {pnl_gap:+.2f}%p)")

    print(f"{'=' * 75}")
    print()
    print("  변경 사항:")
    print("    optimal_hold_days_max: 7 -> 5")
    print("    mstu_riskoff_contrarian_only: True -> False")
    print("    robn_riskoff_momentum_boost: True -> False")
    print("    conl_contrarian_require_riskoff: True -> False")
    print("    spy_streak_max: 3 -> 999 (비활성화)")


def main() -> None:
    print("=" * 65)
    print("  Phase 4A -- hold_days=5, Study B/R13 OFF 파라미터 검증")
    print("=" * 65)
    print(f"\n  실거래 목표: 승률 {ACTUAL_TARGET['win_rate']}% / "
          f"평균 수익 {ACTUAL_TARGET['avg_pnl_pct']}% / "
          f"라운드트립 {ACTUAL_TARGET['round_trips']}건\n")

    rpt = run_warm()
    print_comparison(rpt)


if __name__ == "__main__":
    main()

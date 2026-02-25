#!/usr/bin/env python3
"""
Phase 1 — attach v1 실거래 재현 기준선 백테스트
=================================================
D2SBacktest v1 (R1~R16) + D2S_ENGINE 기본 파라미터로
실거래 기간을 실행해 "재현도"를 측정한다.

실거래 목표 (attach v1 §0-1):
  기간: 2025-02-19 ~ 2026-02-12 (248 거래일)
  거래: 953건 (구매 882 / 판매 71)
  라운드트립: 722건
  승률: 65.5%
  평균 수익률: +7.39%

두 기간을 비교:
  - FULL: 2025-02-19 ~ 2026-02-12 (실거래와 동일, ROBN 워밍업 부족 가능)
  - WARM: 2025-03-03 ~ 2026-02-12 (기술적 지표 워밍업 확보)

Usage:
    pyenv shell ptj_stock_lab && python experiments/backtest_d2s_v1_baseline.py
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

# ── 실거래 목표 ────────────────────────────────────────────────
ACTUAL_TARGET = {
    "period":       "2025-02-19 ~ 2026-02-12",
    "trading_days": 248,
    "total_trades": 953,
    "round_trips":  722,
    "win_rate":     65.5,
    "avg_pnl_pct":  7.39,
}

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"

# ── 실행 기간 정의 ─────────────────────────────────────────────
PERIODS = [
    {
        "label": "FULL",
        "note":  "실거래 기간 일치 (ROBN 워밍업 부족 가능)",
        "start": date(2025, 2, 19),
        "end":   date(2026, 2, 12),
    },
    {
        "label": "WARM",
        "note":  "기술적 지표 워밍업 확보 (실거래 +12일)",
        "start": date(2025, 3, 3),
        "end":   date(2026, 2, 12),
    },
]


def run_period(label: str, note: str, start: date, end: date) -> dict:
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

    path = RESULTS_DIR / f"d2s_v1_baseline_{label.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rpt, f, indent=2, default=str)
    print(f"  [저장] {path.name}")

    return rpt


def print_comparison(results: list[dict]) -> None:
    tgt = ACTUAL_TARGET
    print(f"\n{'=' * 75}")
    print("  attach v1 실거래 재현도 비교")
    print(f"{'=' * 75}")
    print(f"  {'구분':<12}  {'기간':30}  {'승률':>6}  {'평균PnL%':>8}  {'거래수':>6}")
    print(f"  {'-'*12}  {'-'*30}  {'-'*6}  {'-'*8}  {'-'*6}")

    # 실거래 목표
    print(f"  {'[실거래]':<12}  {tgt['period']:30}  "
          f"{tgt['win_rate']:>6.1f}  {tgt['avg_pnl_pct']:>8.2f}  "
          f"{tgt['round_trips']:>6}")

    for r in results:
        period_str = f"{r.get('period', '-')}"
        win_rate   = r.get("win_rate", 0)
        avg_pnl    = r.get("avg_pnl_pct", 0)
        sells      = r.get("sell_trades", 0)
        label      = r.get("period_label", "?")

        # 재현 격차
        wr_gap  = win_rate - tgt["win_rate"]
        pnl_gap = avg_pnl  - tgt["avg_pnl_pct"]

        print(f"  [{label:<10}]  {period_str:30}  "
              f"{win_rate:>6.1f}  {avg_pnl:>8.2f}  {sells:>6}  "
              f"  (wr {wr_gap:+.1f}%p / avg {pnl_gap:+.2f}%p)")

    print(f"{'=' * 75}")
    print()
    print("  해석 기준:")
    print("    승률 격차 ±5%p 이내   → 재현 양호")
    print("    승률 격차 -10%p 이상  → 주요 규칙 미작동 의심")
    print("    평균PnL 격차 -3%p 이상 → 이익실현 로직 점검 필요")


def main() -> None:
    print("=" * 65)
    print("  Phase 1 — attach v1 실거래 재현 기준선 백테스트")
    print("=" * 65)
    print(f"\n  실거래 목표: 승률 {ACTUAL_TARGET['win_rate']}% / "
          f"평균 수익 {ACTUAL_TARGET['avg_pnl_pct']}% / "
          f"라운드트립 {ACTUAL_TARGET['round_trips']}건\n")

    results = []
    for period in PERIODS:
        rpt = run_period(**period)
        results.append(rpt)

    print_comparison(results)


if __name__ == "__main__":
    main()

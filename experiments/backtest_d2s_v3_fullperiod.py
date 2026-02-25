#!/usr/bin/env python3
"""
D2S v3 Walk-Forward best params → 전체 기간 백테스트
=====================================================
d2s_v3_wf_summary.json (W1/W2/W3 best params) 를 로드해서
각각 전체 기간(2025-03-03 ~ 현재)으로 백테스트 실행.

목적:
  - W1 best params가 전체 기간에서도 우수한지 확인
  - WF OOS vs Full-period 성과 비교
  - 실거래 기간(2025-02-19~2026-02-25) 재현 가능성 측정

Usage:
    pyenv shell ptj_stock_lab && python experiments/backtest_d2s_v3_fullperiod.py
    pyenv shell ptj_stock_lab && python experiments/backtest_d2s_v3_fullperiod.py --window W1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3

# ── 기간 설정 ──────────────────────────────────────────────
FULL_START = date(2025, 3,  3)   # 기술적 지표 워밍업 후 (MSTU 상장 기준)
FULL_END   = date(2026, 2, 25)   # 오늘 기준 최신

WF_SUMMARY_PATH = (
    _PROJECT_ROOT / "data" / "results" / "optimization" / "d2s_v3_wf_summary.json"
)
RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"


def load_wf_summary() -> list[dict]:
    with open(WF_SUMMARY_PATH) as f:
        return json.load(f)


def run_window(window_id: str, best_params: dict, verbose: bool = True) -> dict:
    """W1/W2/W3 best params로 전체 기간 백테스트 실행."""
    # D2S_ENGINE_V3 기본값에 best_params 덮어쓰기
    params = {**D2S_ENGINE_V3, **best_params}

    print(f"\n{'=' * 65}")
    print(f"  [{window_id}] best params → 전체 기간 ({FULL_START} ~ {FULL_END})")
    print(f"{'=' * 65}")

    bt = D2SBacktestV3(
        params=params,
        start_date=FULL_START,
        end_date=FULL_END,
    )
    bt.run(verbose=verbose)
    bt.print_report()

    rpt = bt.report()
    rpt["window"] = window_id
    rpt["period_type"] = "full"

    # 결과 저장
    out_path = RESULTS_DIR / f"d2s_v3_{window_id}_full_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rpt, f, indent=2, default=str)
    print(f"\n  [저장] {out_path}")

    return rpt


def compare_summary(results: list[dict]) -> None:
    """W1/W2/W3 전체기간 성과 비교표 출력."""
    print(f"\n{'=' * 75}")
    print("  WF best params → 전체 기간 성과 비교")
    print(f"{'=' * 75}")
    header = f"  {'Window':<6}  {'Win%':>6}  {'Return%':>8}  {'MDD%':>7}  {'Sharpe':>7}  {'Trades':>7}"
    print(header)
    print("  " + "-" * 60)
    for r in results:
        wid = r.get("window", "?")
        wr  = r.get("win_rate", 0)
        ret = r.get("total_return_pct", 0)
        mdd = r.get("mdd_pct", 0)
        sh  = r.get("sharpe_ratio", 0)
        tr  = r.get("sell_trades", 0)
        print(f"  {wid:<6}  {wr:>6.1f}  {ret:>8.2f}  {mdd:>7.2f}  {sh:>7.3f}  {tr:>7}")
    print(f"{'=' * 75}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D2S v3 WF best params 전체 기간 백테스트"
    )
    parser.add_argument(
        "--window",
        choices=["W1", "W2", "W3", "all"],
        default="all",
        help="실행할 창 (기본: all)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    wf_data = load_wf_summary()
    window_map = {w["window"]: w for w in wf_data}

    target_windows = ["W1", "W2", "W3"] if args.window == "all" else [args.window]

    results = []
    for wid in target_windows:
        if wid not in window_map:
            print(f"  [WARN] {wid} 데이터 없음 — 건너뜀")
            continue
        entry = window_map[wid]
        rpt = run_window(wid, entry["best_params"], verbose=args.verbose)
        results.append(rpt)

    if len(results) > 1:
        compare_summary(results)


if __name__ == "__main__":
    main()

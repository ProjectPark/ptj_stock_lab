#!/usr/bin/env python3
"""
Study 7 — IS(bull market) 구간 D2S v3 성능 확인
================================================
목적: IS 구간(2024-09-18~2025-05-31)에서 v2 vs v3 성능 비교
     R19/R20/R21 조합이 bull market에서도 유효한지 검증

비교 3가지:
  v2_IS:  D2SBacktestV2, IS 기간 (2024-09-18~2025-05-31)
  v3_IS:  D2SBacktestV3 (NO_ROBN), IS 기간
  v3_OOS: D2SBacktestV3 (NO_ROBN), OOS 기간 (2025-06-01~2026-02-17)
          ← Optuna #449 검증값 재확인용
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2
from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2, D2S_ENGINE_V3_NO_ROBN

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START  = date(2024, 9, 18)
IS_END    = date(2025, 5, 31)
OOS_START = date(2025, 6, 1)
OOS_END   = date(2026, 2, 17)

def run_bt(label, cls, params, start, end):
    print(f"\n[{label}] {start} ~ {end}")
    bt = cls(params=params, start_date=start, end_date=end)
    bt.run(verbose=False)
    bt.print_report()
    return bt.report()

def main():
    print("=" * 60)
    print("  Study 7: IS(bull market) 구간 v3 성능 확인")
    print("=" * 60)

    r_v2_is  = run_bt("v2  IS ", D2SBacktestV2, D2S_ENGINE_V2,        IS_START,  IS_END)
    r_v3_is  = run_bt("v3  IS ", D2SBacktestV3, D2S_ENGINE_V3_NO_ROBN, IS_START,  IS_END)
    r_v3_oos = run_bt("v3  OOS", D2SBacktestV3, D2S_ENGINE_V3_NO_ROBN, OOS_START, OOS_END)

    print("\n" + "=" * 60)
    print("  IS vs OOS 비교 (v2/v3)")
    print("=" * 60)
    print(f"  {'지표':12s}  {'v2_IS':>10}  {'v3_IS':>10}  {'v3_OOS':>10}")
    print(f"  {'-'*50}")
    for key, label in [
        ("total_return_pct", "총 수익률"),
        ("win_rate",         "승률"),
        ("mdd_pct",          "MDD"),
        ("sharpe_ratio",     "Sharpe"),
        ("sell_trades",      "총 청산"),
    ]:
        v2i = r_v2_is.get(key, 0)
        v3i = r_v3_is.get(key, 0)
        v3o = r_v3_oos.get(key, 0)
        print(f"  {label:12s}  {str(v2i):>10}  {str(v3i):>10}  {str(v3o):>10}")
    print("=" * 60)

    result = {
        "study": "study_7_is_period_v3",
        "run_date": datetime.now().isoformat(),
        "v2_IS":  r_v2_is,
        "v3_IS":  r_v3_is,
        "v3_OOS": r_v3_oos,
    }
    out_path = RESULTS_DIR / f"study_7_is_period_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")

if __name__ == "__main__":
    main()

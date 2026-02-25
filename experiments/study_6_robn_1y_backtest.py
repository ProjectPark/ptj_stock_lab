#!/usr/bin/env python3
"""
Study 6 — ROBN 포함 1년 D2S v3 백테스트
========================================
목적: ROBN 상장(2025-01-31) 이후 4종목(ROBN/CONL/MSTU/AMDL) v3 성능 검증
기간: 2025-01-31 ~ 2026-02-17

비교:
  v2 (D2S_ENGINE_V2 + ROBN 포함 파라미터): 기준선
  v3 (D2S_ENGINE_V3, 4종목): R19+R20+R21 레짐 조건부

Note: D2S_ENGINE_V3는 기본 4종목(ROBN 포함) 사용
      D2S_ENGINE_V3_NO_ROBN는 3종목 (이 스터디에서는 미사용)
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
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2, D2S_ENGINE_V3

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2025, 1, 31)
END_DATE   = date(2026, 2, 17)

def main():
    print("=" * 60)
    print("  Study 6: ROBN 포함 1년 D2S v3 백테스트")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    # v2 기준선 (ROBN 포함, 기간 조정)
    print("\n[v2] 기준선 (4종목, ROBN 포함)")
    bt_v2 = D2SBacktestV2(
        params=D2S_ENGINE_V2,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    bt_v2.run(verbose=False)
    bt_v2.print_report()
    r2 = bt_v2.report()

    # v3 (ROBN 포함, 기간 조정)
    print("\n[v3] R19+R20+R21 레짐 조건부 (4종목, ROBN 포함)")
    bt_v3 = D2SBacktestV3(
        params=D2S_ENGINE_V3,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    bt_v3.run(verbose=True)
    bt_v3.print_report()
    r3 = bt_v3.report()

    # 비교 출력
    print("\n" + "=" * 60)
    print("  v2 vs v3 (ROBN 포함 1년 비교)")
    print("=" * 60)
    for key, label in [
        ("total_return_pct", "총 수익률"),
        ("win_rate",         "승률"),
        ("mdd_pct",          "MDD"),
        ("sharpe_ratio",     "Sharpe"),
        ("sell_trades",      "총 청산"),
        ("avg_pnl_pct",      "평균 수익"),
    ]:
        v2_val = r2.get(key, 0)
        v3_val = r3.get(key, 0)
        diff = v3_val - v2_val if isinstance(v2_val, (int, float)) else "-"
        sign = "+" if isinstance(diff, float) and diff > 0 else ""
        print(f"  {label:12s}: v2={v2_val:>8}  v3={v3_val:>8}  Δ={sign}{diff:.2f}" if isinstance(diff, float) else f"  {label:12s}: v2={v2_val}  v3={v3_val}")
    print("=" * 60)

    # 결과 저장
    result = {
        "study": "study_6_robn_1y",
        "run_date": datetime.now().isoformat(),
        "period": {"start": str(START_DATE), "end": str(END_DATE)},
        "v2": r2,
        "v3": r3,
    }
    out_path = RESULTS_DIR / f"study_6_robn_1y_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")

if __name__ == "__main__":
    main()

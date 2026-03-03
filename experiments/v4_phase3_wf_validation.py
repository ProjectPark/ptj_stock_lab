#!/usr/bin/env python3
"""
PTJ v4 — Phase 3: Robustness Check (Walk-Forward 검증)
=======================================================
목적: Phase 1 확정 파라미터(Study 1+2+5)가 과적합인지 확인.
     고정 파라미터를 여러 OOS 창에 적용해 일관성 검증.

설계:
  - 파라미터: Study 1+2+5 확정값 고정 (Phase 2 Baseline 기준)
  - IS 창: Expanding (2023-01-03 고정)
  - OOS 창: 6개월 Rolling

창 구성 (backtest_1min_3y.parquet):
  W1: IS 2023-01-03~2024-12-31 | OOS 2025-01-01~2025-06-30
  W2: IS 2023-01-03~2025-06-30 | OOS 2025-07-01~2025-12-31
  W3: IS 2023-01-03~2025-12-31 | OOS 2026-01-01~2026-02-17 (Phase 2와 동일)
  FULL: 2023-01-03~2026-02-17

Usage:
    pyenv shell ptj_stock_lab && python experiments/v4_phase3_wf_validation.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.optimizers.optimize_v4_phase2 import V4Phase2Optimizer

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "optimization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── OOS 창 정의 (Expanding IS + Rolling OOS 6개월) ─────────────
WINDOWS = [
    ("W1", date(2023, 1,  3), date(2024, 12, 31), date(2025, 1,  1), date(2025, 6, 30)),
    ("W2", date(2023, 1,  3), date(2025, 6,  30), date(2025, 7,  1), date(2025, 12, 31)),
    ("W3", date(2023, 1,  3), date(2025, 12, 31), date(2026, 1,  1), date(2026, 2, 17)),
]
FULL_WINDOW = ("FULL", date(2023, 1, 3), date(2026, 2, 17))


def main():
    print("=" * 70)
    print("  PTJ v4 — Phase 3: Robustness Check (Walk-Forward 검증)")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # ── 확정 파라미터 로드 ──────────────────────────────────────
    print("\n  [1] Study 1+2+5 확정 파라미터 로드")
    opt = V4Phase2Optimizer()
    params = opt.get_baseline_params()

    key_params = {
        "V4_SWING_TRIGGER_PCT":         params.get("V4_SWING_TRIGGER_PCT"),
        "V4_SWING_STAGE1_DRAWDOWN_PCT": params.get("V4_SWING_STAGE1_DRAWDOWN_PCT"),
        "V4_SWING_STAGE1_ATR_MULT":     params.get("V4_SWING_STAGE1_ATR_MULT"),
        "V4_SWING_STAGE1_HOLD_DAYS":    params.get("V4_SWING_STAGE1_HOLD_DAYS"),
        "V4_CB_BTC_SURGE_PCT":          params.get("V4_CB_BTC_SURGE_PCT"),
        "V4_CB_BTC_CRASH_PCT":          params.get("V4_CB_BTC_CRASH_PCT"),
        "V4_PAIR_FIXED_TP_PCT":         params.get("V4_PAIR_FIXED_TP_PCT"),
        "PAIR_GAP_SELL_THRESHOLD_V2":   params.get("PAIR_GAP_SELL_THRESHOLD_V2"),
        "STOP_LOSS_PCT":                params.get("STOP_LOSS_PCT"),
        "CONL_SELL_PROFIT_PCT":         params.get("CONL_SELL_PROFIT_PCT"),
        "DCA_DROP_PCT":                 params.get("DCA_DROP_PCT"),
        "V4_SIDEWAYS_ATR_DECLINE_PCT":  params.get("V4_SIDEWAYS_ATR_DECLINE_PCT"),
    }
    print("  주요 확정 파라미터:")
    for k, v in key_params.items():
        print(f"    {k:<44s} = {v}")

    # ── 창별 백테스트 ─────────────────────────────────────────
    print(f"\n  [2] Walk-Forward 창별 검증 (고정 파라미터)")
    print(f"  {'창':>4}  {'IS 기간':>25}  {'OOS 기간':>25}  "
          f"{'IS%':>7}  {'OOS%':>7}  {'OOS_Shp':>8}  {'OOS_MDD':>8}  {'거래':>5}")
    print("  " + "-" * 100)

    all_results = []
    t_total_start = time.time()

    for win_id, is_start, is_end, oos_start, oos_end in WINDOWS:
        t0 = time.time()

        r_is = opt.run_single_trial(
            params,
            start_date=is_start.isoformat(),
            end_date=is_end.isoformat(),
        )
        r_oos = opt.run_single_trial(
            params,
            start_date=oos_start.isoformat(),
            end_date=oos_end.isoformat(),
        )

        elapsed = time.time() - t0
        row = {
            "window": win_id,
            "is_start": str(is_start), "is_end": str(is_end),
            "oos_start": str(oos_start), "oos_end": str(oos_end),
            "is_return": r_is.total_return_pct,
            "is_sharpe": r_is.sharpe,
            "is_mdd":    r_is.mdd,
            "is_wr":     r_is.win_rate,
            "is_trades": r_is.total_sells,
            "oos_return": r_oos.total_return_pct,
            "oos_sharpe": r_oos.sharpe,
            "oos_mdd":    r_oos.mdd,
            "oos_wr":     r_oos.win_rate,
            "oos_trades": r_oos.total_sells,
        }
        all_results.append(row)

        oos_positive_mark = "✅" if r_oos.total_return_pct > 0 else "❌"
        print(f"  {win_id:>4}  {str(is_start)+'~'+str(is_end):>25}  "
              f"{str(oos_start)+'~'+str(oos_end):>25}  "
              f"{r_is.total_return_pct:>6.1f}%  "
              f"{oos_positive_mark}{r_oos.total_return_pct:>+6.2f}%  "
              f"{r_oos.sharpe:>8.3f}  "
              f"{r_oos.mdd:>7.1f}%  "
              f"{r_oos.total_sells:>4}회  ({elapsed:.0f}s)")

    # ── FULL 기간 ─────────────────────────────────────────────
    print(f"\n  [3] FULL 기간 검증 ({FULL_WINDOW[1]} ~ {FULL_WINDOW[2]})")
    t0 = time.time()
    win_id, full_start, full_end = FULL_WINDOW
    r_full = opt.run_single_trial(
        params,
        start_date=full_start.isoformat(),
        end_date=full_end.isoformat(),
    )
    elapsed = time.time() - t0
    full_row = {
        "window": win_id,
        "start": str(full_start), "end": str(full_end),
        "return": r_full.total_return_pct,
        "sharpe": r_full.sharpe,
        "mdd":    r_full.mdd,
        "win_rate": r_full.win_rate,
        "trades": r_full.total_sells,
    }
    print(f"  FULL {full_start}~{full_end}:  "
          f"{r_full.total_return_pct:+.2f}%  "
          f"Sharpe={r_full.sharpe:.3f}  "
          f"MDD={r_full.mdd:.1f}%  "
          f"WR={r_full.win_rate:.1f}%  "
          f"거래={r_full.total_sells}회  ({elapsed:.0f}s)")

    # ── 요약 통계 ─────────────────────────────────────────────
    oos_returns = [r["oos_return"] for r in all_results]
    oos_positive = sum(1 for x in oos_returns if x > 0)
    avg_oos = sum(oos_returns) / len(oos_returns) if oos_returns else 0
    avg_oos_sharpe = sum(r["oos_sharpe"] for r in all_results) / len(all_results) if all_results else 0
    avg_oos_mdd = sum(r["oos_mdd"] for r in all_results) / len(all_results) if all_results else 0

    total_elapsed = time.time() - t_total_start
    print("\n" + "=" * 70)
    print("  [Phase 3] WF 검증 요약")
    print("=" * 70)
    print(f"  OOS 창 양수 수익: {oos_positive}/{len(all_results)}")
    print(f"  OOS 평균 수익률: {avg_oos:+.2f}%")
    print(f"  OOS 평균 Sharpe: {avg_oos_sharpe:.3f}")
    print(f"  OOS 평균 MDD   : {avg_oos_mdd:.1f}%")
    print(f"  FULL 수익률    : {r_full.total_return_pct:+.2f}%  Sharpe: {r_full.sharpe:.3f}")
    print(f"  총 실행 시간   : {total_elapsed:.0f}초")

    if oos_positive == len(all_results):
        verdict = "전 창 OOS 양수 — 파라미터 견고성 확인"
        verdict_icon = "✅"
    elif oos_positive >= (len(all_results) + 1) // 2:
        verdict = "과반 창 OOS 양수 — 부분 견고성"
        verdict_icon = "⚠️"
    else:
        verdict = "과반 창 OOS 음수 — 과적합 가능성"
        verdict_icon = "❌"
    print(f"\n  판정: {verdict_icon} {verdict}")
    print("=" * 70)

    # ── 저장 ──────────────────────────────────────────────────
    out = {
        "phase": "v4_phase3",
        "description": "Phase 1 확정값(Study 1+2+5) Walk-Forward 견고성 검증",
        "confirmed_params": key_params,
        "windows": all_results,
        "full": full_row,
        "summary": {
            "oos_positive": oos_positive,
            "oos_total": len(all_results),
            "avg_oos_return": avg_oos,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_oos_mdd": avg_oos_mdd,
            "full_return": r_full.total_return_pct,
            "full_sharpe": r_full.sharpe,
            "verdict": verdict,
        },
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / f"v4_phase3_wf_validation_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()

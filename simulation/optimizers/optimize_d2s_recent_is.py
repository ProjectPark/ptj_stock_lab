#!/usr/bin/env python3
"""
D2S 최근 레짐 IS 최적화 (B-1) — r2
====================================
IS=2025-10-01~2026-01-31, OOS=2026-02-01~2026-02-25

목적: W3 OOS(-32% MDD, 2026 급락장) 원인인 현 레짐에 적응하는 파라미터 탐색.
      IS를 최근 4개월로 짧게 설정해 2026 레짐 특성 학습.

기존 D2S v3 Optuna와 완전 동일한 탐색 공간 + 스코어 함수.

Usage:
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_d2s_recent_is.py \\
        --n-trials 500 --n-jobs 20 \\
        --study-name d2s_v3_recent_is_r2 \\
        --journal data/optuna/d2s_v3_recent_is_r2_journal.log
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 기존 v3 optimizer 함수들 재사용
from simulation.optimizers.optimize_d2s_v3_optuna import (
    _optimize_window,
    _print_wf_summary,
)

# ── 창 정의 ───────────────────────────────────────────────────
RECENT_IS_WINDOW = (
    "recent_is_r2",
    date(2025, 10, 1),   # IS start
    date(2026, 1, 31),   # IS end
    date(2026, 2,  1),   # OOS start
    date(2026, 2, 25),   # OOS end
)

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "optimization"


def main():
    parser = argparse.ArgumentParser(description="D2S 최근 레짐 IS 최적화 (B-1) r2")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-jobs",   type=int, default=20)
    parser.add_argument("--study-name", type=str, default="d2s_v3_recent_is_r2",
                        help="Optuna study name (win_id 파생에 사용)")
    parser.add_argument("--journal", type=str, default=None,
                        help="Journal log path (미지정 시 자동 생성)")
    args = parser.parse_args()

    win_id, is_start, is_end, oos_start, oos_end = RECENT_IS_WINDOW

    print("=" * 70)
    print("  D2S 최근 레짐 IS 최적화 (B-1)")
    print(f"  IS: {is_start} ~ {is_end}  ({(is_end - is_start).days}일)")
    print(f"  OOS: {oos_start} ~ {oos_end}  ({(oos_end - oos_start).days}일)")
    print(f"  Trials: {args.n_trials}  Jobs: {args.n_jobs}")
    print("=" * 70)

    result = _optimize_window(
        win_id=win_id,
        is_start=is_start,
        is_end=is_end,
        oos_start=oos_start,
        oos_end=oos_end,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    # 요약 출력
    _print_wf_summary([result])

    # wf_summary에 추가 저장
    summary_path = RESULTS_DIR / "d2s_v3_wf_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = []

    # RECENT_IS 항목 업데이트 (이미 있으면 교체)
    summary = [s for s in summary if s.get("window") != win_id]
    summary.append(result)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  wf_summary 업데이트: {summary_path.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Study 3 — 2-layer DCA 강화 (2nd layer 진입 조건 gating)
=========================================================
핵심 질문: 2nd DCA layer를 무조건 허용하는 것이 최적인가,
           아니면 진입가 대비 X% 하락 시에만 허용해야 하는가?

테스트 파라미터:
  dca_layer2_drop_threshold: [None(현재=무조건허용), -1%, -2%, -3%, -5%]
  dca_max_layers: [1, 2]

Grid: 5 drop × 2 layer × 3 period = 30 runs

구현: StudyDCABacktest(D2SBacktestV2) 서브클래스
  - _execute_buy() 오버라이드: 2nd layer 진입 가 대비 하락 체크

보고 항목: total_return, win_rate, MDD, Sharpe, buy_trades,
           avg_dca_count (DCA 평균 레이어), forced_exit_count

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_3_dca.py
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

from simulation.strategies.line_c_d2s.d2s_engine import DailySnapshot
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2
from simulation.backtests.backtest_d2s import TradeRecord
from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2


# ============================================================
# StudyDCABacktest — 2nd layer drop threshold gating
# ============================================================

class StudyDCABacktest(D2SBacktestV2):
    """Study 3 DCA 그리드 백테스트.

    추가 파라미터 (params dict에 설정)
    ------------------------------------
    dca_layer2_drop_threshold  float | None
        2nd DCA layer 허용 최소 하락률.
        예: -2.0 → 진입가 대비 -2.0% 이하일 때만 DCA 허용.
        None이면 무조건 허용 (현재 동작 = baseline).
    """

    def __init__(self, params: dict, **kwargs):
        super().__init__(params, **kwargs)
        self._dca_buy_count = 0  # DCA 레이어 추가 매수 횟수

    def _execute_buy(
        self,
        ticker: str,
        size: float,
        snap: DailySnapshot,
        reason: str,
        score: float,
    ) -> TradeRecord | None:
        """2nd layer drop threshold 체크 후 부모 매수 실행."""
        pos = self.positions.get(ticker)
        is_dca = pos is not None

        # ── 2nd layer 하락 gating ──
        if is_dca:
            threshold = self.params.get("dca_layer2_drop_threshold", None)
            if threshold is not None:
                current_price = snap.closes.get(ticker, 0.0)
                if current_price > 0 and pos.entry_price > 0:
                    current_pnl = (current_price - pos.entry_price) / pos.entry_price * 100
                    if current_pnl > threshold:  # 충분히 하락하지 않았으면 DCA 거부
                        return None

        trade = super()._execute_buy(ticker, size, snap, reason, score)
        if trade is not None and is_dca:
            self._dca_buy_count += 1
        return trade


# ============================================================
# 그리드 정의
# ============================================================

# dca_layer2_drop_threshold: None(무조건허용), -1%, -2%, -3%, -5%
DROP_THRESHOLD_GRID: list[float | None] = [None, -1.0, -2.0, -3.0, -5.0]
MAX_LAYERS_GRID: list[int] = [1, 2]

DATE_FULL_START = date(2025, 3, 3)
DATE_FULL_END   = date(2026, 2, 17)
DATE_IS_END     = date(2025, 9, 30)
DATE_OOS_START  = date(2025, 10, 1)

N_JOBS = int(os.environ.get("N_JOBS", "8"))


# ============================================================
# 단일 실행
# ============================================================

def run_bt(
    drop_threshold: float | None,
    max_layers: int,
    start: date,
    end: date,
) -> dict:
    params = deepcopy(D2S_ENGINE_V2)
    params["dca_max_layers"] = max_layers
    if drop_threshold is not None:
        params["dca_layer2_drop_threshold"] = drop_threshold

    bt = StudyDCABacktest(params=params, start_date=start, end_date=end, use_fees=True)
    bt.run(verbose=False)
    r = bt.report()

    forced_exit = sum(1 for t in bt.trades if t.side == "SELL" and "hold_days" in t.reason)
    sell_trades = r["sell_trades"]
    avg_dca = bt._dca_buy_count / sell_trades if sell_trades > 0 else 0.0

    threshold_label = f"{drop_threshold:.0f}%" if drop_threshold is not None else "None(무조건)"

    return {
        "drop_threshold": drop_threshold,
        "drop_label": threshold_label,
        "max_layers": max_layers,
        "total_return": r["total_return_pct"],
        "win_rate": r["win_rate"],
        "mdd": r["mdd_pct"],
        "sharpe": r["sharpe_ratio"],
        "buy_trades": r["buy_trades"],
        "sell_trades": sell_trades,
        "win_trades": r["win_trades"],
        "lose_trades": r["lose_trades"],
        "avg_pnl": r["avg_pnl_pct"],
        "total_pnl": r["total_pnl"],
        "forced_exit_count": forced_exit,
        "avg_dca_count": round(avg_dca, 3),
        "dca_buy_count": bt._dca_buy_count,
        "r17_count": bt._r17_count,
        "r18_count": bt._r18_count,
    }


def _run_bt_task(args: tuple) -> dict:
    drop, layers, period_name, pstart, pend = args
    r = run_bt(drop, layers, pstart, pend)
    r["period"] = period_name
    return r


# ============================================================
# 비교 출력
# ============================================================

def print_results(all_results: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("  Study 3 DCA 그리드 결과 — 2nd layer 진입 조건 gating")
    print("=" * 88)

    for period_name in ["FULL", "IS", "OOS"]:
        sub = [r for r in all_results if r["period"] == period_name]
        # baseline: drop=None, layers=2 (현재 동작)
        base = next(
            (r for r in sub if r["drop_threshold"] is None and r["max_layers"] == 2),
            None,
        )
        if base is None:
            continue

        print(
            f"\n[{period_name}] 기준 (drop=None, lyr=2): "
            f"수익률 {base['total_return']:+.2f}%  Sharpe {base['sharpe']:.3f}  "
            f"매수 {base['buy_trades']}건  avg_dca {base['avg_dca_count']:.2f}회"
        )
        print(
            f"  {'lyr':>4} {'Threshold':>12} {'수익률':>9} {'MDD':>7} {'Sharpe':>7}"
            f" {'매수':>5} {'avg_dca':>8} {'강제청산':>7} {'Δ':>8}"
        )
        print("  " + "-" * 78)

        for layers in [2, 1]:
            for drop in DROP_THRESHOLD_GRID:
                row = next(
                    (r for r in sub if r["max_layers"] == layers and r["drop_threshold"] == drop),
                    None,
                )
                if row is None:
                    continue
                delta = row["total_return"] - base["total_return"]
                is_base = "★" if (drop is None and layers == 2) else " "
                print(
                    f"  {is_base}lyr={layers} {row['drop_label']:>12}"
                    f" {row['total_return']:>+8.2f}%"
                    f" {row['mdd']:>6.1f}%"
                    f" {row['sharpe']:>7.3f}"
                    f" {row['buy_trades']:>5}건"
                    f" {row['avg_dca_count']:>7.2f}회"
                    f" {row['forced_exit_count']:>5}건"
                    f" {delta:>+7.2f}%p"
                )
            print("  " + "·" * 54)

    # IS vs OOS 안정성 (layers=2 고정)
    print("\n[IS vs OOS 안정성 — max_layers=2 고정]")
    print(
        f"  {'Threshold':>12} {'IS 수익률':>10} {'OOS 수익률':>11}"
        f" {'IS Sharpe':>10} {'OOS Sharpe':>11} {'OOS Δ':>8}"
    )
    print("  " + "-" * 68)

    is_results  = [r for r in all_results if r["period"] == "IS"  and r["max_layers"] == 2]
    oos_results = [r for r in all_results if r["period"] == "OOS" and r["max_layers"] == 2]
    base_oos = next(r for r in oos_results if r["drop_threshold"] is None)

    for drop in DROP_THRESHOLD_GRID:
        r_is  = next((r for r in is_results  if r["drop_threshold"] == drop), None)
        r_oos = next((r for r in oos_results if r["drop_threshold"] == drop), None)
        if r_is is None or r_oos is None:
            continue
        oos_delta = r_oos["total_return"] - base_oos["total_return"]
        oos_delta_str = f"{oos_delta:+.2f}%p" if drop is not None else "  —"
        print(
            f"  {r_is['drop_label']:>12}"
            f" {r_is['total_return']:>+9.2f}%"
            f" {r_oos['total_return']:>+10.2f}%"
            f" {r_is['sharpe']:>10.3f}"
            f" {r_oos['sharpe']:>10.3f}"
            f" {oos_delta_str:>8}"
        )

    # 결론
    full_results = [r for r in all_results if r["period"] == "FULL"]
    best_oos     = max(oos_results, key=lambda r: r["total_return"])
    best_sharpe  = max(full_results, key=lambda r: r["sharpe"])
    print(f"\n[결론 요약]")
    print(
        f"  OOS 최고 수익률: lyr={best_oos['max_layers']}  drop={best_oos['drop_label']}"
        f" → {best_oos['total_return']:+.2f}%"
    )
    print(
        f"  FULL 최고 Sharpe: lyr={best_sharpe['max_layers']}  drop={best_sharpe['drop_label']}"
        f" → Sharpe {best_sharpe['sharpe']:.3f}"
    )


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    periods = [
        ("FULL", DATE_FULL_START, DATE_FULL_END),
        ("IS",   DATE_FULL_START, DATE_IS_END),
        ("OOS",  DATE_OOS_START,  DATE_FULL_END),
    ]

    combos = [
        (drop, layers, period_name, pstart, pend)
        for drop in DROP_THRESHOLD_GRID
        for layers in MAX_LAYERS_GRID
        for period_name, pstart, pend in periods
    ]
    total = len(combos)
    n_workers = min(N_JOBS, total)

    print("=" * 70)
    print("  Study 3 DCA 그리드 서치 — 2nd layer 진입 조건 gating")
    print("=" * 70)
    print(f"  drop_threshold 그리드: {DROP_THRESHOLD_GRID}")
    print(f"  max_layers 그리드:     {MAX_LAYERS_GRID}")
    print(f"  기간: FULL / IS / OOS")
    print(f"  총 조합: {total}개  Workers: {n_workers}")
    print()

    with Pool(n_workers) as pool:
        all_results = pool.map(_run_bt_task, combos)

    print(f"  완료: {len(all_results)}/{total}\n")

    print_results(all_results)

    # CSV 저장
    out_dir = _PROJECT_ROOT / "data" / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "study3_dca_grid.csv"

    csv_fields = [
        "period", "drop_threshold", "drop_label", "max_layers",
        "total_return", "win_rate", "mdd", "sharpe",
        "buy_trades", "sell_trades", "win_trades", "lose_trades",
        "avg_pnl", "total_pnl",
        "forced_exit_count", "avg_dca_count", "dca_buy_count",
        "r17_count", "r18_count",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  CSV 저장: {out_path}")

#!/usr/bin/env python3
"""
Axis 4 스터디 — 청산 파라미터 최적화 (take_profit / hold_days)
==============================================================
Phase 1 EDA  : 강제청산 포지션의 일별 가격 경로 복원 → 조기 청산 시 효과 시뮬레이션
Phase 2 Grid : take_profit_pct × hold_days_max 2D 파라미터 서치 (IS/OOS 분리)

핵심 질문:
  Q1. 강제청산(현재 -10.7% 평균)을 더 일찍 끊으면 손실이 줄어드는가?
  Q2. take_profit을 5.9%에서 올리거나 낮추면 전체 성과가 어떻게 변하는가?
  Q3. 최적 (take_profit, hold_days) 조합이 IS와 OOS에서 안정적인가?

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_exit_params.py
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2
from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2


# ============================================================
# Phase 1: EDA — 강제청산 포지션 경로 복원
# ============================================================

def phase1_eda() -> None:
    print("=" * 65)
    print("  Phase 1 EDA — 강제청산 포지션 일별 경로 분석")
    print("=" * 65)

    trades_path = _PROJECT_ROOT / "data" / "results" / "backtests" / "d2s_trades.csv"
    if not trades_path.exists():
        print("  [!] d2s_trades.csv 없음 — 백테스트 먼저 실행")
        return

    df = pd.read_csv(trades_path, parse_dates=["date"])
    market = pd.read_parquet(
        _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily.parquet"
    )
    market.index = pd.to_datetime(market.index)

    # 종가 테이블 (MultiIndex 해제)
    closes = market.xs("Close", axis=1, level=1)
    trading_dates = sorted(closes.index)

    # 매수/매도 분리
    buys  = df[df["side"] == "BUY"].copy()
    sells = df[df["side"] == "SELL"].copy()

    # 강제청산만
    forced = sells[sells["reason"].str.contains("hold_days", na=False)].copy()
    forced["hold_days_val"] = forced["reason"].str.extract(r"hold_days=(\d+)").astype(float)

    print(f"\n  강제청산 건수: {len(forced)}건 / 전체 청산 {len(sells)}건")
    print(f"  강제청산 평균 PnL: {forced['pnl_pct'].mean():+.2f}%  (std {forced['pnl_pct'].std():.1f}%)")

    # 각 매수 거래에 대해 sell 매칭 (날짜+ticker 순서 기반)
    rows = []
    for _, sell_row in forced.iterrows():
        ticker = sell_row["ticker"]
        sell_date = sell_row["date"]

        # 해당 ticker의 매수 중 sell_date 직전 마지막 매수
        ticker_buys = buys[buys["ticker"] == ticker]
        prior_buys  = ticker_buys[ticker_buys["date"] < sell_date]
        if prior_buys.empty:
            continue
        buy_row = prior_buys.iloc[-1]
        entry_date  = buy_row["date"]
        entry_price = buy_row["price"]

        # 보유 기간 동안 일별 가격 추출
        mask = (closes.index >= entry_date) & (closes.index <= sell_date)
        period_closes = closes.loc[mask, ticker].dropna() if ticker in closes.columns else pd.Series(dtype=float)

        if len(period_closes) < 2:
            continue

        # 각 날짜에 청산했을 때의 가상 PnL
        pnl_by_day: dict[int, float] = {}
        for day_idx, (dt, price) in enumerate(period_closes.items()):
            if day_idx == 0:
                continue  # 진입일 제외
            cal_days = (dt - entry_date).days
            pnl_pct = (price - entry_price) / entry_price * 100
            pnl_by_day[cal_days] = pnl_pct

        rows.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": sell_date,
            "entry_price": entry_price,
            "actual_pnl_pct": sell_row["pnl_pct"],
            "pnl_by_day": pnl_by_day,
        })

    if not rows:
        print("  [!] 매칭 실패 — 데이터 구조 불일치")
        return

    # 가상 청산 PnL 집계 (calendar day 기준)
    print(f"\n  분석 가능 포지션: {len(rows)}건")
    print()
    print(f"  {'exit_day':>10} | {'avg_pnl':>8} | {'win_rate':>9} | {'vs_forced':>10}")
    print("  " + "-" * 46)

    actual_avg = sum(r["actual_pnl_pct"] for r in rows) / len(rows)

    for cal_day in [3, 4, 5, 6, 7, 8, 10]:
        pnls = []
        for r in rows:
            # cal_day 이하에서 가장 가까운 거래일의 PnL
            days_avail = [d for d in r["pnl_by_day"] if d <= cal_day]
            if days_avail:
                pnls.append(r["pnl_by_day"][max(days_avail)])
        if pnls:
            avg = np.mean(pnls)
            wr  = np.mean([p > 0 for p in pnls]) * 100
            delta = avg - actual_avg
            marker = " ← 현재" if cal_day == 8 else ("  ★ 최적?" if delta > 3 else "")
            print(f"  day≤{cal_day:2d}   | {avg:>+7.2f}% | {wr:>8.1f}% | {delta:>+9.2f}%p {marker}")

    print()
    # 개별 포지션 경로 시각화 (상위 5개 손실 사례)
    worst = sorted(rows, key=lambda r: r["actual_pnl_pct"])[:5]
    print("  [최대손실 5건 — 조기 청산 효과]")
    print(f"  {'ticker':<6} {'entry':>10} {'exit':>10} {'actual':>8} {'day4':>7} {'day5':>7} {'day6':>7}")
    print("  " + "-" * 62)
    for r in worst:
        d4 = r["pnl_by_day"].get(max([d for d in r["pnl_by_day"] if d<=4], default=None) or -1, float("nan"))
        d5 = r["pnl_by_day"].get(max([d for d in r["pnl_by_day"] if d<=5], default=None) or -1, float("nan"))
        d6 = r["pnl_by_day"].get(max([d for d in r["pnl_by_day"] if d<=6], default=None) or -1, float("nan"))
        def fmt(v): return f"{v:>+6.1f}%" if not np.isnan(v) else "    N/A"
        print(
            f"  {r['ticker']:<6} {str(r['entry_date'].date()):>10} {str(r['exit_date'].date()):>10}"
            f" {r['actual_pnl_pct']:>+7.2f}%"
            f" {fmt(d4)} {fmt(d5)} {fmt(d6)}"
        )


# ============================================================
# Phase 2: 2D 파라미터 그리드 서치
# ============================================================

DATE_FULL_START = date(2025, 3, 3)
DATE_FULL_END   = date(2026, 2, 17)
DATE_IS_END     = date(2025, 9, 30)
DATE_OOS_START  = date(2025, 10, 1)

# 그리드 정의
TAKE_PROFIT_GRID = [4.0, 4.5, 5.0, 5.9, 6.5, 7.5, 8.5]
HOLD_DAYS_GRID   = [4, 5, 6, 7, 8, 10]

N_JOBS = int(os.environ.get("N_JOBS", "20"))


def run_bt(take_profit: float, hold_days: int, start: date, end: date) -> dict:
    params = deepcopy(D2S_ENGINE_V2)
    params["take_profit_pct"]      = take_profit
    params["optimal_hold_days_max"] = hold_days

    bt = D2SBacktestV2(params=params, start_date=start, end_date=end, use_fees=True)
    bt.run(verbose=False)
    r = bt.report()

    forced = sum(1 for t in bt.trades if t.side == "SELL" and "hold_days" in t.reason)
    return {
        "take_profit": take_profit,
        "hold_days":   hold_days,
        "total_return": r["total_return_pct"],
        "win_rate":     r["win_rate"],
        "mdd":          r["mdd_pct"],
        "sharpe":       r["sharpe_ratio"],
        "buy_trades":   r["buy_trades"],
        "avg_pnl":      r["avg_pnl_pct"],
        "forced_exit":  forced,
    }


def _run_bt_task(args: tuple) -> dict:
    """multiprocessing.Pool 용 래퍼."""
    tp, hd, period_name, pstart, pend = args
    r = run_bt(tp, hd, pstart, pend)
    r["period"] = period_name
    return r


def phase2_grid() -> None:
    print("\n" + "=" * 65)
    print("  Phase 2 Grid — take_profit × hold_days 2D 파라미터 서치")
    print("=" * 65)

    periods = [
        ("FULL", DATE_FULL_START, DATE_FULL_END),
        ("IS",   DATE_FULL_START, DATE_IS_END),
        ("OOS",  DATE_OOS_START,  DATE_FULL_END),
    ]

    # baseline 값 먼저 기록
    baseline = {"take_profit": 5.9, "hold_days": 7}

    combos = [
        (tp, hd, period_name, pstart, pend)
        for tp in TAKE_PROFIT_GRID
        for hd in HOLD_DAYS_GRID
        for period_name, pstart, pend in periods
    ]
    total = len(combos)
    n_workers = min(N_JOBS, total)
    print(f"    {total}개 조합 × {n_workers} workers 병렬 실행...")

    with Pool(n_workers) as pool:
        all_results = pool.map(_run_bt_task, combos)

    print(f"    완료: {len(all_results)}/{total}\n")

    # 결과 DataFrame
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(
        _PROJECT_ROOT / "data" / "results" / "analysis" / "exit_params_grid.csv",
        index=False,
    )

    # ── 주요 결과 출력 ──
    for period_name in ["FULL", "IS", "OOS"]:
        sub = res_df[res_df["period"] == period_name].copy()
        base_row = sub[(sub["take_profit"] == 5.9) & (sub["hold_days"] == 7)].iloc[0]

        print(f"  [{period_name}] 기준: tp=5.9% hd=7  수익률={base_row['total_return']:+.2f}%  Sharpe={base_row['sharpe']:.3f}")
        print(f"  {'TP':>5} {'HD':>5} {'수익률':>9} {'MDD':>7} {'Sharpe':>7} {'매수':>5} {'강제청산':>7} {'Δ수익률':>9}")
        print("  " + "-" * 62)

        # 상위 10개 (Sharpe 기준)
        top = sub.nlargest(10, "sharpe")
        for _, row in top.iterrows():
            delta = row["total_return"] - base_row["total_return"]
            is_base = "★" if (row["take_profit"] == 5.9 and row["hold_days"] == 7) else " "
            print(
                f"  {is_base}{row['take_profit']:>4.1f}% {row['hold_days']:>4}일"
                f" {row['total_return']:>+8.2f}%"
                f" {row['mdd']:>6.1f}%"
                f" {row['sharpe']:>7.3f}"
                f" {row['buy_trades']:>5}건"
                f" {row['forced_exit']:>5}건"
                f" {delta:>+8.2f}%p"
            )
        print()

    # ── IS vs OOS 안정성 분석 ──
    print("  [IS vs OOS 안정성 — 동일 파라미터 조합]")
    print(f"  {'TP':>5} {'HD':>5} {'IS 수익률':>10} {'OOS 수익률':>11} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'안정성':>8}")
    print("  " + "-" * 64)

    full = res_df[res_df["period"] == "FULL"]
    is_  = res_df[res_df["period"] == "IS"]
    oos  = res_df[res_df["period"] == "OOS"]

    is_b  = is_[(is_["take_profit"] == 5.9) & (is_["hold_days"] == 7)].iloc[0]
    oos_b = oos[(oos["take_profit"] == 5.9) & (oos["hold_days"] == 7)].iloc[0]

    # OOS 개선이 가장 큰 조합 상위 15개
    merged = is_.merge(oos, on=["take_profit", "hold_days"], suffixes=("_is", "_oos"))
    merged["oos_delta"] = merged["total_return_oos"] - oos_b["total_return"]
    merged["is_delta"]  = merged["total_return_is"]  - is_b["total_return"]
    merged["stability"] = merged["oos_delta"] - merged["is_delta"].abs()  # OOS 개선 - IS 손실

    top_stable = merged.nlargest(12, "oos_delta")
    for _, row in top_stable.iterrows():
        is_base = "★" if (row["take_profit"] == 5.9 and row["hold_days"] == 7) else " "
        print(
            f"  {is_base}{row['take_profit']:>4.1f}% {row['hold_days']:>4}일"
            f" {row['total_return_is']:>+9.2f}%"
            f" {row['total_return_oos']:>+10.2f}%"
            f" {row['sharpe_is']:>10.3f}"
            f" {row['sharpe_oos']:>10.3f}"
            f" {row['oos_delta']:>+7.2f}%p"
        )
    print()

    # ── 결론 ──
    best_oos = oos.nlargest(1, "total_return").iloc[0]
    best_full = full.nlargest(1, "sharpe").iloc[0]
    print("  [결론 요약]")
    print(f"  OOS 최고 수익률: tp={best_oos['take_profit']}% hd={best_oos['hold_days']}일 → {best_oos['total_return']:+.2f}%")
    print(f"  FULL 최고 Sharpe: tp={best_full['take_profit']}% hd={best_full['hold_days']}일 → Sharpe {best_full['sharpe']:.3f}")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    phase1_eda()
    phase2_grid()

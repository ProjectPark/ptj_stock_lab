#!/usr/bin/env python3
"""
손절 라인 최적화 — 1년 vs 1개월 비교
=====================================
다양한 손절 라인(-1% ~ -10%, 손절 없음)으로 백테스트를 반복 실행하여
최적의 손절 값을 찾는다.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from simulation.backtests.backtest import BacktestEngine, fetch_backtest_data

# ============================================================
# 손절 라인 후보
# ============================================================
STOP_LOSS_VALUES = [
    -1.0, -1.5, -2.0, -2.5, -3.0, -3.5,
    -4.0, -5.0, -6.0, -8.0, -10.0, -999.0,  # -999 = 사실상 손절 없음
]

LABELS = {v: f"{v:.1f}%" for v in STOP_LOSS_VALUES}
LABELS[-999.0] = "없음"


def run_optimization(
    start_date: date,
    end_date: date,
    label: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """주어진 기간에 대해 모든 손절 라인으로 백테스트 실행."""
    results = []

    for sl in STOP_LOSS_VALUES:
        engine = BacktestEngine(
            initial_capital=1000.0,
            start_date=start_date,
            end_date=end_date,
            stop_loss_pct=sl,
        )
        engine.run(verbose=False, data=data)
        s = engine.summary()
        s["stop_loss"] = LABELS[sl]
        s["stop_loss_raw"] = sl
        results.append(s)

    df = pd.DataFrame(results)
    return df


def print_table(df: pd.DataFrame, title: str) -> None:
    """결과 테이블을 보기 좋게 출력."""
    print()
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(
        f"  {'손절라인':>8s}  {'최종자산':>10s}  {'수익률':>8s}  "
        f"{'MDD':>7s}  {'Sharpe':>7s}  {'승률':>6s}  "
        f"{'손절횟수':>8s}  {'손절손실':>10s}"
    )
    print("-" * 90)

    # 최고 수익률 행 표시용
    best_idx = df["total_return_pct"].idxmax()

    for i, row in df.iterrows():
        marker = " <-- BEST" if i == best_idx else ""
        print(
            f"  {row['stop_loss']:>8s}  "
            f"${row['final_equity']:>9,.2f}  "
            f"{row['total_return_pct']:>+7.2f}%  "
            f"{row['mdd_pct']:>6.2f}%  "
            f"{row['sharpe']:>7.2f}  "
            f"{row['win_rate']:>5.1f}%  "
            f"{row['stop_loss_count']:>8d}  "
            f"${row['stop_loss_pnl']:>+9.2f}"
            f"{marker}"
        )
    print("=" * 90)


def print_comparison(df_year: pd.DataFrame, df_month: pd.DataFrame) -> None:
    """1년 vs 1개월 최적값 비교."""
    best_year = df_year.loc[df_year["total_return_pct"].idxmax()]
    best_month = df_month.loc[df_month["total_return_pct"].idxmax()]

    print()
    print("=" * 60)
    print("  최적 손절 라인 비교")
    print("=" * 60)
    print(f"  {'':15s}  {'1년':>12s}  {'최근 1개월':>12s}")
    print("-" * 60)
    print(f"  {'최적 손절':15s}  {best_year['stop_loss']:>12s}  {best_month['stop_loss']:>12s}")
    print(f"  {'수익률':15s}  {best_year['total_return_pct']:>+11.2f}%  {best_month['total_return_pct']:>+11.2f}%")
    print(f"  {'MDD':15s}  {best_year['mdd_pct']:>11.2f}%  {best_month['mdd_pct']:>11.2f}%")
    print(f"  {'Sharpe':15s}  {best_year['sharpe']:>12.2f}  {best_month['sharpe']:>12.2f}")
    print(f"  {'승률':15s}  {best_year['win_rate']:>11.1f}%  {best_month['win_rate']:>11.1f}%")
    print(f"  {'손절 횟수':15s}  {best_year['stop_loss_count']:>12.0f}  {best_month['stop_loss_count']:>12.0f}")
    print("=" * 60)


def plot_results(df_year: pd.DataFrame, df_month: pd.DataFrame) -> None:
    """결과 차트 생성."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stop Loss Optimization", fontsize=14, fontweight="bold")

    for col_idx, (df, label) in enumerate([(df_year, "1년"), (df_month, "최근 1개월")]):
        # 손절 없음(-999)은 그래프에서 제외하고 따로 표시
        plot_df = df[df["stop_loss_raw"] > -100].copy()
        x = plot_df["stop_loss_raw"].values
        best_sl = plot_df.loc[plot_df["total_return_pct"].idxmax(), "stop_loss_raw"]

        # 수익률
        ax = axes[0][col_idx]
        colors = ["#4CAF50" if v == best_sl else "#2196F3" for v in x]
        ax.bar(x, plot_df["total_return_pct"].values, width=0.4, color=colors)
        ax.set_title(f"{label} — 수익률 (%)")
        ax.set_xlabel("Stop Loss (%)")
        ax.set_ylabel("Return (%)")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

        # 손절 없음 값 표기
        no_sl = df[df["stop_loss_raw"] == -999.0]
        if not no_sl.empty:
            no_sl_ret = no_sl.iloc[0]["total_return_pct"]
            ax.axhline(y=no_sl_ret, color="red", linestyle=":", alpha=0.7,
                       label=f"손절 없음: {no_sl_ret:+.2f}%")
            ax.legend(fontsize=8)

        # MDD
        ax = axes[1][col_idx]
        ax.bar(x, plot_df["mdd_pct"].values, width=0.4, color="#FF5722", alpha=0.7)
        ax.set_title(f"{label} — 최대 낙폭 (%)")
        ax.set_xlabel("Stop Loss (%)")
        ax.set_ylabel("MDD (%)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = config.CHART_DIR / "stoploss_optimization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  차트 저장: {out}")


def main():
    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print("  ║   손절 라인 최적화 — 1년 vs 1개월 비교       ║")
    print("  ╚═══════════════════════════════════════════════╝")

    # 기간 정의
    end_date = date(2026, 2, 17)
    start_year = date(2025, 2, 18)
    start_month = end_date - timedelta(days=30)

    # 데이터 한 번만 로드
    print("\n[1/3] 데이터 로드")
    fetch_start = start_year - timedelta(days=10)
    data = fetch_backtest_data(fetch_start, end_date)

    # 1년 최적화
    print(f"\n[2/3] 1년 시뮬레이션 ({start_year} ~ {end_date})")
    print(f"  {len(STOP_LOSS_VALUES)}개 손절 라인 테스트 중...", end="", flush=True)
    df_year = run_optimization(start_year, end_date, "1년", data)
    print(" 완료!")
    print_table(df_year, f"1년 백테스트 ({start_year} ~ {end_date})")

    # 1개월 최적화
    print(f"\n[3/3] 최근 1개월 시뮬레이션 ({start_month} ~ {end_date})")
    print(f"  {len(STOP_LOSS_VALUES)}개 손절 라인 테스트 중...", end="", flush=True)
    df_month = run_optimization(start_month, end_date, "1개월", data)
    print(" 완료!")
    print_table(df_month, f"최근 1개월 백테스트 ({start_month} ~ {end_date})")

    # 비교
    print_comparison(df_year, df_month)

    # 차트
    plot_results(df_year, df_month)

    # CSV 저장
    combined = pd.concat([
        df_year.assign(period="1년"),
        df_month.assign(period="1개월"),
    ])
    (config.RESULTS_DIR / "optimization").mkdir(parents=True, exist_ok=True)
    csv_path = config.RESULTS_DIR / "optimization" / "stoploss_optimization.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  결과 CSV: {csv_path}")
    print()


if __name__ == "__main__":
    main()

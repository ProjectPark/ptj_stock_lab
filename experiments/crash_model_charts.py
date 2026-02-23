"""
Polymarket 급락 감지 모델 — 시각화
====================================
3패널 차트:
  1. 누적 수익률 (전략 vs SOXL vs TQQQ)
  2. Crash Score + VIX 타임라인
  3. 포지션 비중 스택 (SOXL / TQQQ / MSTZ)

저장: docs/charts/crash_model_analysis.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.polymarket_crash_model import (
    load_poly_signals,
    load_market_data,
    compute_crash_score,
    backtest,
    assign_position,
    TRAIN_START, TRAIN_END, TARGET_START, TARGET_END,
)

CHART_DIR = ROOT / "docs/charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_COLOR  = "#e8f4fd"
TARGET_COLOR = "#fff3e0"
CRASH_COLOR  = "#ffebee"


def build_full_result() -> pd.DataFrame:
    poly_df   = load_poly_signals()
    market_df = load_market_data()
    merged    = poly_df.join(market_df, how="left")
    merged    = compute_crash_score(merged)
    result    = backtest(merged, TRAIN_START, TARGET_END)

    # 포지션 비중 컬럼 추가
    weights = result["crash_score"].apply(assign_position)
    result["w_soxl"] = weights.apply(lambda d: d.get("SOXL", 0))
    result["w_tqqq"] = weights.apply(lambda d: d.get("TQQQ", 0))
    result["w_mstz"] = weights.apply(lambda d: d.get("MSTZ", 0))
    return result


def plot(result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(18, 14),
        gridspec_kw={"height_ratios": [3, 2, 2]},
        sharex=True,
    )
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "Polymarket Crash Detection Model — Backtest Analysis\n"
        f"Train: {TRAIN_START} ~ {TRAIN_END}   |   Target: {TARGET_START} ~ {TARGET_END}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    train_mask  = (result.index >= TRAIN_START)  & (result.index <= TRAIN_END)
    target_mask = (result.index >= TARGET_START) & (result.index <= TARGET_END)

    # ── 배경 영역 공통 처리 ──────────────────────
    for ax in axes:
        ax.set_facecolor("#fafafa")
        ax.axvspan(result.index[train_mask][0],  result.index[train_mask][-1],
                   color=TRAIN_COLOR,  alpha=0.6, zorder=0, label="_train")
        ax.axvspan(result.index[target_mask][0], result.index[target_mask][-1],
                   color=TARGET_COLOR, alpha=0.6, zorder=0, label="_target")
        ax.axvline(pd.Timestamp(TARGET_START), color="#ff9800", lw=1.2,
                   ls="--", alpha=0.8, zorder=1)

    # ── Panel 1: 누적 수익률 ─────────────────────
    ax1 = axes[0]
    ax1.plot(result.index, (result["strategy_cum"] - 1) * 100,
             color="#1565c0", lw=2.0, label="Crash Model (연속 사이즈)", zorder=3)
    ax1.plot(result.index, (result["soxl_cum"]    - 1) * 100,
             color="#c62828", lw=1.5, ls="-", alpha=0.8, label="SOXL 보유", zorder=2)
    ax1.plot(result.index, (result["tqqq_cum"]    - 1) * 100,
             color="#2e7d32", lw=1.5, ls="-", alpha=0.8, label="TQQQ 보유", zorder=2)
    ax1.axhline(0, color="#666", lw=0.8, ls="--")

    # 급락 구간 표시 (SOXL < -5%)
    crash_days = result[result["SOXL_ret"].fillna(0) < -0.05]
    for dt in crash_days.index:
        ax1.axvline(dt, color="#ef5350", lw=0.8, alpha=0.5, zorder=1)

    ax1.set_ylabel("누적 수익률 (%)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 최종 수익률 레이블
    for series, col, label in [
        (result["strategy_cum"], "#1565c0", "Model"),
        (result["soxl_cum"],     "#c62828", "SOXL"),
        (result["tqqq_cum"],     "#2e7d32", "TQQQ"),
    ]:
        last_val = (series.iloc[-1] - 1) * 100
        ax1.annotate(
            f"{label}: {last_val:+.1f}%",
            xy=(result.index[-1], last_val),
            xytext=(8, 0), textcoords="offset points",
            color=col, fontsize=8, va="center",
        )

    # ── Panel 2: Crash Score + VIX ──────────────
    ax2 = axes[1]
    ax2.fill_between(result.index, result["crash_score"],
                     alpha=0.25, color="#7b1fa2", zorder=2)
    ax2.plot(result.index, result["crash_score"],
             color="#7b1fa2", lw=1.5, label="Crash Score", zorder=3)

    # 모드 임계선
    thresholds = [(0.15, "BULL",    "#2196f3"),
                  (0.40, "LONG-",   "#ff9800"),
                  (0.50, "NEUTRAL", "#9e9e9e"),
                  (0.80, "CRASH",   "#f44336")]
    for val, lbl, col in thresholds:
        ax2.axhline(val, color=col, lw=0.8, ls="--", alpha=0.7)
        ax2.text(result.index[5], val + 0.01, lbl,
                 color=col, fontsize=7, alpha=0.9)

    # VIX 보조축
    if "VIX" in result.columns:
        ax2v = ax2.twinx()
        ax2v.plot(result.index, result["VIX"].fillna(method="ffill"),
                  color="#ef6c00", lw=1.0, ls=":", alpha=0.6, label="VIX")
        ax2v.set_ylabel("VIX", fontsize=9, color="#ef6c00")
        ax2v.tick_params(axis="y", labelcolor="#ef6c00")
        ax2v.set_ylim(10, 60)

    ax2.set_ylabel("Crash Score (0~1)", fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: 포지션 비중 스택 ────────────────
    ax3 = axes[2]
    ax3.stackplot(
        result.index,
        result["w_soxl"] * 100,
        result["w_tqqq"] * 100,
        result["w_mstz"] * 100,
        labels=["SOXL (롱 3x)", "TQQQ (롱 3x)", "MSTZ (인버스 2x)"],
        colors=["#1565c0", "#388e3c", "#c62828"],
        alpha=0.75,
    )
    ax3.set_ylabel("포지션 비중 (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax3.grid(axis="y", alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── X축 포맷 ────────────────────────────────
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    # ── 범례 패치 ───────────────────────────────
    train_patch  = mpatches.Patch(color=TRAIN_COLOR,  alpha=0.8, label=f"훈련 기간 ({TRAIN_START[:7]}~{TRAIN_END[:7]})")
    target_patch = mpatches.Patch(color=TARGET_COLOR, alpha=0.8, label=f"타겟 기간 ({TARGET_START[:7]}~{TARGET_END[:7]})")
    crash_line   = mpatches.Patch(color="#ef5350",    alpha=0.5, label="SOXL 급락 (-5% 이상)")
    fig.legend(
        handles=[train_patch, target_patch, crash_line],
        loc="lower center", ncol=3, fontsize=9,
        bbox_to_anchor=(0.5, 0.01), framealpha=0.9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = CHART_DIR / "crash_model_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"차트 저장: {out_path}")


def main() -> None:
    print("데이터 로드 및 백테스트 실행 중...")
    result = build_full_result()
    print(f"  기간: {result.index[0].date()} ~ {result.index[-1].date()} ({len(result)}일)")
    print("차트 생성 중...")
    plot(result)


if __name__ == "__main__":
    main()

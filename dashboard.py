"""
PTJ 매매법 - 정적 대시보드 (matplotlib 6-panel PNG)
===================================================
generate_dashboard(data, changes, signals) -> Path
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ── Color palette ────────────────────────────────────────────────────────
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
COLOR_GRAY = "#7f7f7f"
COLOR_LIGHT_GREEN = "#d5f5d5"
COLOR_LIGHT_RED = "#fdd5d5"
COLOR_GOLD = "#e6ac00"


# ── Helpers ──────────────────────────────────────────────────────────────
def _safe_pct(value: float | None) -> str:
    """Format a percentage value safely."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _bar_color(signal: str) -> str:
    """Map signal string to bar color."""
    mapping = {"SELL": COLOR_GREEN, "ENTRY": COLOR_RED, "HOLD": COLOR_GRAY}
    return mapping.get(signal, COLOR_GRAY)


def _pct_color(value: float) -> str:
    """Return green for positive, red for negative."""
    return COLOR_GREEN if value >= 0 else COLOR_RED


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a price series to base 100."""
    first = series.iloc[0]
    if first == 0 or pd.isna(first):
        return series
    return (series / first) * 100


# ── Panel 1: 시황 요약 ──────────────────────────────────────────────────
def _panel_market_summary(ax: plt.Axes, signals: dict) -> None:
    """Text-based market summary panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    gold = signals.get("gold", {})
    bearish = signals.get("bearish", {})
    stop_loss = signals.get("stop_loss", [])

    gold_warning = gold.get("warning", False)

    # Background color
    bg_color = COLOR_LIGHT_RED if gold_warning else COLOR_LIGHT_GREEN
    ax.set_facecolor(bg_color)

    # Title
    status_text = "매매 금지" if gold_warning else "매매 가능"
    status_color = COLOR_RED if gold_warning else COLOR_GREEN
    ax.text(
        0.5, 0.92, f"시황 요약 — {status_text}",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        ha="center", va="top", color=status_color,
    )

    # Market data lines
    gold_pct = gold.get("change_pct", 0.0)
    spy_pct = bearish.get("spy_pct", 0.0)
    qqq_pct = bearish.get("qqq_pct", 0.0)

    lines = [
        f"GLD (금): {_safe_pct(gold_pct)}",
        f"SPY (S&P 500): {_safe_pct(spy_pct)}",
        f"QQQ (나스닥): {_safe_pct(qqq_pct)}",
    ]

    # Market condition
    if bearish.get("market_down") and bearish.get("gold_up"):
        lines.append("")
        lines.append("시장 하락 + 금 상승 → 금 2x ETF 검토")
    elif bearish.get("market_down"):
        lines.append("")
        lines.append("시장 하락 → 방어주 검토")

    # Stop loss alerts
    if stop_loss:
        lines.append("")
        lines.append("── 손절 경고 ──")
        for alert in stop_loss:
            ticker = alert.get("ticker", "?")
            chg = alert.get("change_pct", 0.0)
            lines.append(f"  {ticker}: {_safe_pct(chg)} (손절 라인 도달)")

    body = "\n".join(lines)
    ax.text(
        0.5, 0.78, body,
        transform=ax.transAxes, fontsize=9, ha="center", va="top",
        linespacing=1.6,
    )


# ── Panel 2: 쌍둥이 페어 갭 (horizontal bar) ────────────────────────────
def _panel_twin_gap(ax: plt.Axes, signals: dict) -> None:
    """Horizontal bar chart for twin pair gaps."""
    twin_pairs = signals.get("twin_pairs", [])

    if not twin_pairs:
        ax.text(0.5, 0.5, "쌍둥이 데이터 없음", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=COLOR_GRAY)
        ax.set_title("쌍둥이 페어 갭", fontsize=12, fontweight="bold")
        ax.axis("off")
        return

    pair_names = [p.get("pair", "?") for p in twin_pairs]
    gaps = [p.get("gap", 0.0) for p in twin_pairs]
    sigs = [p.get("signal", "HOLD") for p in twin_pairs]
    colors = [_bar_color(s) for s in sigs]

    y_pos = np.arange(len(pair_names))

    bars = ax.barh(y_pos, gaps, color=colors, height=0.5, alpha=0.85, edgecolor="white")

    # Labels on bars
    for i, (bar, gap, sig) in enumerate(zip(bars, gaps, sigs)):
        label = f"{gap:+.2f}% [{sig}]"
        x_offset = 0.05 if gap >= 0 else -0.05
        ha = "left" if gap >= 0 else "right"
        ax.text(
            gap + x_offset, i, label,
            va="center", ha=ha, fontsize=8, fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_names, fontsize=9)
    ax.set_title("쌍둥이 페어 갭", fontsize=12, fontweight="bold")
    ax.set_xlabel("갭 (%)", fontsize=9)

    # Threshold lines
    sell_thresh = config.PAIR_GAP_SELL_THRESHOLD
    entry_thresh = config.PAIR_GAP_ENTRY_THRESHOLD

    for thresh in [sell_thresh, -sell_thresh]:
        ax.axvline(thresh, color=COLOR_GREEN, linestyle="--", linewidth=0.8, alpha=0.6)
    for thresh in [entry_thresh, -entry_thresh]:
        ax.axvline(thresh, color=COLOR_RED, linestyle="--", linewidth=0.8, alpha=0.6)

    # Add threshold labels at the top
    ylim_top = len(pair_names) - 0.5
    ax.text(sell_thresh, ylim_top + 0.15, f"매도 {sell_thresh}%",
            fontsize=6, color=COLOR_GREEN, ha="center", va="bottom")
    ax.text(-sell_thresh, ylim_top + 0.15, f"매도 -{sell_thresh}%",
            fontsize=6, color=COLOR_GREEN, ha="center", va="bottom")
    ax.text(entry_thresh, ylim_top + 0.15, f"진입 {entry_thresh}%",
            fontsize=6, color=COLOR_RED, ha="center", va="bottom")
    ax.text(-entry_thresh, ylim_top + 0.15, f"진입 -{entry_thresh}%",
            fontsize=6, color=COLOR_RED, ha="center", va="bottom")

    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.2)

    # Expand x range to fit labels
    max_abs = max(abs(g) for g in gaps) if gaps else 1.0
    margin = max(max_abs * 0.4, entry_thresh + 0.5)
    ax.set_xlim(-margin, margin)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLOR_GREEN, label="SELL (매도)"),
        mpatches.Patch(color=COLOR_RED, label="ENTRY (진입)"),
        mpatches.Patch(color=COLOR_GRAY, label="HOLD (관망)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7)


# ── Panel 3: 조건부 매매 (traffic light) ─────────────────────────────────
def _panel_conditional(ax: plt.Axes, signals: dict) -> None:
    """Traffic light style display for conditional trading signals."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("조건부 매매 (COIN)", fontsize=12, fontweight="bold")

    cond = signals.get("conditional", {})
    triggers = cond.get("triggers", {})
    all_positive = cond.get("all_positive", False)
    target_pct = cond.get("target_pct", 0.0)

    trigger_tickers = config.CONDITIONAL_TRIGGERS  # ["ETHU", "XXRP", "SOLT"]

    # Draw trigger boxes — evenly spaced across the top half
    n_triggers = len(trigger_tickers)
    box_width = 0.22
    box_height = 0.20
    spacing = 0.8 / max(n_triggers, 1)
    start_x = 0.5 - (n_triggers - 1) * spacing / 2

    for i, ticker in enumerate(trigger_tickers):
        cx = start_x + i * spacing
        cy = 0.70

        trig_info = triggers.get(ticker, {})
        is_positive = trig_info.get("positive", False)
        change_pct = trig_info.get("change_pct", 0.0)

        face_color = COLOR_GREEN if is_positive else COLOR_RED
        edge_color = "#333333"

        # Draw rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (cx - box_width / 2, cy - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=face_color, edgecolor=edge_color, linewidth=1.5, alpha=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Ticker name
        ticker_name = config.TICKERS.get(ticker, {}).get("name", ticker)
        ax.text(cx, cy + 0.03, ticker, transform=ax.transAxes,
                fontsize=10, fontweight="bold", ha="center", va="center", color="white")
        ax.text(cx, cy - 0.05, f"{_safe_pct(change_pct)}", transform=ax.transAxes,
                fontsize=8, ha="center", va="center", color="white")

    # Arrow or indicator
    arrow_y = 0.48
    if all_positive:
        ax.annotate(
            "", xy=(0.5, 0.38), xytext=(0.5, arrow_y),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=2.5),
        )
        ax.text(0.5, 0.44, "3종목 전체 양전", transform=ax.transAxes,
                fontsize=8, ha="center", va="center", color=COLOR_GREEN, fontweight="bold")
    else:
        positive_count = sum(1 for v in triggers.values() if v.get("positive", False))
        ax.text(0.5, 0.44, f"양전 {positive_count}/{n_triggers} — 조건 미충족",
                transform=ax.transAxes,
                fontsize=8, ha="center", va="center", color=COLOR_GRAY)

    # COIN target box
    target_color = COLOR_GREEN if all_positive else "#cccccc"
    target_edge = COLOR_GREEN if all_positive else COLOR_GRAY
    target_box = mpatches.FancyBboxPatch(
        (0.30, 0.10), 0.40, 0.22,
        boxstyle="round,pad=0.03",
        facecolor=target_color if all_positive else "#f0f0f0",
        edgecolor=target_edge, linewidth=2.0, alpha=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(target_box)

    target_label = config.CONDITIONAL_TARGET
    target_name = config.TICKERS.get(target_label, {}).get("name", target_label)
    text_color = "white" if all_positive else "#333333"
    status_label = "매수 시그널!" if all_positive else "대기"
    ax.text(0.5, 0.24, f"{target_label} ({target_name})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="center", color=text_color)
    ax.text(0.5, 0.16, f"{_safe_pct(target_pct)}  [{status_label}]",
            transform=ax.transAxes,
            fontsize=9, ha="center", va="center", color=text_color)


# ── Panel 4: 쌍둥이 가격 추이 (line chart) ──────────────────────────────
def _panel_twin_prices(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Normalized price trends for twin pairs (last 30 trading days)."""
    ax.set_title("쌍둥이 가격 추이 (30일, 기준=100)", fontsize=12, fontweight="bold")

    if data.empty:
        ax.text(0.5, 0.5, "가격 데이터 없음", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=COLOR_GRAY)
        return

    line_styles_lead = ["-", "-", "-"]
    line_styles_follow = ["--", "--", "--"]
    colors_pairs = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (key, pair_info) in enumerate(config.TWIN_PAIRS.items()):
        lead_ticker = pair_info["lead"]
        follow_ticker = pair_info["follow"]
        label = pair_info["label"]
        color = colors_pairs[i % len(colors_pairs)]

        for ticker, ls, role in [
            (lead_ticker, line_styles_lead[i], "선행"),
            (follow_ticker, line_styles_follow[i], "후행"),
        ]:
            tdf = data[data["ticker"] == ticker].copy()
            if tdf.empty:
                continue

            tdf = tdf.sort_values("Date")
            # Last 30 trading days
            tdf = tdf.tail(30)
            if len(tdf) < 2:
                continue

            prices = tdf["Close"].values.astype(float)
            normalized = _normalize_series(pd.Series(prices))
            dates = pd.to_datetime(tdf["Date"].values)

            ax.plot(
                dates, normalized,
                linestyle=ls, color=color, linewidth=1.3, alpha=0.9,
                label=f"{ticker} ({role})",
            )

    ax.axhline(100, color="black", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_ylabel("정규화 가격 (기준=100)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(axis="both", alpha=0.2)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)


# ── Panel 5: 하락장 방어주 (bar chart) ───────────────────────────────────
def _panel_bearish(ax: plt.Axes, signals: dict) -> None:
    """Bar chart for bearish / defensive tickers + GLD."""
    ax.set_title("하락장 방어주", fontsize=12, fontweight="bold")

    bearish = signals.get("bearish", {})
    picks = bearish.get("bearish_picks", [])

    # Add GLD to the list
    gold = signals.get("gold", {})
    gold_pct = gold.get("change_pct", 0.0)
    all_items = list(picks) + [{"ticker": "GLD", "name": "금 ETF", "change_pct": gold_pct}]

    if not all_items:
        ax.text(0.5, 0.5, "데이터 없음", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=COLOR_GRAY)
        return

    tickers = [item.get("ticker", "?") for item in all_items]
    names = [item.get("name", "") for item in all_items]
    pcts = [item.get("change_pct", 0.0) for item in all_items]
    colors = [_pct_color(p) for p in pcts]

    x_pos = np.arange(len(tickers))
    bars = ax.bar(x_pos, pcts, color=colors, alpha=0.85, edgecolor="white", width=0.6)

    # Value labels on bars
    for bar, pct in zip(bars, pcts):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.05 if y >= 0 else -0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2, y + offset,
            f"{pct:+.2f}%",
            ha="center", va=va, fontsize=8, fontweight="bold",
        )

    # X labels: ticker + name
    labels = [f"{t}\n({n})" for t, n in zip(tickers, names)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("등락률 (%)", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="y", labelsize=8)


# ── Panel 6: 전체 종목 히트맵 ────────────────────────────────────────────
def _panel_heatmap(ax: plt.Axes, changes: dict) -> None:
    """Color-coded grid showing all tickers and their change percentages."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("전체 종목 히트맵", fontsize=12, fontweight="bold")

    if not changes:
        ax.text(0.5, 0.5, "데이터 없음", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=COLOR_GRAY)
        return

    tickers = sorted(changes.keys())
    n = len(tickers)

    if n == 0:
        return

    # Grid layout: determine cols and rows
    n_cols = min(5, n)
    n_rows = math.ceil(n / n_cols)

    # Cell dimensions
    margin_x = 0.04
    margin_y = 0.06
    usable_w = 1.0 - 2 * margin_x
    usable_h = 0.88 - margin_y  # leave room for title
    cell_w = usable_w / n_cols
    cell_h = usable_h / n_rows
    padding = 0.008

    # Compute color scale based on actual data range
    pcts = [changes[t].get("change_pct", 0.0) for t in tickers]
    max_abs = max(abs(p) for p in pcts) if pcts else 1.0
    if max_abs == 0:
        max_abs = 1.0

    for idx, ticker in enumerate(tickers):
        row = idx // n_cols
        col = idx % n_cols

        pct = changes[ticker].get("change_pct", 0.0)

        # Position: top-left origin
        x = margin_x + col * cell_w + padding
        y = (0.88 - margin_y) - (row + 1) * cell_h + padding
        w = cell_w - 2 * padding
        h = cell_h - 2 * padding

        # Color: gradient from red (negative) through white (0) to green (positive)
        intensity = min(abs(pct) / max_abs, 1.0)
        if pct >= 0:
            # Green gradient
            r = 1.0 - 0.6 * intensity
            g = 1.0 - 0.15 * intensity
            b = 1.0 - 0.6 * intensity
        else:
            # Red gradient
            r = 1.0 - 0.05 * intensity
            g = 1.0 - 0.6 * intensity
            b = 1.0 - 0.6 * intensity

        face_color = (r, g, b)

        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.005",
            facecolor=face_color, edgecolor="#999999", linewidth=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Ticker name
        cx = x + w / 2
        cy = y + h / 2

        ticker_name = config.TICKERS.get(ticker, {}).get("name", "")
        display_name = ticker_name if len(ticker_name) <= 8 else ticker_name[:7] + ".."

        # Choose text color for readability
        text_color = "#333333" if intensity < 0.7 else "#111111"

        ax.text(cx, cy + h * 0.15, ticker, transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="center", color=text_color)
        ax.text(cx, cy - h * 0.05, display_name, transform=ax.transAxes,
                fontsize=5.5, ha="center", va="center", color=text_color)
        ax.text(cx, cy - h * 0.25, f"{pct:+.2f}%", transform=ax.transAxes,
                fontsize=7, ha="center", va="center", color=text_color, fontweight="bold")


# ── Main entry point ─────────────────────────────────────────────────────
def generate_dashboard(
    data: pd.DataFrame,
    changes: dict,
    signals: dict,
) -> Path:
    """Generate a 6-panel PTJ trading dashboard and save as PNG.

    Parameters
    ----------
    data : pd.DataFrame
        3 months of daily OHLCV data with columns
        [Date, Open, High, Low, Close, Volume, ticker].
    changes : dict
        Latest per-ticker change data from ``fetch_data.get_latest_changes()``.
    signals : dict
        Signal bundle from ``signals.generate_all_signals()``.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    # Ensure defaults for missing signal keys
    if signals is None:
        signals = {}
    signals.setdefault("gold", {"change_pct": 0.0, "warning": False, "message": ""})
    signals.setdefault("twin_pairs", [])
    signals.setdefault("conditional", {
        "triggers": {}, "all_positive": False,
        "target": config.CONDITIONAL_TARGET, "target_pct": 0.0, "message": "",
    })
    signals.setdefault("stop_loss", [])
    signals.setdefault("bearish", {
        "market_down": False, "gold_up": False,
        "spy_pct": 0.0, "qqq_pct": 0.0, "bearish_picks": [], "message": "",
    })

    if changes is None:
        changes = {}
    if data is None:
        data = pd.DataFrame()

    # Create figure
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 11),
        gridspec_kw={"hspace": 0.35, "wspace": 0.28},
    )

    # Date for suptitle
    date_str = ""
    if changes:
        sample = next(iter(changes.values()), {})
        date_str = sample.get("date", "")
    fig.suptitle(
        f"PTJ 매매 대시보드  [{date_str}]",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Panel 1 (top-left): 시황 요약
    _panel_market_summary(axes[0, 0], signals)

    # Panel 2 (top-center): 쌍둥이 페어 갭
    _panel_twin_gap(axes[0, 1], signals)

    # Panel 3 (top-right): 조건부 매매
    _panel_conditional(axes[0, 2], signals)

    # Panel 4 (bottom-left): 쌍둥이 가격 추이
    _panel_twin_prices(axes[1, 0], data)

    # Panel 5 (bottom-center): 하락장 방어주
    _panel_bearish(axes[1, 1], signals)

    # Panel 6 (bottom-right): 전체 종목 히트맵
    _panel_heatmap(axes[1, 2], changes)

    # Save
    output_path = config.CHART_DIR / "ptj_dashboard.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[Dashboard] 저장 완료: {output_path}")
    return output_path


# ── CLI entry ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import fetch_data

    print("[Dashboard] 데이터 수집 중...")
    all_data = fetch_data.fetch_all()
    if all_data.empty:
        print("[Dashboard] 데이터 수집 실패")
        sys.exit(1)

    chg = fetch_data.get_latest_changes(all_data)

    import signals as sig_mod
    sigs = sig_mod.generate_all_signals(chg)

    path = generate_dashboard(all_data, chg, sigs)
    print(f"[Dashboard] 완료: {path}")

"""Line D — 3축 자금흐름 차트: 넣은 돈 / 굴린 돈 / 번 돈."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd

# 한글 폰트 설정 (macOS)
for font_name in ["Apple SD Gothic Neo", "Nanum Gothic", "AppleGothic"]:
    if any(f.name == font_name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font_name
        break
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.strategies.line_d_history.engine import JunTradeEngine
from simulation.strategies.line_d_history.params import JunTradeParams

OUT_DIR = ROOT / "docs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── 백테스트 실행 ──
    print("백테스트 실행 중...")
    params = JunTradeParams()
    engine = JunTradeEngine(params=params, start_date=date(2023, 10, 1), end_date=date(2026, 2, 17))
    bt_result = engine.run(verbose=False)
    bt_trades = bt_result["closed_trades"]
    bt_snaps = pd.DataFrame(bt_result["daily_snapshots"])
    bt_snaps["date"] = pd.to_datetime(bt_snaps["date"])

    # ── 실거래 데이터 로드 ──
    tx = pd.read_csv(ROOT / "history" / "거래내역_20231006_20260212.csv", encoding="utf-8-sig")
    tx["거래일자"] = pd.to_datetime(tx["거래일자"])
    tx["거래대금_원"] = tx["거래대금_원"].astype(float)

    dep_df = pd.read_csv(ROOT / "data" / "results" / "analysis" / "jun_trade_deposit_events.csv")
    dep_df["date"] = pd.to_datetime(dep_df["date"])

    bal = pd.read_csv(ROOT / "data" / "results" / "analysis" / "jun_trade_daily_balance.csv")
    bal["date"] = pd.to_datetime(bal["date"])

    # ── 일별 매수 / 실현손익 누적 ──
    buy_daily = (
        tx[tx["거래구분"] == "구매"]
        .groupby("거래일자")["거래대금_원"]
        .sum()
        .sort_index()
    )
    buy_daily = buy_daily.reindex(
        pd.date_range(buy_daily.index.min(), buy_daily.index.max(), freq="D"), fill_value=0
    )
    buy_cum = buy_daily.cumsum()

    # 입금 누적 (일별로 펼치기)
    dep_series = dep_df.set_index("date")["deposit"].reindex(
        pd.date_range(dep_df["date"].min(), dep_df["date"].max(), freq="D"), fill_value=0
    ).cumsum()

    # 실현손익 누적 (거래 종료일 기준)
    pnl_daily = bal.set_index("date")["realized_pnl"].sort_index()
    pnl_cum = pnl_daily.cumsum()

    # 만원 단위로 변환
    def to_man(s): return s / 1e4

    # ── 현금 잔고 일별 시리즈 ──
    cash_series = bal.set_index("date")["cash"].sort_index()
    # 매수/매도 일별 합계 (현금흐름 바)
    daily_buy = tx[tx["거래구분"] == "구매"].groupby("거래일자")["거래대금_원"].sum()
    daily_sell = tx[tx["거래구분"] == "판매"].groupby("거래일자")["거래대금_원"].sum()
    dep_events = dep_df.set_index("date")["deposit"]

    # ── 수익률 시리즈 계산 ──
    bal_indexed = bal.set_index("date").sort_index()
    dep_aligned = dep_series.reindex(bal_indexed.index, method="ffill").fillna(0)
    pnl_aligned = bal_indexed["realized_pnl"].cumsum()

    # 누적 ROI: 실현손익 누적 / 당시 투자원가 (굴린 돈 대비)
    invested_aligned = bal_indexed["invested"].replace(0, float("nan"))
    roi_series = (pnl_aligned / invested_aligned) * 100

    # 월별 실현손익 합
    bal_monthly = bal_indexed["realized_pnl"].resample("ME").sum()

    # ── 누적입금 대비 ROI (변경 전 버전) ──
    roi_vs_deposit = (pnl_aligned / dep_aligned.replace(0, float("nan"))) * 100

    # ── 차트 (6패널) ──
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        6, 1, figsize=(16, 26),
        gridspec_kw={"height_ratios": [2, 3, 2, 2, 2, 2]},
    )
    fig.suptitle("JUN Trade — Money Flow: Cash / Deposited / Traded / Earned / ROI / Backtest", fontsize=14, fontweight="bold")

    # ─── 최상단: 현금 총 흐름 ───
    # 현금 잔고 라인
    ax0.fill_between(cash_series.index, 0, to_man(cash_series),
                     alpha=0.15, color="steelblue")
    ax0.plot(cash_series.index, to_man(cash_series),
             color="steelblue", linewidth=1.5, label="현금 잔고")

    # 매수 지출 (아래 방향 빨강)
    ax0_twin = ax0.twinx()
    ax0_twin.bar(daily_buy.index, -to_man(daily_buy),
                 width=1.5, color="#e74c3c", alpha=0.5, label="매수 지출")
    # 매도 수입 (위 방향 초록)
    ax0_twin.bar(daily_sell.index, to_man(daily_sell),
                 width=1.5, color="#2ecc71", alpha=0.5, label="매도 수입")
    # 입금 이벤트 (주황 점선)
    for d, amt in dep_events.items():
        ax0_twin.axvline(x=d, color="darkorange", alpha=0.6, linewidth=1.0, linestyle="--")

    # 입금 범례용 더미
    ax0_twin.plot([], [], color="darkorange", linestyle="--", linewidth=1.5, label="입금 이벤트")
    ax0_twin.axhline(y=0, color="black", linewidth=0.6)

    ax0.set_ylabel("현금 잔고 (만원)", fontsize=10)
    ax0_twin.set_ylabel("일별 입출금 (만원)", fontsize=10)
    ax0.set_title("현금 흐름 — 잔고(선) + 일별 매수지출↓ / 매도수입↑ + 입금(주황)", fontsize=10, color="gray")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}만"))
    ax0_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+,.0f}만"))
    ax0.grid(True, alpha=0.3)

    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax0_twin.get_legend_handles_labels()
    ax0.legend(lines0 + lines1, labels0 + labels1, loc="upper left", fontsize=8)

    # 2025-07-11 전환점 표시
    pivot = pd.Timestamp("2025-07-11")
    ax0.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    # ─── 중단: 넣은 돈 vs 굴린 돈 ───
    ax1.fill_between(buy_cum.index, 0, to_man(buy_cum),
                     alpha=0.08, color="steelblue")
    ax1.plot(buy_cum.index, to_man(buy_cum),
             color="steelblue", linewidth=2,
             label=f"굴린 돈 (누적 매수액)  →  {buy_cum.iloc[-1]/1e8:.1f}억")

    ax1.fill_between(dep_series.index, 0, to_man(dep_series),
                     alpha=0.2, color="orange")
    ax1.plot(dep_series.index, to_man(dep_series),
             color="darkorange", linewidth=2, linestyle="--",
             label=f"넣은 돈 (누적 입금액)  →  {dep_series.iloc[-1]/1e8:.1f}억")

    # 2025-07-11 전환점
    pivot = pd.Timestamp("2025-07-11")
    ax1.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    # 배율 표시
    ratio = buy_cum.iloc[-1] / dep_series.iloc[-1]
    ax1.annotate(
        f"굴린 돈은 넣은 돈의 {ratio:.1f}배\n(같은 돈을 {ratio:.1f}번 돌림)",
        xy=(pivot, to_man(buy_cum.reindex([pivot], method="nearest").iloc[0])),
        xytext=(-120, 30), textcoords="offset points",
        fontsize=10, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
    )

    ax1.set_ylabel("금액 (만원)", fontsize=11)
    ax1.set_title("넣은 돈 vs 실제로 굴린 돈 — 같은 돈이 여러 번 회전", fontsize=10, color="gray")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}만"))

    # ─── 하단: 번 돈 (누적 실현손익) ───
    colors_pos = pnl_cum >= 0
    ax2.fill_between(pnl_cum.index, 0, to_man(pnl_cum),
                     where=colors_pos, alpha=0.3, color="#2ecc71")
    ax2.fill_between(pnl_cum.index, 0, to_man(pnl_cum),
                     where=~colors_pos, alpha=0.3, color="#e74c3c")
    ax2.plot(pnl_cum.index, to_man(pnl_cum),
             color="#2c3e50", linewidth=1.5,
             label=f"번/잃은 돈 (누적 실현손익)  →  {pnl_cum.iloc[-1]/1e4:+,.0f}만원")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    # 최고점 표시
    peak_idx = to_man(pnl_cum).idxmax()
    peak_val = to_man(pnl_cum).max()
    ax2.annotate(f"최고 +{peak_val:,.0f}만",
                 xy=(peak_idx, peak_val),
                 xytext=(10, 10), textcoords="offset points",
                 fontsize=9, color="#27ae60",
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8))

    final_pnl = pnl_cum.iloc[-1] / 1e4
    ax2.annotate(f"최종: {final_pnl:+,.0f}만원",
                 xy=(pnl_cum.index[-1], to_man(pnl_cum).iloc[-1]),
                 xytext=(-80, 15 if final_pnl >= 0 else -25), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 color="#27ae60" if final_pnl >= 0 else "#e74c3c")

    ax2.set_ylabel("금액 (만원)", fontsize=11)
    ax2.set_title("번/잃은 돈 (실현 손익 누적) — 청산 완료된 거래만 포함", fontsize=10, color="gray")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+,.0f}만"))

    # ─── 최하단: 금액 대비 수익률 (ROI%) ───
    # 누적 ROI 선
    ax3.plot(roi_series.index, roi_series,
             color="#8e44ad", linewidth=1.5,
             label=f"누적 수익률 (실현손익 ÷ 누적입금)")
    ax3.fill_between(roi_series.index, 0, roi_series,
                     where=roi_series >= 0, alpha=0.15, color="#2ecc71")
    ax3.fill_between(roi_series.index, 0, roi_series,
                     where=roi_series < 0, alpha=0.15, color="#e74c3c")
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    # 월별 수익률 막대 — 해당월 시작 투자원가 대비
    invested_monthly_start = bal_indexed["invested"].resample("ME").first().ffill().replace(0, float("nan"))
    monthly_roi_proper = (bal_monthly / invested_monthly_start) * 100

    ax3_twin = ax3.twinx()
    bar_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in monthly_roi_proper]
    ax3_twin.bar(monthly_roi_proper.index, monthly_roi_proper,
                 width=20, color=bar_colors, alpha=0.4, label="월별 수익률 (투입원가 대비)")
    ax3_twin.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax3_twin.set_ylabel("월별 수익률 (%)", fontsize=10)
    ax3_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    # 최고/최저 ROI 표시
    peak_roi_idx = roi_series.idxmax()
    peak_roi_val = roi_series.max()
    trough_roi_idx = roi_series.idxmin()
    trough_roi_val = roi_series.min()
    ax3.annotate(f"최고 {peak_roi_val:+.1f}%",
                 xy=(peak_roi_idx, peak_roi_val),
                 xytext=(8, 5), textcoords="offset points",
                 fontsize=9, color="#27ae60", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8))
    ax3.annotate(f"최저 {trough_roi_val:+.1f}%",
                 xy=(trough_roi_idx, trough_roi_val),
                 xytext=(8, -18), textcoords="offset points",
                 fontsize=9, color="#c0392b", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    final_roi = roi_series.iloc[-1]
    ax3.annotate(f"최종 {final_roi:+.1f}%",
                 xy=(roi_series.index[-1], final_roi),
                 xytext=(-70, 10 if final_roi >= 0 else -20), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 color="#27ae60" if final_roi >= 0 else "#c0392b")

    ax3.set_ylabel("누적 수익률 (%)", fontsize=10)
    ax3.set_xlabel("날짜", fontsize=11)
    ax3.set_title("금액 대비 수익률 — 누적(선) + 월별(막대)", fontsize=10, color="gray")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax3.grid(True, alpha=0.3)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3t, labels3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines3t, labels3 + labels3t, loc="upper left", fontsize=8)

    # ─── 최하단: 백테스트 시뮬레이션 PnL (실현 + 미실현) ───
    ax4.fill_between(bt_snaps["date"], 0, bt_snaps["total_pnl_usd"],
                     where=bt_snaps["total_pnl_usd"] >= 0, alpha=0.2, color="steelblue")
    ax4.fill_between(bt_snaps["date"], 0, bt_snaps["total_pnl_usd"],
                     where=bt_snaps["total_pnl_usd"] < 0, alpha=0.2, color="salmon")
    ax4.plot(bt_snaps["date"], bt_snaps["total_pnl_usd"],
             color="steelblue", linewidth=1.5, label="총 PnL (실현+미실현)")
    ax4.plot(bt_snaps["date"], bt_snaps["realized_pnl_usd"],
             color="gray", linewidth=0.8, linestyle="--", alpha=0.7, label="실현 PnL만")
    ax4.axhline(y=0, color="black", linewidth=0.6)
    ax4.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    # MDD
    cum_max = bt_snaps["total_pnl_usd"].cummax()
    drawdown = bt_snaps["total_pnl_usd"] - cum_max
    mdd_idx = drawdown.idxmin()
    mdd_val = drawdown.iloc[mdd_idx]
    ax4.annotate(f"MDD ${mdd_val:,.0f}",
                 xy=(bt_snaps["date"].iloc[mdd_idx], bt_snaps["total_pnl_usd"].iloc[mdd_idx]),
                 xytext=(10, -18), textcoords="offset points",
                 fontsize=8, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    final_bt = bt_snaps["total_pnl_usd"].iloc[-1]
    n_trades = len(bt_trades)
    win_rate = sum(1 for t in bt_trades if t.pnl_pct > 0) / n_trades * 100
    ax4.annotate(f"${final_bt:+,.0f}  |  {n_trades}건  |  승률 {win_rate:.0f}%",
                 xy=(bt_snaps["date"].iloc[-1], final_bt),
                 xytext=(-130, 10), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="steelblue")

    ax4.set_ylabel("PnL (USD)", fontsize=10)
    ax4.set_xlabel("날짜", fontsize=11)
    ax4.set_title("백테스트 시뮬레이션 — 고정 $740/거래, 입금 없음 (실현+미실현 PnL)", fontsize=10, color="gray")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:+,.0f}"))

    # ─── 패널6: 누적 수익률 (누적입금 대비, 변경 전) ───
    ax5.plot(roi_vs_deposit.index, roi_vs_deposit,
             color="#e67e22", linewidth=1.5, label="누적 수익률 (실현손익 ÷ 누적입금)")
    ax5.fill_between(roi_vs_deposit.index, 0, roi_vs_deposit,
                     where=roi_vs_deposit >= 0, alpha=0.2, color="#2ecc71")
    ax5.fill_between(roi_vs_deposit.index, 0, roi_vs_deposit,
                     where=roi_vs_deposit < 0, alpha=0.2, color="#e74c3c")
    ax5.axhline(y=0, color="black", linewidth=0.8)
    ax5.axvline(x=pivot, color="red", alpha=0.4, linewidth=1.2, linestyle=":")

    peak5_idx = roi_vs_deposit.idxmax()
    peak5_val = roi_vs_deposit.max()
    trough5_idx = roi_vs_deposit.idxmin()
    trough5_val = roi_vs_deposit.min()
    ax5.annotate(f"최고 {peak5_val:+.1f}%",
                 xy=(peak5_idx, peak5_val),
                 xytext=(8, 5), textcoords="offset points",
                 fontsize=9, color="#27ae60", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8))
    ax5.annotate(f"최저 {trough5_val:+.1f}%",
                 xy=(trough5_idx, trough5_val),
                 xytext=(8, -18), textcoords="offset points",
                 fontsize=9, color="#c0392b", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))
    final5 = roi_vs_deposit.iloc[-1]
    ax5.annotate(f"최종 {final5:+.1f}%",
                 xy=(roi_vs_deposit.index[-1], final5),
                 xytext=(-80, 10 if final5 >= 0 else -20), textcoords="offset points",
                 fontsize=10, fontweight="bold",
                 color="#27ae60" if final5 >= 0 else "#c0392b")

    ax5.set_ylabel("수익률 (%)", fontsize=10)
    ax5.set_xlabel("날짜", fontsize=11)
    ax5.set_title("누적 수익률 — 실현손익 ÷ 누적입금 (2025-07 대규모 입금 시 희석 효과 포함)", fontsize=10, color="gray")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "line_d_backtest_result.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


if __name__ == "__main__":
    main()

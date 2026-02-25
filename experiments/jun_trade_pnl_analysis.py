"""
JUN 매매법 수익 분석 + 자금 유입 변동 추적
데이터 소스: history/거래내역_20231006_20260212.csv, 수익_거래내역.csv, 손해_거래내역.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
HISTORY = BASE / "history"
OUTPUT = BASE / "data" / "results" / "analysis"
CHARTS = BASE / "docs" / "charts"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

# ── 1. 데이터 로드 ──────────────────────────────────────────────
trades = pd.read_csv(HISTORY / "거래내역_20231006_20260212.csv", parse_dates=["거래일자"])
profit = pd.read_csv(HISTORY / "수익_거래내역.csv", parse_dates=["판매일자", "최초매수일"])
loss = pd.read_csv(HISTORY / "손해_거래내역.csv", parse_dates=["판매일자", "최초매수일"])

# 수익/손해 합치기
profit["구분"] = "수익"
loss["구분"] = "손해"
closed = pd.concat([profit, loss], ignore_index=True).sort_values("판매일자").reset_index(drop=True)

print(f"전체 거래내역: {len(trades)}건")
print(f"청산 거래: {len(closed)}건 (수익 {len(profit)} / 손해 {len(loss)})")

# ── 2. 누적 실현 손익 ──────────────────────────────────────────
closed["cum_pnl"] = closed["실현손익_원"].cumsum()
closed["month"] = closed["판매일자"].dt.to_period("M")

monthly_pnl = closed.groupby("month").agg(
    실현손익=("실현손익_원", "sum"),
    거래수=("실현손익_원", "count"),
    승률=("실현손익_원", lambda x: (x > 0).mean() * 100),
    매도금액합=("판매금액_원", "sum"),
    매수금액합=("매수금액_원", "sum"),
).reset_index()
monthly_pnl["누적손익"] = monthly_pnl["실현손익"].cumsum()
monthly_pnl["month_dt"] = monthly_pnl["month"].dt.to_timestamp()

print("\n── 월별 실현 손익 ──")
print(monthly_pnl[["month", "실현손익", "누적손익", "거래수", "승률"]].to_string(index=False))

# ── 3. 자금 유입 분석 (매수 거래대금 추적) ─────────────────────
buys = trades[trades["거래구분"] == "구매"].copy()
buys["month"] = buys["거래일자"].dt.to_period("M")

monthly_buy = buys.groupby("month").agg(
    총매수금액_원=("거래대금_원", "sum"),
    매수건수=("거래대금_원", "count"),
    평균매수금액=("거래대금_원", "mean"),
    중앙값매수금액=("거래대금_원", "median"),
    최대매수금액=("거래대금_원", "max"),
).reset_index()
monthly_buy["month_dt"] = monthly_buy["month"].dt.to_timestamp()

# 일별 매수 총액
daily_buy = buys.groupby("거래일자").agg(
    일매수총액=("거래대금_원", "sum"),
    매수건수=("거래대금_원", "count"),
).reset_index()

# 자금 유입 변동 감지: 이전 20일 평균 대비 2배 이상 급증
daily_buy["rolling_avg"] = daily_buy["일매수총액"].rolling(20, min_periods=5).mean()
daily_buy["ratio"] = daily_buy["일매수총액"] / daily_buy["rolling_avg"]
capital_jumps = daily_buy[daily_buy["ratio"] > 2.0].copy()

print("\n── 월별 매수 투입 금액 ──")
print(monthly_buy[["month", "총매수금액_원", "매수건수", "평균매수금액", "최대매수금액"]].to_string(index=False))

print(f"\n── 자금 유입 급증일 (20일 평균 대비 2배 이상): {len(capital_jumps)}건 ──")
if len(capital_jumps) > 0:
    print(capital_jumps[["거래일자", "일매수총액", "rolling_avg", "ratio"]].head(20).to_string(index=False))

# ── 4. 1회 매수 금액 분포 변화 ──────────────────────────────────
buys["quarter"] = buys["거래일자"].dt.to_period("Q")
quarter_stats = buys.groupby("quarter").agg(
    중앙값_원=("거래대금_원", "median"),
    평균_원=("거래대금_원", "mean"),
    최대_원=("거래대금_원", "max"),
    건수=("거래대금_원", "count"),
).reset_index()

print("\n── 분기별 1회 매수 금액 변화 ──")
print(quarter_stats.to_string(index=False))

# ── 5. 종목별 총손익 ──────────────────────────────────────────
ticker_pnl = closed.groupby("종목명").agg(
    건수=("실현손익_원", "count"),
    총손익=("실현손익_원", "sum"),
    평균손익률=("손익률_%", "mean"),
    승률=("실현손익_원", lambda x: (x > 0).mean() * 100),
).sort_values("총손익", ascending=False).reset_index()

print("\n── 종목별 총 실현손익 ──")
print(ticker_pnl.to_string(index=False))

# ── 6. 시각화 ──────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(18, 20))
fig.suptitle("JUN 매매법 수익 분석 + 자금 유입 추적\n(2023-10 ~ 2026-02)", fontsize=16, fontweight="bold")

# (1) 월별 실현 손익 바차트
ax = axes[0, 0]
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly_pnl["실현손익"]]
ax.bar(monthly_pnl["month_dt"], monthly_pnl["실현손익"] / 1e6, color=colors, width=25, alpha=0.8)
ax.set_title("월별 실현 손익 (백만원)", fontsize=13)
ax.set_ylabel("백만원")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

# (2) 누적 실현 손익
ax = axes[0, 1]
ax.fill_between(monthly_pnl["month_dt"], monthly_pnl["누적손익"] / 1e6, alpha=0.3, color="#3498db")
ax.plot(monthly_pnl["month_dt"], monthly_pnl["누적손익"] / 1e6, "o-", color="#3498db", markersize=4)
ax.set_title("누적 실현 손익 (백만원)", fontsize=13)
ax.set_ylabel("백만원")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
ax.grid(alpha=0.3)

# (3) 월별 매수 투입 금액
ax = axes[1, 0]
ax.bar(monthly_buy["month_dt"], monthly_buy["총매수금액_원"] / 1e6, color="#9b59b6", width=25, alpha=0.7)
ax.set_title("월별 매수 투입 금액 (백만원)", fontsize=13)
ax.set_ylabel("백만원")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)

# (4) 일별 매수 총액 + 자금 유입 급증 표시
ax = axes[1, 1]
ax.plot(daily_buy["거래일자"], daily_buy["일매수총액"] / 1e6, alpha=0.5, linewidth=0.8, color="#34495e")
ax.plot(daily_buy["거래일자"], daily_buy["rolling_avg"] / 1e6, color="#e67e22", linewidth=1.5, label="20일 이동평균")
if len(capital_jumps) > 0:
    ax.scatter(capital_jumps["거래일자"], capital_jumps["일매수총액"] / 1e6,
               color="red", s=40, zorder=5, label=f"급증일 ({len(capital_jumps)}건)")
ax.set_title("일별 매수 금액 + 자금 유입 급증 감지", fontsize=13)
ax.set_ylabel("백만원")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.legend()
ax.grid(alpha=0.3)

# (5) 1회 매수 금액 분포 변화 (분기별 boxplot)
ax = axes[2, 0]
quarters = buys["quarter"].unique()
bp_data = []
bp_labels = []
for q in sorted(quarters):
    vals = buys[buys["quarter"] == q]["거래대금_원"].values / 1e6
    # clip outliers for visibility
    q99 = np.percentile(vals, 99) if len(vals) > 10 else vals.max()
    bp_data.append(vals[vals <= q99])
    bp_labels.append(str(q))
bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#1abc9c")
    patch.set_alpha(0.6)
ax.set_title("분기별 1회 매수 금액 분포 (백만원)", fontsize=13)
ax.set_ylabel("백만원")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.3)

# (6) 종목별 총손익 Top/Bottom
ax = axes[2, 1]
top_n = 15
ticker_sorted = ticker_pnl.sort_values("총손익")
show = pd.concat([ticker_sorted.head(5), ticker_sorted.tail(10)]).drop_duplicates()
colors_t = ["#2ecc71" if v >= 0 else "#e74c3c" for v in show["총손익"]]
ax.barh(show["종목명"], show["총손익"] / 1e6, color=colors_t, alpha=0.8)
ax.set_title("종목별 총 실현손익 (백만원)", fontsize=13)
ax.set_xlabel("백만원")
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = CHARTS / "jun_trade_pnl_capital_flow.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"\n차트 저장: {chart_path}")

# ── 7. 자금 유입 변동 상세 정리 ─────────────────────────────────
# 월별 "순 투입금액" = 매수총액 - 매도총액 → 양수면 순 자금 유입
sells = trades[trades["거래구분"] == "판매"].copy()
sells["month"] = sells["거래일자"].dt.to_period("M")

monthly_sell = sells.groupby("month").agg(
    총매도금액_원=("거래대금_원", "sum"),
).reset_index()

flow = monthly_buy[["month", "총매수금액_원"]].merge(monthly_sell, on="month", how="outer").fillna(0)
flow["순투입"] = flow["총매수금액_원"] - flow["총매도금액_원"]
flow["누적순투입"] = flow["순투입"].cumsum()
flow["month_dt"] = flow["month"].dt.to_timestamp()

print("\n── 월별 순 자금 투입 (매수 - 매도) ──")
print(flow[["month", "총매수금액_원", "총매도금액_원", "순투입", "누적순투입"]].to_string(index=False))

# 추가 차트: 순 자금 투입 추이
fig2, ax2 = plt.subplots(figsize=(14, 5))
colors_f = ["#3498db" if v >= 0 else "#e74c3c" for v in flow["순투입"]]
ax2.bar(flow["month_dt"], flow["순투입"] / 1e6, color=colors_f, width=25, alpha=0.7, label="월별 순투입")
ax2.plot(flow["month_dt"], flow["누적순투입"] / 1e6, "o-", color="#2c3e50", markersize=5, linewidth=2, label="누적 순투입")
ax2.set_title("월별 순 자금 투입 (매수 - 매도) 및 누적 추이", fontsize=14, fontweight="bold")
ax2.set_ylabel("백만원")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.tick_params(axis="x", rotation=45)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
chart_path2 = CHARTS / "jun_trade_capital_net_flow.png"
plt.savefig(chart_path2, dpi=150, bbox_inches="tight")
print(f"차트 저장: {chart_path2}")

# ── 8. CSV 저장 ──────────────────────────────────────────────
monthly_pnl.to_csv(OUTPUT / "jun_trade_monthly_pnl.csv", index=False)
flow.to_csv(OUTPUT / "jun_trade_capital_flow.csv", index=False)
ticker_pnl.to_csv(OUTPUT / "jun_trade_ticker_pnl.csv", index=False)

if len(capital_jumps) > 0:
    capital_jumps.to_csv(OUTPUT / "jun_trade_capital_jumps.csv", index=False)

print("\n분석 완료!")
print(f"  - 월별 P&L: {OUTPUT / 'jun_trade_monthly_pnl.csv'}")
print(f"  - 자금 흐름: {OUTPUT / 'jun_trade_capital_flow.csv'}")
print(f"  - 종목별 P&L: {OUTPUT / 'jun_trade_ticker_pnl.csv'}")

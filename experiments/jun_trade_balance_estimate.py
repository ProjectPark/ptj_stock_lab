"""
JUN 매매법 — 추정 잔고 역산 (동적 입금 감지)
거래내역으로 일별 현금 잔고 + 보유 포지션 추적
현금이 부족해지는 시점을 외부 입금으로 처리
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent.parent
HISTORY = BASE / "history"
OUTPUT = BASE / "data" / "results" / "analysis"
CHARTS = BASE / "docs" / "charts"

# ── 1. 데이터 로드 ──────────────────────────────────────────
trades = pd.read_csv(HISTORY / "거래내역_20231006_20260212.csv", parse_dates=["거래일자"])
trades["거래수량"] = pd.to_numeric(trades["거래수량"], errors="coerce").fillna(0)
trades = trades.sort_values("거래일자").reset_index(drop=True)

print(f"거래내역: {len(trades)}건, 기간: {trades['거래일자'].min().date()} ~ {trades['거래일자'].max().date()}")

# ── 2. 일별 현금 흐름 계산 ──────────────────────────────────
trades["현금변동"] = trades.apply(
    lambda r: (r["거래대금_원"] - r["수수료_원"]) if r["거래구분"] == "판매"
    else -(r["거래대금_원"] + r["수수료_원"]),
    axis=1,
)

daily_cashflow = trades.groupby("거래일자")["현금변동"].sum().reset_index()
daily_cashflow.columns = ["date", "cashflow"]

# ── 3. 동적 입금 감지 + 현금 잔고 추적 ──────────────────────
# 초기 시드: 첫 거래일 매수금액의 2배로 시작 (보수적 추정)
first_day_buy = abs(daily_cashflow.iloc[0]["cashflow"]) if daily_cashflow.iloc[0]["cashflow"] < 0 else 100_000
seed = first_day_buy * 2
print(f"초기 시드 추정: {seed:,.0f}원")

cash = seed
deposits = []  # (date, amount) — 외부 입금 추정
cash_history = []

for _, row in daily_cashflow.iterrows():
    d = row["date"]
    cf = row["cashflow"]

    cash += cf

    # 현금이 0 이하 → 외부 입금 필요
    if cash < 0:
        deposit = abs(cash) * 1.5  # 부족분의 1.5배 입금 (여유분)
        deposits.append({"date": d, "deposit": deposit})
        cash += deposit

    cash_history.append({"date": d, "cash": cash})

cash_df = pd.DataFrame(cash_history)
deposit_df = pd.DataFrame(deposits) if deposits else pd.DataFrame(columns=["date", "deposit"])

total_deposits = seed + deposit_df["deposit"].sum() if len(deposit_df) > 0 else seed
print(f"외부 입금 감지: {len(deposit_df)}건")
print(f"총 추정 투입금: {total_deposits:,.0f}원 (시드 {seed:,.0f} + 추가 입금 {total_deposits - seed:,.0f})")

if len(deposit_df) > 0:
    print("\n── 추정 외부 입금 이벤트 ──")
    deposit_df["누적입금"] = seed + deposit_df["deposit"].cumsum()
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(deposit_df.to_string(index=False))

# ── 4. 보유 포지션 추적 ─────────────────────────────────────
positions = defaultdict(lambda: {"qty": 0.0, "cost": 0.0})
daily_invested = {}
daily_positions_snapshot = {}

for _, row in trades.iterrows():
    d = row["거래일자"]
    ticker = row["종목명"]
    qty = row["거래수량"]
    amount = row["거래대금_원"]

    if row["거래구분"] == "구매":
        positions[ticker]["qty"] += qty
        positions[ticker]["cost"] += amount
    else:
        if positions[ticker]["qty"] > 0:
            sell_ratio = min(qty / positions[ticker]["qty"], 1.0)
            positions[ticker]["cost"] *= (1 - sell_ratio)
            positions[ticker]["qty"] = max(0, positions[ticker]["qty"] - qty)

    total_invested = sum(p["cost"] for p in positions.values())
    daily_invested[d] = total_invested
    # 보유 종목 수
    daily_positions_snapshot[d] = sum(1 for p in positions.values() if p["qty"] > 0)

invested_df = pd.DataFrame(list(daily_invested.items()), columns=["date", "invested"]).sort_values("date")
pos_count_df = pd.DataFrame(list(daily_positions_snapshot.items()), columns=["date", "num_positions"]).sort_values("date")

# ── 5. 통합 잔고 ────────────────────────────────────────────
balance = cash_df.merge(invested_df, on="date", how="outer").sort_values("date")
balance = balance.merge(pos_count_df, on="date", how="left")
balance["cash"] = balance["cash"].ffill()
balance["invested"] = balance["invested"].ffill().fillna(0)
balance["num_positions"] = balance["num_positions"].ffill().fillna(0).astype(int)
balance["total"] = balance["cash"] + balance["invested"]

# 입금 표시
balance = balance.merge(deposit_df[["date", "deposit"]], on="date", how="left") if len(deposit_df) > 0 else balance.assign(deposit=0)
balance["deposit"] = balance["deposit"].fillna(0)

# 누적 투입금 추적
balance["cum_deposit"] = seed + balance["deposit"].cumsum()

# 실현손익 합치기
profit_df = pd.read_csv(HISTORY / "수익_거래내역.csv", parse_dates=["판매일자"])
loss_df = pd.read_csv(HISTORY / "손해_거래내역.csv", parse_dates=["판매일자"])
closed = pd.concat([profit_df, loss_df])
daily_pnl = closed.groupby("판매일자")["실현손익_원"].sum().reset_index()
daily_pnl.columns = ["date", "realized_pnl"]
balance = balance.merge(daily_pnl, on="date", how="left")
balance["realized_pnl"] = balance["realized_pnl"].fillna(0)

# ── 6. 월별 요약 ────────────────────────────────────────────
balance["month"] = balance["date"].dt.to_period("M")
monthly = balance.groupby("month").agg(
    월말현금=("cash", "last"),
    월말투자원가=("invested", "last"),
    월말총자산=("total", "last"),
    월실현손익=("realized_pnl", "sum"),
    월입금=("deposit", "sum"),
    누적투입=("cum_deposit", "last"),
    보유종목수=("num_positions", "last"),
).reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

# 수익률: 전월 총자산 대비 실현손익
monthly["수익률_%"] = monthly["월실현손익"] / monthly["월말총자산"].shift(1) * 100
# 투자효율: 누적 투입 대비 총자산
monthly["투자효율_%"] = (monthly["월말총자산"] / monthly["누적투입"] - 1) * 100

print("\n── 월별 추정 잔고 ──")
pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
cols = ["month", "월말현금", "월말투자원가", "월말총자산", "월실현손익", "월입금", "누적투입", "보유종목수", "수익률_%", "투자효율_%"]
print(monthly[cols].to_string(index=False))

# ── 7. 시각화 ──────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 22))
fig.suptitle("JUN 매매법 — 추정 잔고 + 외부 입금 감지\n(2023-10 ~ 2026-02)", fontsize=16, fontweight="bold")

# (1) 추정 총자산 + 누적 투입금 비교
ax = axes[0]
ax.fill_between(balance["date"], balance["cash"] / 1e6, alpha=0.4, color="#3498db", label="현금")
ax.fill_between(balance["date"], balance["total"] / 1e6,
                balance["cash"] / 1e6, alpha=0.4, color="#e67e22", label="투자원가")
ax.plot(balance["date"], balance["total"] / 1e6, color="#2c3e50", linewidth=1.5, label="추정 총자산")
ax.plot(balance["date"], balance["cum_deposit"] / 1e6, "--", color="#e74c3c", linewidth=2, label="누적 투입금")
# 입금 이벤트 표시
if len(deposit_df) > 0:
    dep_merged = balance[balance["deposit"] > 0]
    ax.scatter(dep_merged["date"], dep_merged["total"] / 1e6,
               color="red", s=80, zorder=5, marker="^", label=f"입금 ({len(deposit_df)}건)")
ax.set_title("추정 총자산 vs 누적 투입금 (백만원)", fontsize=13)
ax.set_ylabel("백만원")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# (2) 현금 vs 투자 비율
ax = axes[1]
total_nz = balance["total"].replace(0, np.nan)
cash_pct = balance["cash"] / total_nz * 100
ax.fill_between(balance["date"], cash_pct, alpha=0.5, color="#3498db", label="현금 비중")
ax.fill_between(balance["date"], 100, cash_pct, alpha=0.5, color="#e67e22", label="투자 비중")
ax.set_title("현금 vs 투자 비중 (%)", fontsize=13)
ax.set_ylabel("%")
ax.set_ylim(0, 100)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

# (3) 투자효율: (총자산 / 누적투입 - 1) × 100
ax = axes[2]
valid_eff = monthly.dropna(subset=["투자효율_%"])
colors_e = ["#2ecc71" if v >= 0 else "#e74c3c" for v in valid_eff["투자효율_%"]]
ax.bar(valid_eff["month_dt"], valid_eff["투자효율_%"], color=colors_e, width=25, alpha=0.8)
ax.set_title("투자효율 — 누적 투입 대비 총자산 수익률 (%)", fontsize=13)
ax.set_ylabel("%")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)

# (4) 보유 종목 수 추이
ax = axes[3]
ax.step(balance["date"], balance["num_positions"], where="post", color="#8e44ad", linewidth=1.2)
ax.fill_between(balance["date"], balance["num_positions"], alpha=0.2, color="#8e44ad", step="post")
ax.set_title("보유 종목 수 추이", fontsize=13)
ax.set_ylabel("종목 수")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = CHARTS / "jun_trade_balance_estimate.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"\n차트 저장: {chart_path}")

# ── 8. CSV 저장 ──────────────────────────────────────────────
balance_out = balance[["date", "cash", "invested", "total", "deposit",
                       "cum_deposit", "realized_pnl", "num_positions"]].copy()
balance_out.to_csv(OUTPUT / "jun_trade_daily_balance.csv", index=False)
monthly.to_csv(OUTPUT / "jun_trade_monthly_balance.csv", index=False)

if len(deposit_df) > 0:
    deposit_df.to_csv(OUTPUT / "jun_trade_deposit_events.csv", index=False)

print(f"일별 잔고: {OUTPUT / 'jun_trade_daily_balance.csv'}")
print(f"월별 잔고: {OUTPUT / 'jun_trade_monthly_balance.csv'}")
if len(deposit_df) > 0:
    print(f"입금 이벤트: {OUTPUT / 'jun_trade_deposit_events.csv'}")

"""
Polymarket Crash Model — 완전한 트레이딩 시뮬레이션
=====================================================
원본(수동 튜닝) 파라미터 기준 실전 매매 시뮬레이션

기능:
  - 일별 포지션 리밸런싱 + 실제 매수/매도 이벤트 기록
  - 실현 손익 / 미실현 손익 추적
  - 거래 수수료 반영 (0.01% per trade)
  - 월별 P&L 요약
  - 벤치마크(SOXL, TQQQ) 비교
  - 종합 차트 저장

시작자본: $100,000
훈련: 2024-02-01 ~ 2025-12-31
타겟: 2026-01-01 ~ 2026-02-17
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.polymarket_crash_model import (
    load_poly_signals,
    load_market_data,
    compute_crash_score,
    assign_position,
    mode_label,
    TRAIN_START, TRAIN_END, TARGET_START, TARGET_END,
)

CHART_DIR = ROOT / "docs/charts"
RESULT_DIR = ROOT / "data/results/backtests"
CHART_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── 설정 ──────────────────────────────────────
INITIAL_CAPITAL   = 100_000.0   # 시작 자본 ($)
COMMISSION_RATE   = 0.0001      # 수수료 0.01% per side
REBALANCE_THRESH  = 0.02        # 비중 변화 2% 이상 시 리밸런싱


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    action: Literal["BUY", "SELL"]
    shares: float
    price: float
    amount: float           # 거래대금
    commission: float
    realized_pnl: float     # 매도 시 실현 손익
    crash_score: float
    mode: str
    note: str = ""


@dataclass
class Portfolio:
    """포트폴리오 상태 추적."""
    cash: float = INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)  # ticker → {shares, avg_cost}
    trades: list = field(default_factory=list)
    daily_values: list = field(default_factory=list)

    def market_value(self, prices: dict) -> float:
        val = self.cash
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, 0)
            val += pos["shares"] * price
        return val

    def position_weights(self, prices: dict) -> dict:
        total = self.market_value(prices)
        if total <= 0:
            return {}
        weights = {}
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, 0)
            weights[ticker] = pos["shares"] * price / total
        return weights

    def buy(self, date, ticker, target_amount, price, crash_score, mode):
        if price <= 0 or target_amount <= 0:
            return
        commission = target_amount * COMMISSION_RATE
        net_amount = target_amount + commission
        if net_amount > self.cash:
            target_amount = self.cash / (1 + COMMISSION_RATE) * 0.999
            commission = target_amount * COMMISSION_RATE
            net_amount = target_amount + commission

        shares = target_amount / price
        self.cash -= net_amount

        if ticker not in self.positions:
            self.positions[ticker] = {"shares": 0, "avg_cost": 0}

        pos = self.positions[ticker]
        total_cost = pos["shares"] * pos["avg_cost"] + target_amount
        pos["shares"] += shares
        pos["avg_cost"] = total_cost / pos["shares"] if pos["shares"] > 0 else 0

        self.trades.append(Trade(
            date=date, ticker=ticker, action="BUY",
            shares=shares, price=price, amount=target_amount,
            commission=commission, realized_pnl=0.0,
            crash_score=crash_score, mode=mode,
        ))

    def sell(self, date, ticker, target_amount, price, crash_score, mode):
        if ticker not in self.positions or price <= 0:
            return
        pos = self.positions[ticker]
        if pos["shares"] <= 0:
            return

        max_amount = pos["shares"] * price
        sell_amount = min(target_amount, max_amount)
        shares = sell_amount / price
        commission = sell_amount * COMMISSION_RATE
        avg_cost = pos["avg_cost"]
        realized_pnl = (price - avg_cost) * shares - commission

        self.cash += sell_amount - commission
        pos["shares"] -= shares
        if pos["shares"] < 1e-8:
            del self.positions[ticker]

        self.trades.append(Trade(
            date=date, ticker=ticker, action="SELL",
            shares=shares, price=price, amount=sell_amount,
            commission=commission, realized_pnl=realized_pnl,
            crash_score=crash_score, mode=mode,
        ))

    def sell_all(self, date, ticker, price, crash_score, mode, note=""):
        if ticker not in self.positions or price <= 0:
            return
        pos = self.positions[ticker]
        self.sell(date, ticker, pos["shares"] * price, price, crash_score, mode)
        if self.trades:
            self.trades[-1].note = note


# ─────────────────────────────────────────────
# 시뮬레이션 엔진
# ─────────────────────────────────────────────

TICKERS = ["SOXL", "TQQQ", "MSTZ"]
RET_COLS = {"SOXL": "SOXL_ret", "TQQQ": "TQQQ_ret", "MSTZ": "MSTZ_ret"}


def run_simulation(
    df: pd.DataFrame,
    start: str,
    end: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[Portfolio, pd.DataFrame]:
    """완전한 트레이딩 시뮬레이션 실행."""
    period = df.loc[start:end].copy()
    portfolio = Portfolio(cash=initial_capital)

    daily_records = []

    # 가격 컬럼 추출 (수익률 → 가격 재구성)
    price_df = {}
    for ticker in TICKERS:
        ret_col = RET_COLS[ticker]
        if ret_col in period.columns:
            rets = period[ret_col].fillna(0)
            price_df[ticker] = (1 + rets).cumprod() * 100  # 기준가 $100
        else:
            price_df[ticker] = pd.Series(100.0, index=period.index)
    prices_df = pd.DataFrame(price_df)

    for i, (dt, row) in enumerate(period.iterrows()):
        prices = {t: prices_df.loc[dt, t] for t in TICKERS}

        # 전날 신호 → 오늘 목표 포지션
        if i == 0:
            score = 0.0
        else:
            prev_score = period["crash_score"].iloc[i - 1]
            score = float(prev_score)

        target_weights = assign_position(score)
        mode = mode_label(score)
        total_value = portfolio.market_value(prices)

        # 현재 비중
        current_weights = portfolio.position_weights(prices)

        # 리밸런싱 필요 여부 판단
        all_tickers = set(target_weights.keys()) | set(current_weights.keys())
        needs_rebalance = False
        for t in all_tickers:
            target_w = target_weights.get(t, 0)
            current_w = current_weights.get(t, 0)
            if abs(target_w - current_w) >= REBALANCE_THRESH:
                needs_rebalance = True
                break

        if needs_rebalance:
            # 1단계: 줄여야 할 포지션 먼저 매도
            for t in list(portfolio.positions.keys()):
                target_w = target_weights.get(t, 0)
                current_w = current_weights.get(t, 0)
                if target_w < current_w - REBALANCE_THRESH:
                    sell_amount = (current_w - target_w) * total_value
                    portfolio.sell(dt, t, sell_amount, prices[t], score, mode)

            # 2단계: 늘려야 할 포지션 매수
            total_value = portfolio.market_value(prices)
            for t in target_weights:
                target_w = target_weights[t]
                current_w = portfolio.position_weights(prices).get(t, 0)
                if target_w > current_w + REBALANCE_THRESH:
                    buy_amount = (target_w - current_w) * total_value
                    portfolio.buy(dt, t, buy_amount, prices[t], score, mode)

        # 일별 기록
        total_value = portfolio.market_value(prices)

        unrealized_pnl = sum(
            (prices[t] - pos["avg_cost"]) * pos["shares"]
            for t, pos in portfolio.positions.items()
            if t in prices
        )
        realized_pnl_cumsum = sum(
            tr.realized_pnl for tr in portfolio.trades
        )

        daily_records.append({
            "date": dt,
            "portfolio_value": total_value,
            "cash": portfolio.cash,
            "crash_score": score,
            "mode": mode,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl_cum": realized_pnl_cumsum,
            "daily_ret": (total_value / daily_records[-1]["portfolio_value"] - 1)
            if daily_records else 0.0,
            **{f"w_{t.lower()}": portfolio.position_weights(prices).get(t, 0)
               for t in TICKERS},
        })

    daily_df = pd.DataFrame(daily_records).set_index("date")
    return portfolio, daily_df


# ─────────────────────────────────────────────
# 벤치마크 시뮬레이션
# ─────────────────────────────────────────────

def run_benchmark(df: pd.DataFrame, start: str, end: str, ticker: str) -> pd.DataFrame:
    """단순 보유 벤치마크."""
    period = df.loc[start:end].copy()
    ret_col = RET_COLS.get(ticker, f"{ticker}_ret")
    if ret_col not in period.columns:
        return pd.DataFrame()
    rets = period[ret_col].fillna(0)
    cum = (1 + rets).cumprod() * INITIAL_CAPITAL
    return pd.DataFrame({"portfolio_value": cum}, index=period.index)


# ─────────────────────────────────────────────
# 성과 지표
# ─────────────────────────────────────────────

def calc_metrics(daily_df: pd.DataFrame, label: str) -> dict:
    rets = daily_df["daily_ret"].dropna()
    if len(rets) < 5:
        return {}
    total_ret = daily_df["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - 1
    ann_ret   = (1 + total_ret) ** (252 / len(rets)) - 1
    vol       = rets.std() * np.sqrt(252)
    sharpe    = ann_ret / vol if vol > 1e-6 else 0
    cum       = daily_df["portfolio_value"] / INITIAL_CAPITAL
    mdd       = (cum / cum.cummax() - 1).min()
    win_rate  = (rets > 0).mean()
    return {
        "label": label,
        "시작자본": f"${INITIAL_CAPITAL:,.0f}",
        "최종자본": f"${daily_df['portfolio_value'].iloc[-1]:,.0f}",
        "총수익": f"{total_ret:+.1%}",
        "순손익": f"${daily_df['portfolio_value'].iloc[-1] - INITIAL_CAPITAL:+,.0f}",
        "연수익": f"{ann_ret:+.1%}",
        "변동성": f"{vol:.1%}",
        "샤프": f"{sharpe:.2f}",
        "MDD": f"{mdd:.1%}",
        "승률": f"{win_rate:.1%}",
        "거래일수": len(rets),
    }


# ─────────────────────────────────────────────
# 월별 P&L
# ─────────────────────────────────────────────

def monthly_pnl(daily_df: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_df["portfolio_value"].resample("ME").last()
    monthly_ret = monthly.pct_change()
    monthly_pnl_abs = monthly.diff()
    result = pd.DataFrame({
        "월말 자산($)": monthly.round(0),
        "월 수익률":   monthly_ret.map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-"),
        "월 손익($)":  monthly_pnl_abs.map(lambda x: f"${x:+,.0f}" if pd.notna(x) else "-"),
    })
    return result


# ─────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────

def print_report(
    portfolio: Portfolio,
    daily_df: pd.DataFrame,
    bm_soxl: pd.DataFrame,
    bm_tqqq: pd.DataFrame,
    period_label: str,
) -> None:
    print(f"\n{'='*68}")
    print(f" 트레이딩 시뮬레이션 — {period_label}")
    print(f" 시작자본: ${INITIAL_CAPITAL:,.0f}  |  리밸런싱 임계값: {REBALANCE_THRESH:.0%}")
    print(f"{'='*68}")

    # 성과 비교
    m_strat = calc_metrics(daily_df, "Crash Model (원본)")
    m_soxl  = calc_metrics(
        bm_soxl.rename(columns={"portfolio_value": "portfolio_value"}).assign(
            daily_ret=bm_soxl["portfolio_value"].pct_change().fillna(0)
        ), "SOXL 보유"
    )
    m_tqqq  = calc_metrics(
        bm_tqqq.rename(columns={"portfolio_value": "portfolio_value"}).assign(
            daily_ret=bm_tqqq["portfolio_value"].pct_change().fillna(0)
        ), "TQQQ 보유"
    )

    print(f"\n{'항목':12s} {'Crash Model':>15s} {'SOXL 보유':>13s} {'TQQQ 보유':>13s}")
    print("-" * 56)
    keys = ["최종자본", "총수익", "순손익", "연수익", "샤프", "MDD", "승률"]
    for k in keys:
        print(f"  {k:10s} {m_strat.get(k,''):>15s} {m_soxl.get(k,''):>13s} {m_tqqq.get(k,''):>13s}")

    # 모드 분포
    mode_dist = daily_df["mode"].value_counts()
    print(f"\n포지션 모드 분포:")
    for mode, cnt in mode_dist.items():
        avg_w_soxl = daily_df[daily_df["mode"] == mode]["w_soxl"].mean()
        print(f"  {mode:8s}: {cnt:3d}일  평균 SOXL 비중 {avg_w_soxl:.0%}")

    # 거래 요약
    trades = portfolio.trades
    buys  = [t for t in trades if t.action == "BUY"]
    sells = [t for t in trades if t.action == "SELL"]
    total_commission = sum(t.commission for t in trades)
    total_realized   = sum(t.realized_pnl for t in trades)

    print(f"\n거래 요약:")
    print(f"  총 거래건수:  {len(trades):4d}건  (매수 {len(buys):3d} / 매도 {len(sells):3d})")
    print(f"  총 수수료:   ${total_commission:,.0f}")
    print(f"  실현 손익:   ${total_realized:+,.0f}")
    print(f"  미실현 손익: ${daily_df['unrealized_pnl'].iloc[-1]:+,.0f}")

    # 월별 P&L
    print(f"\n월별 P&L:")
    mp = monthly_pnl(daily_df)
    print(f"  {'월':8s} {'월말자산':>14s} {'월수익률':>10s} {'월손익':>12s}")
    print("  " + "-" * 48)
    for dt, row in mp.iterrows():
        print(f"  {str(dt.date())[:7]:8s} {row['월말 자산($)']:>14,.0f} "
              f"{row['월 수익률']:>10s} {row['월 손익($)']:>12s}")

    # 주요 매매 이벤트 (상위 거래)
    print(f"\n주요 매매 이벤트 (거래금액 상위 20건):")
    print(f"  {'날짜':12s} {'종목':6s} {'매수/매도':6s} {'금액':>10s} {'손익':>10s} {'score':6s} {'모드':6s}")
    print("  " + "-" * 62)
    top_trades = sorted(trades, key=lambda t: t.amount, reverse=True)[:20]
    for t in top_trades:
        pnl_str = f"${t.realized_pnl:+,.0f}" if t.action == "SELL" else "-"
        print(
            f"  {str(t.date.date()):12s} {t.ticker:6s} {t.action:6s} "
            f"${t.amount:>9,.0f} {pnl_str:>10s} "
            f"{t.crash_score:5.2f} {t.mode:6s}"
        )

    # 최대 이익/손실 거래
    sell_trades = [t for t in trades if t.action == "SELL" and t.realized_pnl != 0]
    if sell_trades:
        best  = max(sell_trades, key=lambda t: t.realized_pnl)
        worst = min(sell_trades, key=lambda t: t.realized_pnl)
        print(f"\n  최대 이익 거래: {best.date.date()} {best.ticker} ${best.realized_pnl:+,.0f}")
        print(f"  최대 손실 거래: {worst.date.date()} {worst.ticker} ${worst.realized_pnl:+,.0f}")

    # 급락일 방어 효과
    print(f"\n급락일 방어 효과 (SOXL < -5%):")
    print(f"  {'날짜':12s} {'SOXL':8s} {'전략':8s} {'절감':8s} {'SOXL 비중':>10s}")
    print("  " + "-" * 52)
    soxl_rets = (bm_soxl["portfolio_value"].pct_change().fillna(0))
    crash_days = soxl_rets[soxl_rets < -0.05]
    for dt in crash_days.index:
        if dt in daily_df.index:
            strat_ret = daily_df.loc[dt, "daily_ret"]
            soxl_ret  = soxl_rets.loc[dt]
            saved     = strat_ret - soxl_ret
            w_soxl    = daily_df.loc[dt, "w_soxl"]
            flag      = "✓ 방어" if saved > 0 else "✗ 손실"
            print(
                f"  {str(dt.date()):12s} {soxl_ret:+7.2%}  {strat_ret:+7.2%}  "
                f"{saved:+7.2%}  {w_soxl:>9.0%}  {flag}"
            )


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

def plot_simulation(
    daily_df: pd.DataFrame,
    bm_soxl: pd.DataFrame,
    bm_tqqq: pd.DataFrame,
    portfolio: Portfolio,
    period_label: str,
    filename: str,
) -> None:
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        f"Polymarket Crash Model — 트레이딩 시뮬레이션 ({period_label})\n"
        f"시작자본: ${INITIAL_CAPITAL:,.0f}  |  수수료: {COMMISSION_RATE:.2%}  |  리밸런싱 임계값: {REBALANCE_THRESH:.0%}",
        fontsize=13, fontweight="bold",
    )

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.30,
                          height_ratios=[3, 2, 1.5, 1.5])

    # Panel 1: 포트폴리오 가치 (전체 상단)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#fafafa")

    ax1.plot(daily_df.index, daily_df["portfolio_value"],
             color="#1565c0", lw=2.2, label="Crash Model", zorder=4)
    ax1.plot(bm_soxl.index, bm_soxl["portfolio_value"],
             color="#c62828", lw=1.5, alpha=0.8, label="SOXL 보유", zorder=3)
    ax1.plot(bm_tqqq.index, bm_tqqq["portfolio_value"],
             color="#2e7d32", lw=1.5, alpha=0.8, label="TQQQ 보유", zorder=3)
    ax1.axhline(INITIAL_CAPITAL, color="#888", lw=0.8, ls="--", alpha=0.6)

    # 매수/매도 마커
    for t in portfolio.trades:
        if t.date in daily_df.index:
            val = daily_df.loc[t.date, "portfolio_value"]
            color = "#1565c0" if t.action == "BUY" else "#c62828"
            marker = "^" if t.action == "BUY" else "v"
            ax1.scatter(t.date, val, color=color, s=35, zorder=5,
                       alpha=0.7, marker=marker)

    # 수익률 레이블
    final_strat = daily_df["portfolio_value"].iloc[-1]
    final_soxl  = bm_soxl["portfolio_value"].iloc[-1]
    final_tqqq  = bm_tqqq["portfolio_value"].iloc[-1]
    for val, col, lbl in [(final_strat, "#1565c0", "Model"),
                           (final_soxl,  "#c62828", "SOXL"),
                           (final_tqqq,  "#2e7d32", "TQQQ")]:
        pct = (val / INITIAL_CAPITAL - 1) * 100
        ax1.annotate(
            f"{lbl}: ${val:,.0f} ({pct:+.1f}%)",
            xy=(daily_df.index[-1], val),
            xytext=(6, 0), textcoords="offset points",
            color=col, fontsize=8, va="center", fontweight="bold",
        )

    ax1.set_ylabel("포트폴리오 가치 ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("포트폴리오 가치 (매수▲ / 매도▼)", fontsize=11)

    # Panel 2: 포지션 비중 스택
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor("#fafafa")
    ax2.stackplot(
        daily_df.index,
        daily_df["w_soxl"] * 100,
        daily_df["w_tqqq"] * 100,
        daily_df["w_mstz"] * 100,
        labels=["SOXL", "TQQQ", "MSTZ"],
        colors=["#1565c0", "#388e3c", "#c62828"],
        alpha=0.75,
    )
    ax2.set_ylabel("포지션 비중 (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("포지션 비중 (SOXL/TQQQ/MSTZ)", fontsize=11)

    # Panel 3L: 일별 수익률 바 차트
    ax3l = fig.add_subplot(gs[2, 0])
    ax3l.set_facecolor("#fafafa")
    rets = daily_df["daily_ret"] * 100
    colors = ["#c62828" if r < 0 else "#1565c0" for r in rets]
    ax3l.bar(daily_df.index, rets, color=colors, width=1.0, alpha=0.7)
    ax3l.axhline(0, color="#333", lw=0.8)
    ax3l.set_ylabel("일 수익률 (%)", fontsize=10)
    ax3l.set_title("일별 수익률", fontsize=11)
    ax3l.grid(axis="y", alpha=0.3)

    # Panel 3R: 월별 P&L 바
    ax3r = fig.add_subplot(gs[2, 1])
    ax3r.set_facecolor("#fafafa")
    mp = daily_df["portfolio_value"].resample("ME").last().pct_change().dropna() * 100
    colors_m = ["#c62828" if r < 0 else "#1565c0" for r in mp]
    ax3r.bar(range(len(mp)), mp.values, color=colors_m, alpha=0.8)
    ax3r.set_xticks(range(len(mp)))
    ax3r.set_xticklabels([d.strftime("%y-%m") for d in mp.index],
                         rotation=45, ha="right", fontsize=7)
    ax3r.axhline(0, color="#333", lw=0.8)
    ax3r.set_ylabel("월 수익률 (%)", fontsize=10)
    ax3r.set_title("월별 수익률", fontsize=11)
    ax3r.grid(axis="y", alpha=0.3)

    # Panel 4L: Crash Score
    ax4l = fig.add_subplot(gs[3, 0])
    ax4l.set_facecolor("#fafafa")
    ax4l.fill_between(daily_df.index, daily_df["crash_score"] * 100,
                      alpha=0.25, color="#7b1fa2")
    ax4l.plot(daily_df.index, daily_df["crash_score"] * 100,
              color="#7b1fa2", lw=1.5)
    ax4l.axhline(15, color="#2196f3", lw=0.7, ls="--", alpha=0.7)
    ax4l.axhline(40, color="#ff9800", lw=0.7, ls="--", alpha=0.7)
    ax4l.set_ylabel("Crash Score (%)", fontsize=10)
    ax4l.set_title("Crash Score", fontsize=11)
    ax4l.grid(axis="y", alpha=0.3)

    # Panel 4R: 실현/미실현 손익
    ax4r = fig.add_subplot(gs[3, 1])
    ax4r.set_facecolor("#fafafa")
    ax4r.plot(daily_df.index, daily_df["realized_pnl_cum"],
              color="#2e7d32", lw=1.5, label="실현 손익")
    ax4r.plot(daily_df.index, daily_df["unrealized_pnl"],
              color="#1565c0", lw=1.5, ls="--", label="미실현 손익")
    total_pnl = daily_df["realized_pnl_cum"] + daily_df["unrealized_pnl"]
    ax4r.plot(daily_df.index, total_pnl, color="#ff9800", lw=2.0, label="총 손익")
    ax4r.axhline(0, color="#333", lw=0.8)
    ax4r.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax4r.set_ylabel("손익 ($)", fontsize=10)
    ax4r.set_title("실현/미실현 손익", fontsize=11)
    ax4r.legend(fontsize=8)
    ax4r.grid(axis="y", alpha=0.3)

    # X축 포맷
    for ax in [ax1, ax2, ax3l, ax4l]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # 범례
    buy_m  = mpatches.Patch(color="#1565c0", alpha=0.7, label="매수(▲)")
    sell_m = mpatches.Patch(color="#c62828", alpha=0.7, label="매도(▼)")
    fig.legend(handles=[buy_m, sell_m], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = CHART_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"차트 저장: {out_path}")


# ─────────────────────────────────────────────
# 결과 CSV 저장
# ─────────────────────────────────────────────

def save_csv(portfolio: Portfolio, daily_df: pd.DataFrame, prefix: str) -> None:
    # 일별 포트폴리오
    daily_df.to_csv(RESULT_DIR / f"{prefix}_daily.csv")

    # 거래 로그
    trade_rows = [{
        "date":         t.date.date(),
        "ticker":       t.ticker,
        "action":       t.action,
        "shares":       round(t.shares, 4),
        "price":        round(t.price, 4),
        "amount_usd":   round(t.amount, 2),
        "commission":   round(t.commission, 4),
        "realized_pnl": round(t.realized_pnl, 2),
        "crash_score":  round(t.crash_score, 3),
        "mode":         t.mode,
        "note":         t.note,
    } for t in portfolio.trades]

    pd.DataFrame(trade_rows).to_csv(
        RESULT_DIR / f"{prefix}_trades.csv", index=False
    )
    print(f"결과 저장: {RESULT_DIR}/{prefix}_daily.csv")
    print(f"결과 저장: {RESULT_DIR}/{prefix}_trades.csv")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    print("데이터 로드 중...")
    poly_df   = load_poly_signals()
    market_df = load_market_data()
    merged    = poly_df.join(market_df, how="left")
    merged    = compute_crash_score(merged)
    print(f"  데이터: {merged.index[0].date()} ~ {merged.index[-1].date()}")

    # ── 훈련 기간 시뮬레이션
    print(f"\n훈련 기간 시뮬레이션 ({TRAIN_START} ~ {TRAIN_END})...")
    port_train, daily_train = run_simulation(merged, TRAIN_START, TRAIN_END)
    bm_soxl_train = run_benchmark(merged, TRAIN_START, TRAIN_END, "SOXL")
    bm_tqqq_train = run_benchmark(merged, TRAIN_START, TRAIN_END, "TQQQ")
    print_report(port_train, daily_train, bm_soxl_train, bm_tqqq_train, f"훈련 기간 {TRAIN_START}~{TRAIN_END}")
    save_csv(port_train, daily_train, "sim_train")
    plot_simulation(daily_train, bm_soxl_train, bm_tqqq_train, port_train,
                    f"훈련 기간 {TRAIN_START}~{TRAIN_END}",
                    "sim_train.png")

    # ── 타겟 기간 시뮬레이션
    print(f"\n타겟 기간 시뮬레이션 ({TARGET_START} ~ {TARGET_END})...")
    port_target, daily_target = run_simulation(merged, TARGET_START, TARGET_END)
    bm_soxl_target = run_benchmark(merged, TARGET_START, TARGET_END, "SOXL")
    bm_tqqq_target = run_benchmark(merged, TARGET_START, TARGET_END, "TQQQ")
    print_report(port_target, daily_target, bm_soxl_target, bm_tqqq_target, f"타겟 기간 {TARGET_START}~{TARGET_END}")
    save_csv(port_target, daily_target, "sim_target")
    plot_simulation(daily_target, bm_soxl_target, bm_tqqq_target, port_target,
                    f"타겟 기간 {TARGET_START}~{TARGET_END}",
                    "sim_target.png")

    print(f"\n완료. 차트: docs/charts/sim_train.png / sim_target.png")


if __name__ == "__main__":
    main()

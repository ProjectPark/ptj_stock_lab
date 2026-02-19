#!/usr/bin/env python3
"""
추가 패턴 탐색 — taejun_history_2023 엔진 규칙 후보
===================================================
기존 분석에서 안 본 차원들:
  1. 시간 패턴: 월별/주별 매매 강도 변화
  2. 연속 행동 패턴: 승리 후 행동, 손절 후 행동
  3. 종목 조합 패턴: 같은 날 어떤 종목들을 같이 사는가
  4. 보유기간 추정: 매수→매도 간격
  5. 변동성 레짐별 행동: 고변동 vs 저변동
  6. 모멘텀 vs 역발상: 종목 추세 방향과 매수 관계
  7. 라운드트립 승/패 분석: 수익 거래 vs 손실 거래의 진입 조건 차이

사용법:
  pyenv shell ptj_stock_lab && python experiments/extract_rules_v2_patterns.py
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE / "data" / "results" / "analysis"


def load_data():
    daily = pd.read_csv(ANALYSIS_DIR / "decision_log.csv")
    trades = pd.read_csv(ANALYSIS_DIR / "decision_log_trades.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    trades["date"] = pd.to_datetime(trades["date"])
    daily["is_buy"] = daily["action"].isin(["BUY", "BUY+SELL"]).astype(int)
    return daily, trades


# ════════════════════════════════════════════════════════════════
# 1. 월별 행동 변화 — 레짐 시프트 감지
# ════════════════════════════════════════════════════════════════
def monthly_regime_analysis(daily: pd.DataFrame, trades: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 1] 월별 행동 변화 — 매매 강도 추이")
    print("=" * 70)

    daily["month"] = daily["date"].dt.to_period("M")
    buy_trades = trades[trades["action"] == "구매"]
    buy_trades = buy_trades.copy()
    buy_trades["month"] = buy_trades["date"].dt.to_period("M")

    print(f"\n  {'월':>8s} {'거래일':>5s} {'매수일':>5s} {'매수율':>6s} {'매수건':>5s} "
          f"{'매수금액':>10s} {'건당평균':>8s} {'종목수':>5s}")
    print("  " + "-" * 60)

    for month in sorted(daily["month"].unique()):
        m_daily = daily[daily["month"] == month]
        m_buys = buy_trades[buy_trades["month"] == month]
        buy_rate = m_daily["is_buy"].mean()
        total_amt = m_buys["amount_usd"].sum()
        avg_per = m_buys["amount_usd"].mean() if len(m_buys) > 0 else 0
        n_tickers = m_buys["yf_ticker"].nunique()
        print(f"  {str(month):>8s} {len(m_daily):>5d} {m_daily['is_buy'].sum():>5d} "
              f"{buy_rate*100:>5.0f}% {len(m_buys):>5d} "
              f"${total_amt:>9,.0f} ${avg_per:>7,.0f} {n_tickers:>5d}")

    # 전반기 vs 후반기
    mid_date = daily["date"].median()
    first_half = daily[daily["date"] < mid_date]
    second_half = daily[daily["date"] >= mid_date]
    print(f"\n  전반기 매수율: {first_half['is_buy'].mean()*100:.1f}% "
          f"({first_half['date'].min().date()} ~ {first_half['date'].max().date()})")
    print(f"  후반기 매수율: {second_half['is_buy'].mean()*100:.1f}% "
          f"({second_half['date'].min().date()} ~ {second_half['date'].max().date()})")


# ════════════════════════════════════════════════════════════════
# 2. 종목 조합 패턴 — 같은 날 함께 산 종목
# ════════════════════════════════════════════════════════════════
def ticker_combo_analysis(trades: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 2] 종목 조합 — 같은 날 함께 매수한 종목 쌍")
    print("=" * 70)

    buy_trades = trades[trades["action"] == "구매"]
    daily_tickers = buy_trades.groupby("date")["yf_ticker"].apply(
        lambda x: frozenset(x.dropna().unique())
    )

    # 동시 매수 빈도
    pair_count = defaultdict(int)
    for tickers in daily_tickers:
        tlist = sorted(tickers)
        for i in range(len(tlist)):
            for j in range(i + 1, len(tlist)):
                pair_count[(tlist[i], tlist[j])] += 1

    print(f"\n  [동시 매수 빈도 상위 10쌍]")
    print(f"  {'종목 A':<8s} {'종목 B':<8s} {'동시매수일':>8s}")
    print("  " + "-" * 28)
    for (a, b), cnt in sorted(pair_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {a:<8s} {b:<8s} {cnt:>8d}")

    # 단독 매수 vs 복수 종목 매수
    solo_days = daily_tickers[daily_tickers.apply(len) == 1]
    multi_days = daily_tickers[daily_tickers.apply(len) > 1]
    print(f"\n  단독 종목 매수일: {len(solo_days)}일")
    print(f"  복수 종목 매수일: {len(multi_days)}일")

    if len(solo_days) > 0:
        print(f"\n  [단독 매수 시 선택 종목]")
        solo_tickers = [list(s)[0] for s in solo_days]
        for ticker, cnt in pd.Series(solo_tickers).value_counts().head(5).items():
            print(f"  {ticker:<8s}: {cnt}일")


# ════════════════════════════════════════════════════════════════
# 3. 라운드트립 — 매수→매도 간격 + 승/패 분석
# ════════════════════════════════════════════════════════════════
def round_trip_analysis(trades: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 3] 라운드트립 — 보유기간 + 수익/손실 조건 비교")
    print("=" * 70)

    buys = trades[trades["action"] == "구매"].copy()
    sells = trades[trades["action"] == "판매"].copy()

    # FIFO 매칭 (간이)
    pending = defaultdict(list)  # ticker → [(date, price, qty, market_state)]
    trips = []

    for _, row in buys.sort_values("date").iterrows():
        ticker = row["yf_ticker"]
        if pd.isna(ticker):
            continue
        pending[ticker].append({
            "buy_date": row["date"],
            "buy_price": row["price_usd"],
            "qty": row["qty"],
            "buy_spy": row.get("SPY_pct", np.nan),
            "buy_gld": row.get("GLD_pct", np.nan),
            "buy_btc": row.get("poly_btc_up", np.nan),
            "buy_ticker_pct": row.get("ticker_pct", np.nan),
        })

    for _, row in sells.sort_values("date").iterrows():
        ticker = row["yf_ticker"]
        if pd.isna(ticker) or not pending[ticker]:
            continue

        sell_qty = row["qty"]
        while sell_qty > 0.001 and pending[ticker]:
            buy = pending[ticker][0]
            matched = min(sell_qty, buy["qty"])

            pnl_pct = (row["price_usd"] / buy["buy_price"] - 1) * 100 if buy["buy_price"] > 0 else 0
            hold_days = (row["date"] - buy["buy_date"]).days

            trips.append({
                "ticker": ticker,
                "buy_date": buy["buy_date"],
                "sell_date": row["date"],
                "buy_price": buy["buy_price"],
                "sell_price": row["price_usd"],
                "qty": matched,
                "pnl_pct": round(pnl_pct, 2),
                "hold_days": hold_days,
                "buy_spy": buy["buy_spy"],
                "buy_gld": buy["buy_gld"],
                "buy_btc": buy["buy_btc"],
                "buy_ticker_pct": buy["buy_ticker_pct"],
                "sell_spy": row.get("SPY_pct", np.nan),
                "sell_ticker_pct": row.get("ticker_pct", np.nan),
            })

            buy["qty"] -= matched
            sell_qty -= matched
            if buy["qty"] < 0.001:
                pending[ticker].pop(0)

    if not trips:
        print("  라운드트립 매칭 실패")
        return

    rt = pd.DataFrame(trips)
    wins = rt[rt["pnl_pct"] > 0]
    losses = rt[rt["pnl_pct"] <= 0]

    print(f"\n  총 라운드트립: {len(rt)}건 (수익 {len(wins)}, 손실 {len(losses)})")
    print(f"  승률: {len(wins)/len(rt)*100:.1f}%")
    print(f"  평균 수익률: {rt['pnl_pct'].mean():+.2f}%")

    # 보유기간 분포
    print(f"\n  [보유기간 분포]")
    for label, lo, hi in [("당일", 0, 0), ("1일", 1, 1), ("2~3일", 2, 3),
                           ("4~7일", 4, 7), ("1~2주", 8, 14), ("2주+", 15, 999)]:
        sub = rt[(rt["hold_days"] >= lo) & (rt["hold_days"] <= hi)]
        if len(sub) == 0:
            continue
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg_pnl = sub["pnl_pct"].mean()
        print(f"  {label:<8s}: {len(sub):>4d}건  승률 {wr:.0f}%  평균 {avg_pnl:+.2f}%")

    # 수익 vs 손실 거래의 진입 시점 비교
    print(f"\n  [수익 vs 손실 — 진입 시점 시장 상태]")
    print(f"  {'지표':<18s} {'수익거래':>10s} {'손실거래':>10s} {'차이':>10s}")
    print("  " + "-" * 50)
    for col in ["buy_spy", "buy_gld", "buy_btc", "buy_ticker_pct"]:
        w_mean = wins[col].dropna().mean()
        l_mean = losses[col].dropna().mean()
        if pd.isna(w_mean) or pd.isna(l_mean):
            continue
        diff = w_mean - l_mean
        print(f"  {col:<18s} {w_mean:>+10.3f} {l_mean:>+10.3f} {diff:>+10.3f}")

    # 보유기간별 최적 매도 시점
    print(f"\n  [종목별 보유기간 & 성과]")
    print(f"  {'종목':<8s} {'건수':>5s} {'승률':>6s} {'평균수익':>8s} {'평균보유':>8s} {'최적보유':>8s}")
    print("  " + "-" * 48)
    for ticker in rt["ticker"].value_counts().head(8).index:
        sub = rt[rt["ticker"] == ticker]
        wr = (sub["pnl_pct"] > 0).mean() * 100
        avg_pnl = sub["pnl_pct"].mean()
        avg_hold = sub["hold_days"].mean()
        # 최적 보유기간: 수익이 가장 높은 보유일수 구간
        if len(sub) > 5:
            best_hold = sub.loc[sub["pnl_pct"].idxmax(), "hold_days"]
        else:
            best_hold = avg_hold
        print(f"  {ticker:<8s} {len(sub):>5d} {wr:>5.0f}% {avg_pnl:>+7.2f}% "
              f"{avg_hold:>7.1f}일 {best_hold:>7.0f}일")

    # 저장
    rt.to_csv(ANALYSIS_DIR / "round_trips.csv", index=False, encoding="utf-8-sig")
    return rt


# ════════════════════════════════════════════════════════════════
# 4. 변동성 레짐별 행동
# ════════════════════════════════════════════════════════════════
def volatility_regime_analysis(daily: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 4] 변동성 레짐별 행동")
    print("=" * 70)

    vol = daily["SPY_vol_5d"].dropna()
    if len(vol) < 20:
        print("  변동성 데이터 부족")
        return

    # 3분위
    q33 = vol.quantile(0.33)
    q66 = vol.quantile(0.66)

    daily["vol_regime"] = "중"
    daily.loc[daily["SPY_vol_5d"] <= q33, "vol_regime"] = "저"
    daily.loc[daily["SPY_vol_5d"] > q66, "vol_regime"] = "고"

    print(f"\n  변동성 분위: 저 ≤ {q33:.3f}% / 중 / 고 > {q66:.3f}%")

    print(f"\n  {'레짐':<4s} {'일수':>5s} {'매수일':>5s} {'매수율':>6s} {'평균매수건':>8s} {'평균금액':>10s}")
    print("  " + "-" * 42)

    for regime in ["저", "중", "고"]:
        sub = daily[daily["vol_regime"] == regime]
        rate = sub["is_buy"].mean()
        avg_cnt = sub["buy_count"].mean()
        avg_amt = sub["buy_amount_usd"].mean()
        print(f"  {regime:<4s} {len(sub):>5d} {sub['is_buy'].sum():>5d} "
              f"{rate*100:>5.0f}% {avg_cnt:>8.1f} ${avg_amt:>9,.0f}")


# ════════════════════════════════════════════════════════════════
# 5. 모멘텀 vs 역발상 패턴
# ════════════════════════════════════════════════════════════════
def momentum_vs_contrarian(trades: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 5] 모멘텀 vs 역발상 — 매수 종목의 최근 추세")
    print("=" * 70)

    buy_trades = trades[trades["action"] == "구매"].copy()
    ticker_pct = buy_trades["ticker_pct"].dropna()

    if len(ticker_pct) < 10:
        print("  데이터 부족")
        return

    # 매수 시점 종목 등락 구간별
    bins = [(-999, -5), (-5, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 5), (5, 999)]

    print(f"\n  [매수 시점 종목 당일 등락별 — 건수 & 금액]")
    print(f"  {'구간':<14s} {'건수':>5s} {'비율':>6s} {'평균금액':>10s} {'총금액':>12s}")
    print("  " + "-" * 50)

    for lo, hi in bins:
        mask = (buy_trades["ticker_pct"] >= lo) & (buy_trades["ticker_pct"] < hi) & buy_trades["ticker_pct"].notna()
        sub = buy_trades[mask]
        if len(sub) < 2:
            continue
        pct = len(sub) / len(buy_trades) * 100
        avg_amt = sub["amount_usd"].mean()
        total = sub["amount_usd"].sum()
        label = f"[{lo:+.0f}, {hi:+.0f}%)" if abs(lo) < 100 else (f"< {hi:+.0f}%" if lo < -100 else f">= {lo:+.0f}%")
        print(f"  {label:<14s} {len(sub):>5d} {pct:>5.1f}% ${avg_amt:>9,.0f} ${total:>11,.0f}")

    # 종목별 모멘텀/역발상 성향
    print(f"\n  [종목별 — 하락 중 매수 비율]")
    print(f"  {'종목':<8s} {'매수건':>5s} {'하락매수':>7s} {'하락율':>6s} {'하락시평균금액':>12s} {'상승시평균금액':>12s}")
    print("  " + "-" * 55)

    for ticker in buy_trades["yf_ticker"].value_counts().head(8).index:
        sub = buy_trades[buy_trades["yf_ticker"] == ticker]
        valid = sub["ticker_pct"].dropna()
        if len(valid) < 5:
            continue
        down = sub[sub["ticker_pct"] < 0]
        up = sub[sub["ticker_pct"] >= 0]
        down_rate = len(down) / len(valid) * 100
        down_amt = down["amount_usd"].mean() if len(down) > 0 else 0
        up_amt = up["amount_usd"].mean() if len(up) > 0 else 0
        print(f"  {ticker:<8s} {len(valid):>5d} {len(down):>7d} {down_rate:>5.0f}% "
              f"${down_amt:>11,.0f} ${up_amt:>11,.0f}")


# ════════════════════════════════════════════════════════════════
# 6. 연속 행동 패턴 — 승리/패배 후 반응
# ════════════════════════════════════════════════════════════════
def sequential_behavior(daily: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 6] 연속 행동 — 전일 행동이 다음날에 미치는 영향")
    print("=" * 70)

    daily_sorted = daily.sort_values("date").reset_index(drop=True)

    daily_sorted["prev_action"] = daily_sorted["action"].shift(1)
    daily_sorted["prev_buy_amount"] = daily_sorted["buy_amount_usd"].shift(1)
    daily_sorted["prev_spy"] = daily_sorted["SPY_pct"].shift(1)

    valid = daily_sorted.dropna(subset=["prev_action"])

    # 전일 행동별 → 다음날 매수 확률
    print(f"\n  [전일 행동 → 다음날 매수 확률]")
    print(f"  {'전일 행동':<12s} {'일수':>5s} {'다음날 매수':>8s} {'매수율':>7s}")
    print("  " + "-" * 35)
    for prev in ["HOLD", "BUY", "SELL", "BUY+SELL"]:
        sub = valid[valid["prev_action"] == prev]
        if len(sub) < 5:
            continue
        rate = sub["is_buy"].mean()
        print(f"  {prev:<12s} {len(sub):>5d} {sub['is_buy'].sum():>8d} {rate*100:>6.1f}%")

    # 전일 SPY 방향별 → 다음날 매수 확률
    print(f"\n  [전일 SPY 방향 → 다음날 매수 확률]")
    valid["prev_spy_dir"] = "보합"
    valid.loc[valid["prev_spy"] > 0.5, "prev_spy_dir"] = "상승(>0.5%)"
    valid.loc[valid["prev_spy"] < -0.5, "prev_spy_dir"] = "하락(<-0.5%)"

    for direction in ["하락(<-0.5%)", "보합", "상승(>0.5%)"]:
        sub = valid[valid["prev_spy_dir"] == direction]
        if len(sub) < 5:
            continue
        rate = sub["is_buy"].mean()
        print(f"  {direction:<16s}: {len(sub):>3d}일 → 매수율 {rate*100:.1f}%")

    # 연속 매수일 후 관망 확률
    daily_sorted["buy_streak"] = 0
    streak = 0
    for i in range(len(daily_sorted)):
        if daily_sorted.loc[i, "is_buy"] == 1:
            streak += 1
        else:
            streak = 0
        daily_sorted.loc[i, "buy_streak"] = streak

    daily_sorted["prev_streak"] = daily_sorted["buy_streak"].shift(1)
    print(f"\n  [연속 매수일수 후 다음날 매수 확률]")
    for streak_len in [0, 1, 2, 3, 4, 5]:
        sub = daily_sorted[daily_sorted["prev_streak"] == streak_len]
        if len(sub) < 3:
            continue
        rate = sub["is_buy"].mean()
        label = f"{streak_len}일 연속 매수 후" if streak_len > 0 else "관망일 다음"
        print(f"  {label:<18s}: {len(sub):>3d}일 → 매수율 {rate*100:.1f}%")


# ════════════════════════════════════════════════════════════════
# 7. SPY-GLD 조합 패턴 (2차원 분석)
# ════════════════════════════════════════════════════════════════
def spy_gld_combined(daily: pd.DataFrame):
    print()
    print("=" * 70)
    print("  [패턴 7] SPY × GLD 2차원 매수 확률 히트맵")
    print("=" * 70)

    spy_bins = [(-999, -0.5, "SPY↓↓"), (-0.5, 0, "SPY↓"), (0, 0.5, "SPY↑"), (0.5, 999, "SPY↑↑")]
    gld_bins = [(-999, -0.5, "GLD↓↓"), (-0.5, 0, "GLD↓"), (0, 0.5, "GLD↑"), (0.5, 999, "GLD↑↑")]

    print(f"\n  {'':>10s}", end="")
    for _, _, gl in gld_bins:
        print(f" {gl:>8s}", end="")
    print()
    print("  " + "-" * 44)

    for spy_lo, spy_hi, spy_label in spy_bins:
        print(f"  {spy_label:<10s}", end="")
        for gld_lo, gld_hi, _ in gld_bins:
            mask = (
                (daily["SPY_pct"] >= spy_lo) & (daily["SPY_pct"] < spy_hi)
                & (daily["GLD_pct"] >= gld_lo) & (daily["GLD_pct"] < gld_hi)
                & daily["SPY_pct"].notna() & daily["GLD_pct"].notna()
            )
            sub = daily[mask]
            if len(sub) < 3:
                print(f" {'---':>8s}", end="")
            else:
                rate = sub["is_buy"].mean() * 100
                print(f" {rate:>6.0f}%({len(sub):>1d})", end="" if len(sub) < 10 else "")
                if len(sub) >= 10:
                    print(f"", end="")
                else:
                    print(f"", end="")
            print("", end="")
        print()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  추가 패턴 탐색 — taejun_history_2023 규칙 후보     ║")
    print("  ╚══════════════════════════════════════════════════════╝")

    daily, trades = load_data()
    print(f"  일별 로그: {len(daily)}일, 거래 로그: {len(trades)}건")

    monthly_regime_analysis(daily, trades)
    ticker_combo_analysis(trades)
    rt = round_trip_analysis(trades)
    volatility_regime_analysis(daily)
    momentum_vs_contrarian(trades)
    sequential_behavior(daily)
    spy_gld_combined(daily)

    print("\n  완료!")
    print()


if __name__ == "__main__":
    main()

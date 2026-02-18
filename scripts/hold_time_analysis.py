"""
보유시간별 수익률 분석
- 분석1: 09:30 가상매수 후 시간별 수익률 분포
- 분석2: 최적 보유시간 분석 (장중 최고/최저 도달 시점)
- 분석3: 실제 거래와 5분봉 매칭 (매수 시점 추정 후 추적)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
HIST_DIR = BASE / "history"
OUT_DIR = BASE / "stock_history"

# ── Load data ────────────────────────────────────────────────────────────────
df5 = pd.read_parquet(DATA_DIR / "backtest_5min.parquet")
df5["date"] = pd.to_datetime(df5["date"]).dt.date
df5 = df5.sort_values(["symbol", "date", "timestamp"]).reset_index(drop=True)

trades = pd.read_csv(HIST_DIR / "거래내역서_20250218_20260217_1.csv", encoding="utf-8-sig")
with open(OUT_DIR / "ticker_mapping.json", encoding="utf-8") as f:
    ticker_map = json.load(f)

name_to_yf = {name: info["yf"] for name, info in ticker_map.items()}

# ── Hold period definitions ──────────────────────────────────────────────────
HOLD_MINUTES = {"30min": 30, "1h": 60, "2h": 120, "3h": 180, "5h": 300, "close": None}

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 1: 가상 매수(09:30) 후 시간별 수익률 분포
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("분석 1: 09:30 가상매수 후 시간별 수익률 분포")
print("=" * 70)

all_returns = []

for (symbol, date), day_df in df5.groupby(["symbol", "date"]):
    day_df = day_df.sort_values("timestamp").reset_index(drop=True)
    if len(day_df) < 2:
        continue

    open_price = day_df.iloc[0]["open"]
    if open_price <= 0 or np.isnan(open_price):
        continue

    open_ts = day_df.iloc[0]["timestamp"]
    row = {"symbol": symbol, "date": date, "open_price": open_price}

    for label, mins in HOLD_MINUTES.items():
        if mins is None:
            price = day_df.iloc[-1]["close"]
        else:
            target_ts = open_ts + pd.Timedelta(minutes=mins)
            mask = day_df["timestamp"] <= target_ts
            if mask.any():
                price = day_df.loc[mask].iloc[-1]["close"]
            else:
                price = np.nan
        row[f"pct_{label}"] = (price / open_price - 1) * 100 if not np.isnan(price) else np.nan

    # 장중 최고/최저 수익률 + 도달 시점(분)
    max_returns = (day_df["high"] / open_price - 1) * 100
    min_returns = (day_df["low"] / open_price - 1) * 100
    row["intraday_max_pct"] = max_returns.max()
    row["intraday_min_pct"] = min_returns.min()
    row["max_reach_min"] = int(max_returns.idxmax() - day_df.index[0]) * 5
    row["min_reach_min"] = int(min_returns.idxmin() - day_df.index[0]) * 5

    all_returns.append(row)

returns_df = pd.DataFrame(all_returns)
print(f"총 {len(returns_df)} 건 (종목 x 거래일)\n")

# ── 종목별 통계 ──
period_cols = [f"pct_{k}" for k in HOLD_MINUTES.keys()]
stats_rows = []

for symbol in sorted(returns_df["symbol"].unique()):
    sdf = returns_df[returns_df["symbol"] == symbol]
    row = {"symbol": symbol, "trading_days": len(sdf)}

    for col in period_cols:
        vals = sdf[col].dropna()
        period = col.replace("pct_", "")
        row[f"{period}_mean"] = vals.mean()
        row[f"{period}_median"] = vals.median()
        row[f"{period}_std"] = vals.std()
        row[f"{period}_max"] = vals.max()
        row[f"{period}_min"] = vals.min()
        row[f"{period}_win_rate"] = (vals > 0).mean() * 100

    row["avg_max_reach_min"] = sdf["max_reach_min"].mean()
    row["avg_min_reach_min"] = sdf["min_reach_min"].mean()
    row["intraday_max_pct_avg"] = sdf["intraday_max_pct"].mean()
    row["intraday_min_pct_avg"] = sdf["intraday_min_pct"].mean()
    row["rule_5h_vs_close"] = sdf["pct_5h"].dropna().mean() - sdf["pct_close"].dropna().mean()

    stats_rows.append(row)

# 전체 통합(ALL)
all_row = {"symbol": "ALL", "trading_days": len(returns_df)}
for col in period_cols:
    vals = returns_df[col].dropna()
    period = col.replace("pct_", "")
    all_row[f"{period}_mean"] = vals.mean()
    all_row[f"{period}_median"] = vals.median()
    all_row[f"{period}_std"] = vals.std()
    all_row[f"{period}_max"] = vals.max()
    all_row[f"{period}_min"] = vals.min()
    all_row[f"{period}_win_rate"] = (vals > 0).mean() * 100
all_row["avg_max_reach_min"] = returns_df["max_reach_min"].mean()
all_row["avg_min_reach_min"] = returns_df["min_reach_min"].mean()
all_row["intraday_max_pct_avg"] = returns_df["intraday_max_pct"].mean()
all_row["intraday_min_pct_avg"] = returns_df["intraday_min_pct"].mean()
all_row["rule_5h_vs_close"] = returns_df["pct_5h"].dropna().mean() - returns_df["pct_close"].dropna().mean()
stats_rows.append(all_row)

stats_df = pd.DataFrame(stats_rows)

# ── 콘솔 출력 ──
print("[종목별 x 보유시간별 평균 수익률 (%)]")
print("-" * 110)
header = f"  {'종목':<8s} {'거래일':>5s} {'30min':>8s} {'1h':>8s} {'2h':>8s} {'3h':>8s} {'5h':>8s} {'장마감':>8s}"
print(header)
for _, r in stats_df.iterrows():
    print(f"  {r['symbol']:<8s} {int(r['trading_days']):>5d} "
          f"{r['30min_mean']:>+8.2f} {r['1h_mean']:>+8.2f} {r['2h_mean']:>+8.2f} "
          f"{r['3h_mean']:>+8.2f} {r['5h_mean']:>+8.2f} {r['close_mean']:>+8.2f}")
print()

print("[종목별 양전확률 (%)]")
print("-" * 110)
header = f"  {'종목':<8s} {'30min':>8s} {'1h':>8s} {'2h':>8s} {'3h':>8s} {'5h':>8s} {'장마감':>8s}"
print(header)
for _, r in stats_df.iterrows():
    print(f"  {r['symbol']:<8s} "
          f"{r['30min_win_rate']:>8.1f} {r['1h_win_rate']:>8.1f} {r['2h_win_rate']:>8.1f} "
          f"{r['3h_win_rate']:>8.1f} {r['5h_win_rate']:>8.1f} {r['close_win_rate']:>8.1f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 2: 최적 보유시간 분석
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("분석 2: 최적 보유시간 (장중 최고 수익률 도달 시점)")
print("=" * 70)

print(f"\n  {'종목':<8s} {'최고점(분)':>10s} {'최저점(분)':>10s} {'최고수익%':>10s} {'5h매도%':>9s} {'장마감%':>9s} {'5h-장마감':>9s}")
print("  " + "-" * 70)
for _, r in stats_df.iterrows():
    print(f"  {r['symbol']:<8s} "
          f"{r['avg_max_reach_min']:>8.0f}분 {r['avg_min_reach_min']:>8.0f}분 "
          f"{r['intraday_max_pct_avg']:>+9.2f} {r['5h_mean']:>+8.2f} {r['close_mean']:>+8.2f} {r['rule_5h_vs_close']:>+8.2f}")

overall_max = returns_df["max_reach_min"].mean()
overall_min = returns_df["min_reach_min"].mean()
print(f"\n  전체 평균 장중 최고 도달: {overall_max:.0f}분 ({overall_max/60:.1f}시간)")
print(f"  전체 평균 장중 최저 도달: {overall_min:.0f}분 ({overall_min/60:.1f}시간)")
print(f"  5h vs 장마감: {'5시간 매도 유리' if all_row['rule_5h_vs_close'] > 0 else '장마감 매도 유리'} "
      f"(차이 {all_row['rule_5h_vs_close']:+.3f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 3: 실제 거래와 5분봉 매칭
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("분석 3: 실제 거래 매칭 (매수가 → 5분봉 시점 추정 → 추적)")
print("=" * 70)

def map_name_to_ticker(name):
    for map_name, yf_ticker in name_to_yf.items():
        if name.startswith(map_name) or map_name.startswith(name):
            return yf_ticker
    return None

buys = trades[trades["거래구분"] == "구매"].copy()
buys["yf_ticker"] = buys["종목명"].apply(map_name_to_ticker)
buys["trade_date"] = pd.to_datetime(buys["거래일자"], format="%Y.%m.%d").dt.date
buys["단가_달러"] = pd.to_numeric(buys["단가_달러"], errors="coerce")
buys["거래수량"] = pd.to_numeric(buys["거래수량"].astype(str).str.replace(",", ""), errors="coerce")

available_symbols = set(df5["symbol"].unique())
tracking_rows = []
matched = 0
unmatched = set()

for _, buy in buys.iterrows():
    ticker = buy["yf_ticker"]
    tdate = buy["trade_date"]
    buy_price = buy["단가_달러"]

    if ticker is None or pd.isna(buy_price) or buy_price <= 0:
        unmatched.add(buy["종목명"])
        continue
    if ticker not in available_symbols:
        unmatched.add(f"{buy['종목명']} -> {ticker}")
        continue

    day_df = df5[(df5["symbol"] == ticker) & (df5["date"] == tdate)].sort_values("timestamp").reset_index(drop=True)
    if len(day_df) == 0:
        continue

    matched += 1

    # 매수 시점 추정: close가 매수가에 가장 가까운 5분봉
    price_diff = (day_df["close"] - buy_price).abs()
    buy_bar_idx = price_diff.idxmin()
    buy_bar = day_df.loc[buy_bar_idx]
    buy_ts = buy_bar["timestamp"]

    # 매수 시점 이후 바
    after_buy = day_df[day_df["timestamp"] >= buy_ts].sort_values("timestamp")
    if after_buy.empty:
        continue

    row = {
        "trade_date": tdate,
        "ticker": ticker,
        "종목명": buy["종목명"],
        "qty": buy["거래수량"],
        "buy_price_csv": buy_price,
        "buy_price_est": buy_bar["close"],
        "buy_time_est": str(buy_ts),
    }

    close_bar = after_buy.iloc[-1]
    for label, mins in HOLD_MINUTES.items():
        if mins is None:
            price = close_bar["close"]
        else:
            target_ts = buy_ts + pd.Timedelta(minutes=mins)
            mask = after_buy["timestamp"] <= target_ts
            if mask.any():
                price = after_buy.loc[mask].iloc[-1]["close"]
            else:
                price = np.nan

        row[f"price_{label}"] = price
        row[f"ret_{label}"] = (price / buy_price - 1) * 100 if (not np.isnan(price) and buy_price > 0) else np.nan

    # 장중 최대/최소 (매수 이후)
    if len(after_buy) > 0:
        row["max_after_buy"] = after_buy["high"].max()
        row["min_after_buy"] = after_buy["low"].min()
        row["max_possible_ret"] = (after_buy["high"].max() / buy_price - 1) * 100
        row["max_possible_loss"] = (after_buy["low"].min() / buy_price - 1) * 100

    tracking_rows.append(row)

tracking_df = pd.DataFrame(tracking_rows)

print(f"\n  총 매수 건: {len(buys)}, 매핑+매칭: {matched}")
if unmatched:
    print(f"  매칭 실패: {unmatched}")

if len(tracking_df) > 0:
    print("\n[실제 매수 후 시간별 수익률 통계]")
    print(f"  {'기간':>8s} {'평균':>9s} {'중앙값':>9s} {'표준편차':>9s} {'양전확률':>8s} {'건수':>5s}")
    print("  " + "-" * 55)
    for label in HOLD_MINUTES.keys():
        col = f"ret_{label}"
        vals = tracking_df[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {label:>8s} {vals.mean():>+8.2f}% {vals.median():>+8.2f}% "
              f"{vals.std():>8.2f}% {(vals > 0).mean()*100:>7.1f}% {len(vals):>5d}")

    print("\n[종목별 실제 매수 후 장마감 수익률]")
    for ticker, grp in tracking_df.groupby("ticker"):
        vals = grp["ret_close"].dropna()
        if len(vals) == 0:
            continue
        print(f"  {ticker:<8s}: 평균 {vals.mean():+.2f}%  중앙값 {vals.median():+.2f}%  "
              f"양전 {(vals > 0).mean()*100:.1f}%  ({len(vals)}건)")

    tracking_df.to_csv(OUT_DIR / "backtest_actual_trade_tracking.csv", index=False, encoding="utf-8-sig")
    print(f"\n-> 저장: backtest_actual_trade_tracking.csv")
else:
    print("  매칭된 거래 없음")
    pd.DataFrame().to_csv(OUT_DIR / "backtest_actual_trade_tracking.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 저장
# ═══════════════════════════════════════════════════════════════════════════════
stats_df.to_csv(OUT_DIR / "backtest_hold_time_analysis.csv", index=False, encoding="utf-8-sig")
print(f"-> 저장: backtest_hold_time_analysis.csv")
print("\n분석 완료!")

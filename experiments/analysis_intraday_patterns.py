"""
Intraday Pattern Analysis around Trade Entries
===============================================
Analyzes 5-minute chart patterns around actual trade entries to understand:
1. Entry time distribution (winning vs losing)
2. Pre-entry price action (dip-buy vs momentum-buy)
3. Post-entry price action (MFE/MAE)
4. Intraday volatility at entry
5. Volume profile at entry
6. Open-to-entry analysis (gap, opening range breakout)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
OHLCV_5MIN = BASE / "data" / "market" / "ohlcv" / "backtest_5min.parquet"
TRADES_CSV = BASE / "data" / "results" / "analysis" / "decision_log_trades.csv"
ROUND_TRIPS = BASE / "data" / "results" / "analysis" / "round_trips.csv"
OUTPUT_CSV = BASE / "data" / "results" / "analysis" / "intraday_patterns_report.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("INTRADAY PATTERN ANALYSIS AROUND TRADE ENTRIES")
print("=" * 80)

ohlcv = pd.read_parquet(OHLCV_5MIN)
trades = pd.read_csv(TRADES_CSV)
round_trips = pd.read_csv(ROUND_TRIPS)

print(f"\n5min OHLCV: {len(ohlcv):,} rows, {ohlcv['symbol'].nunique()} symbols")
print(f"Trade log:  {len(trades):,} rows")
print(f"Round trips: {len(round_trips):,} trips")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Filter buys only
buys = trades[trades["action"] == "구매"].copy()
buys["date"] = pd.to_datetime(buys["date"]).dt.strftime("%Y-%m-%d")
print(f"\nBuy trades: {len(buys)}")

# Ensure ohlcv date is string for matching
ohlcv["date_str"] = pd.to_datetime(ohlcv["date"]).dt.strftime("%Y-%m-%d")

# Find overlapping tickers
ohlcv_symbols = set(ohlcv["symbol"].unique())
trade_tickers = set(buys["yf_ticker"].dropna().unique())
overlap = sorted(ohlcv_symbols & trade_tickers)
print(f"Overlapping tickers ({len(overlap)}): {overlap}")

# Filter buys to only those with 5min data
buys = buys[buys["yf_ticker"].isin(overlap)].copy()
print(f"Buy trades with 5min data: {len(buys)}")

# Round trips: join pnl_pct so we know if a buy was a winner
# Build a mapping: (ticker, buy_date) -> pnl_pct (take the amount-weighted average if multiple)
rt = round_trips.copy()
rt["buy_date"] = pd.to_datetime(rt["buy_date"]).dt.strftime("%Y-%m-%d")
rt_pnl = rt.groupby(["ticker", "buy_date"]).agg(
    pnl_pct=("pnl_pct", "mean"),
    hold_days=("hold_days", "mean"),
).reset_index()

buys = buys.merge(
    rt_pnl, left_on=["yf_ticker", "date"], right_on=["ticker", "buy_date"], how="left"
)
buys["is_winner"] = buys["pnl_pct"] > 0
buys["outcome"] = buys["is_winner"].map({True: "Winner", False: "Loser"})
# Drop rows without pnl info (no round trip matched)
matched = buys.dropna(subset=["pnl_pct"]).copy()
print(f"Buy trades matched to round trips: {len(matched)}")
print(f"  Winners: {matched['is_winner'].sum()}, Losers: {(~matched['is_winner']).sum()}")

# ---------------------------------------------------------------------------
# Helper: get 5min bars for a (symbol, date)
# ---------------------------------------------------------------------------
# Build a dict of DataFrames keyed by (symbol, date_str) for fast lookup
ohlcv_sorted = ohlcv.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
ohlcv_grouped = {
    (sym, dt): grp.reset_index(drop=True)
    for (sym, dt), grp in ohlcv_sorted.groupby(["symbol", "date_str"])
}


def get_day_bars(symbol: str, date_str: str) -> pd.DataFrame | None:
    """Return 5min bars for a symbol on a given date, or None."""
    return ohlcv_grouped.get((symbol, date_str))


# ---------------------------------------------------------------------------
# Step 1: Match each buy to its most likely 5min bar (closest price on day)
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("STEP 1: Matching buy trades to 5min bars by closest price...")
print("-" * 80)

entry_records = []
for idx, row in matched.iterrows():
    sym = row["yf_ticker"]
    dt = row["date"]
    price = row["price_usd"]

    bars = get_day_bars(sym, dt)
    if bars is None or bars.empty:
        continue

    # Find bar with close closest to trade price
    price_diff = (bars["close"] - price).abs()
    best_idx = price_diff.idxmin()
    bar = bars.loc[best_idx]

    entry_records.append({
        "trade_idx": idx,
        "symbol": sym,
        "date": dt,
        "price_usd": price,
        "bar_idx_in_day": best_idx,
        "entry_time": bar["timestamp"],
        "bar_close": bar["close"],
        "bar_open": bar["open"],
        "bar_high": bar["high"],
        "bar_low": bar["low"],
        "bar_volume": bar["volume"],
        "bar_vwap": bar["vwap"],
        "price_diff": abs(bar["close"] - price),
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
        "hold_days": row["hold_days"],
        "amount_usd": row["amount_usd"],
    })

entries = pd.DataFrame(entry_records)
print(f"Matched entries: {len(entries)}")

# Extract time components
entries["entry_hour"] = entries["entry_time"].dt.hour
entries["entry_minute"] = entries["entry_time"].dt.minute
entries["entry_hhmm"] = entries["entry_time"].dt.strftime("%H:%M")

# ---------------------------------------------------------------------------
# ANALYSIS 1: Entry Time Distribution
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 1: ENTRY TIME DISTRIBUTION")
print("=" * 80)

# Create 30-minute time slots
def time_to_slot(ts):
    h, m = ts.hour, ts.minute
    slot_m = (m // 30) * 30
    return f"{h:02d}:{slot_m:02d}"

entries["time_slot"] = entries["entry_time"].apply(time_to_slot)

# Overall distribution
time_dist = entries["time_slot"].value_counts().sort_index()
print("\nEntry time distribution (30-min slots):")
print(f"{'Slot':<10} {'Count':>6} {'Pct':>7}")
print("-" * 25)
for slot, count in time_dist.items():
    pct = count / len(entries) * 100
    print(f"{slot:<10} {count:>6} {pct:>6.1f}%")

# Winners vs Losers
print("\nWinners vs Losers by time slot:")
print(f"{'Slot':<10} {'Win':>6} {'Lose':>6} {'Win%':>7} {'AvgPnL_W':>10} {'AvgPnL_L':>10}")
print("-" * 55)
for slot in sorted(entries["time_slot"].unique()):
    subset = entries[entries["time_slot"] == slot]
    w = subset[subset["is_winner"]]
    l = subset[~subset["is_winner"]]
    win_pct = len(w) / len(subset) * 100 if len(subset) > 0 else 0
    avg_w = w["pnl_pct"].mean() if len(w) > 0 else 0
    avg_l = l["pnl_pct"].mean() if len(l) > 0 else 0
    print(f"{slot:<10} {len(w):>6} {len(l):>6} {win_pct:>6.1f}% {avg_w:>9.2f}% {avg_l:>9.2f}%")

# Peak entry time
peak_slot = time_dist.idxmax()
print(f"\nPeak entry slot: {peak_slot} ({time_dist.max()} trades, "
      f"{time_dist.max()/len(entries)*100:.1f}%)")

# Best performing entry window
slot_pnl = entries.groupby("time_slot")["pnl_pct"].mean().sort_values(ascending=False)
print(f"Best avg PnL slot: {slot_pnl.index[0]} (avg {slot_pnl.iloc[0]:+.2f}%)")
print(f"Worst avg PnL slot: {slot_pnl.index[-1]} (avg {slot_pnl.iloc[-1]:+.2f}%)")

# ---------------------------------------------------------------------------
# ANALYSIS 2: Pre-entry Price Action (1 hour = 12 bars before)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 2: PRE-ENTRY PRICE ACTION (1 HOUR BEFORE)")
print("=" * 80)

pre_records = []
for _, row in entries.iterrows():
    sym = row["symbol"]
    dt = row["date"]
    bars = get_day_bars(sym, dt)
    if bars is None:
        continue

    # Find the bar index in the day's bars that matches entry_time
    match_mask = bars["timestamp"] == row["entry_time"]
    if not match_mask.any():
        continue
    bar_pos = match_mask.idxmax()

    # Get up to 12 bars before entry
    start_pos = max(0, bar_pos - 12)
    pre_bars = bars.loc[start_pos:bar_pos - 1] if bar_pos > 0 else pd.DataFrame()

    if len(pre_bars) < 3:
        continue

    pre_open = pre_bars.iloc[0]["open"]
    pre_close = pre_bars.iloc[-1]["close"]
    pre_change_pct = (pre_close - pre_open) / pre_open * 100

    # Was it a dip-buy (price declining) or momentum-buy (price rising)?
    is_dip_buy = pre_change_pct < 0

    # Pre-entry volatility (range of the pre-bars)
    pre_high = pre_bars["high"].max()
    pre_low = pre_bars["low"].min()
    pre_range_pct = (pre_high - pre_low) / pre_low * 100

    pre_records.append({
        "symbol": sym,
        "date": dt,
        "entry_time": row["entry_time"],
        "pre_change_pct": pre_change_pct,
        "is_dip_buy": is_dip_buy,
        "pre_range_pct": pre_range_pct,
        "pre_bars_count": len(pre_bars),
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
    })

pre_df = pd.DataFrame(pre_records)
print(f"\nEntries with pre-entry data: {len(pre_df)}")

# Dip-buy vs Momentum-buy
dip_buys = pre_df[pre_df["is_dip_buy"]]
mom_buys = pre_df[~pre_df["is_dip_buy"]]
print(f"\nDip buys (price declining before entry): {len(dip_buys)} ({len(dip_buys)/len(pre_df)*100:.1f}%)")
print(f"Momentum buys (price rising before entry): {len(mom_buys)} ({len(mom_buys)/len(pre_df)*100:.1f}%)")

print(f"\nDip-buy outcomes:")
print(f"  Avg PnL: {dip_buys['pnl_pct'].mean():+.2f}%")
print(f"  Win rate: {dip_buys['is_winner'].mean()*100:.1f}%")
print(f"  Median pre-change: {dip_buys['pre_change_pct'].median():.2f}%")

print(f"\nMomentum-buy outcomes:")
print(f"  Avg PnL: {mom_buys['pnl_pct'].mean():+.2f}%")
print(f"  Win rate: {mom_buys['is_winner'].mean()*100:.1f}%")
print(f"  Median pre-change: {mom_buys['pre_change_pct'].median():+.2f}%")

# Winners vs Losers: dip vs momentum proportions
print("\nWinners vs Losers: Entry Style Breakdown:")
for outcome in ["Winner", "Loser"]:
    subset = pre_df[pre_df["outcome"] == outcome]
    dip_pct = subset["is_dip_buy"].mean() * 100
    print(f"  {outcome}s: {dip_pct:.1f}% dip-buy, {100-dip_pct:.1f}% momentum-buy")

# Quartile analysis of pre-change
pre_df["pre_change_q"] = pd.qcut(pre_df["pre_change_pct"], q=4, labels=["Q1(big dip)", "Q2", "Q3", "Q4(big rise)"])
print("\nPnL by pre-entry price change quartile:")
print(f"{'Quartile':<18} {'Count':>6} {'AvgPnL':>9} {'WinRate':>9} {'AvgPreChg':>10}")
print("-" * 55)
for q in ["Q1(big dip)", "Q2", "Q3", "Q4(big rise)"]:
    subset = pre_df[pre_df["pre_change_q"] == q]
    print(f"{q:<18} {len(subset):>6} {subset['pnl_pct'].mean():>8.2f}% "
          f"{subset['is_winner'].mean()*100:>7.1f}% {subset['pre_change_pct'].mean():>9.2f}%")

# ---------------------------------------------------------------------------
# ANALYSIS 3: Post-entry Price Action (1 hour = 12 bars after)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 3: POST-ENTRY PRICE ACTION (1 HOUR AFTER)")
print("=" * 80)

post_records = []
for _, row in entries.iterrows():
    sym = row["symbol"]
    dt = row["date"]
    bars = get_day_bars(sym, dt)
    if bars is None:
        continue

    match_mask = bars["timestamp"] == row["entry_time"]
    if not match_mask.any():
        continue
    bar_pos = match_mask.idxmax()

    # Get up to 12 bars after entry
    end_pos = min(len(bars) - 1, bar_pos + 12)
    post_bars = bars.loc[bar_pos + 1:end_pos] if bar_pos < len(bars) - 1 else pd.DataFrame()

    if len(post_bars) < 3:
        continue

    entry_price = row["bar_close"]

    # Post-entry metrics
    post_highs = post_bars["high"].values
    post_lows = post_bars["low"].values
    post_closes = post_bars["close"].values

    # MFE: max favorable excursion (best price vs entry)
    mfe_pct = (post_highs.max() - entry_price) / entry_price * 100

    # MAE: max adverse excursion (worst price vs entry)
    mae_pct = (entry_price - post_lows.min()) / entry_price * 100

    # Net change in 1hr
    post_change_pct = (post_closes[-1] - entry_price) / entry_price * 100

    # Did price continue up (good) or reverse down (bad)?
    is_continuation = post_change_pct > 0

    # How quickly did it reach max adverse (bar count to MAE)
    mae_bar = np.argmin(post_lows)

    post_records.append({
        "symbol": sym,
        "date": dt,
        "entry_price": entry_price,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "post_change_pct": post_change_pct,
        "is_continuation": is_continuation,
        "mae_bar": mae_bar,
        "post_bars_count": len(post_bars),
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
    })

post_df = pd.DataFrame(post_records)
print(f"\nEntries with post-entry data: {len(post_df)}")

print(f"\nOverall post-entry metrics (1hr):")
print(f"  Avg MFE: +{post_df['mfe_pct'].mean():.2f}%  (median: +{post_df['mfe_pct'].median():.2f}%)")
print(f"  Avg MAE: -{post_df['mae_pct'].mean():.2f}%  (median: -{post_df['mae_pct'].median():.2f}%)")
print(f"  Avg net change: {post_df['post_change_pct'].mean():+.2f}%")
print(f"  Continuation rate: {post_df['is_continuation'].mean()*100:.1f}%")

print(f"\nWinners vs Losers:")
print(f"{'Metric':<25} {'Winners':>10} {'Losers':>10}")
print("-" * 48)
for metric, label in [("mfe_pct", "Avg MFE (%)"),
                       ("mae_pct", "Avg MAE (%)"),
                       ("post_change_pct", "Avg 1hr Change (%)"),
                       ("is_continuation", "Continuation Rate")]:
    w = post_df[post_df["is_winner"]][metric].mean()
    l = post_df[~post_df["is_winner"]][metric].mean()
    if metric == "is_continuation":
        print(f"{label:<25} {w*100:>9.1f}% {l*100:>9.1f}%")
    elif metric == "mae_pct":
        print(f"{label:<25} {-w:>9.2f}% {-l:>9.2f}%")
    else:
        print(f"{label:<25} {w:>+9.2f}% {l:>+9.2f}%")

# MFE/MAE ratio
post_df["mfe_mae_ratio"] = post_df["mfe_pct"] / post_df["mae_pct"].replace(0, np.nan)
print(f"\nMFE/MAE Ratio:")
print(f"  Winners avg: {post_df.loc[post_df['is_winner'], 'mfe_mae_ratio'].mean():.2f}")
print(f"  Losers avg:  {post_df.loc[~post_df['is_winner'], 'mfe_mae_ratio'].mean():.2f}")

# Bar count to MAE
print(f"\nAvg bars to max adverse excursion:")
print(f"  Winners: {post_df.loc[post_df['is_winner'], 'mae_bar'].mean():.1f} bars ({post_df.loc[post_df['is_winner'], 'mae_bar'].mean()*5:.0f} min)")
print(f"  Losers:  {post_df.loc[~post_df['is_winner'], 'mae_bar'].mean():.1f} bars ({post_df.loc[~post_df['is_winner'], 'mae_bar'].mean()*5:.0f} min)")

# ---------------------------------------------------------------------------
# ANALYSIS 4: Intraday Volatility at Entry
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 4: INTRADAY VOLATILITY AT ENTRY")
print("=" * 80)

vol_records = []
for _, row in entries.iterrows():
    sym = row["symbol"]
    dt = row["date"]
    bars = get_day_bars(sym, dt)
    if bars is None or len(bars) < 10:
        continue

    # Calculate ATR-like measure for 5min bars on entry day
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values

    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    atr_5min = np.mean(tr)
    atr_pct = atr_5min / bars["close"].mean() * 100

    # Day range
    day_high = bars["high"].max()
    day_low = bars["low"].min()
    day_range_pct = (day_high - day_low) / day_low * 100

    vol_records.append({
        "symbol": sym,
        "date": dt,
        "atr_5min": atr_5min,
        "atr_pct": atr_pct,
        "day_range_pct": day_range_pct,
        "num_bars": len(bars),
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
    })

vol_df = pd.DataFrame(vol_records)
print(f"\nEntries with volatility data: {len(vol_df)}")

print(f"\nOverall intraday volatility stats:")
print(f"  Avg 5min ATR%: {vol_df['atr_pct'].mean():.3f}%")
print(f"  Avg day range%: {vol_df['day_range_pct'].mean():.2f}%")

# Quartile analysis
vol_df["vol_quartile"] = pd.qcut(vol_df["atr_pct"], q=4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
print(f"\nPnL by intraday volatility quartile:")
print(f"{'Quartile':<15} {'Count':>6} {'AvgPnL':>9} {'WinRate':>9} {'AvgATR%':>9} {'AvgRange%':>10}")
print("-" * 62)
for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
    subset = vol_df[vol_df["vol_quartile"] == q]
    print(f"{q:<15} {len(subset):>6} {subset['pnl_pct'].mean():>8.2f}% "
          f"{subset['is_winner'].mean()*100:>7.1f}% {subset['atr_pct'].mean():>8.3f}% "
          f"{subset['day_range_pct'].mean():>9.2f}%")

print(f"\nWinners vs Losers volatility:")
print(f"  Winners avg ATR%: {vol_df.loc[vol_df['is_winner'], 'atr_pct'].mean():.3f}%  "
      f"day range: {vol_df.loc[vol_df['is_winner'], 'day_range_pct'].mean():.2f}%")
print(f"  Losers avg ATR%:  {vol_df.loc[~vol_df['is_winner'], 'atr_pct'].mean():.3f}%  "
      f"day range: {vol_df.loc[~vol_df['is_winner'], 'day_range_pct'].mean():.2f}%")

# ---------------------------------------------------------------------------
# ANALYSIS 5: Volume Profile at Entry
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 5: VOLUME PROFILE AT ENTRY")
print("=" * 80)

# First build average volume per time slot per symbol
# (to compute relative volume)
ohlcv_sorted["time_slot"] = ohlcv_sorted["timestamp"].apply(time_to_slot)
avg_vol_by_slot = ohlcv_sorted.groupby(["symbol", "time_slot"])["volume"].mean()

vol_profile_records = []
for _, row in entries.iterrows():
    sym = row["symbol"]
    dt = row["date"]
    bar_vol = row["bar_volume"]
    t_slot = time_to_slot(row["entry_time"])

    # Average volume for this symbol at this time slot
    avg_vol = avg_vol_by_slot.get((sym, t_slot), np.nan)
    if pd.isna(avg_vol) or avg_vol == 0:
        continue

    rel_vol = bar_vol / avg_vol

    # Day's average bar volume
    bars = get_day_bars(sym, dt)
    if bars is None:
        continue
    day_avg_vol = bars["volume"].mean()
    rel_vol_day = bar_vol / day_avg_vol if day_avg_vol > 0 else np.nan

    vol_profile_records.append({
        "symbol": sym,
        "date": dt,
        "entry_time": row["entry_time"],
        "bar_volume": bar_vol,
        "avg_slot_volume": avg_vol,
        "relative_volume": rel_vol,
        "rel_vol_day": rel_vol_day,
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
    })

vp_df = pd.DataFrame(vol_profile_records)
print(f"\nEntries with volume profile data: {len(vp_df)}")

print(f"\nOverall volume profile stats:")
print(f"  Avg relative volume (vs time slot avg): {vp_df['relative_volume'].mean():.2f}x")
print(f"  Median relative volume: {vp_df['relative_volume'].median():.2f}x")

# High-volume vs low-volume entries
vp_df["vol_category"] = pd.cut(
    vp_df["relative_volume"],
    bins=[0, 0.5, 0.8, 1.2, 2.0, np.inf],
    labels=["Very Low (<0.5x)", "Low (0.5-0.8x)", "Normal (0.8-1.2x)",
            "High (1.2-2x)", "Surge (>2x)"]
)
print(f"\nPnL by relative volume category:")
print(f"{'Category':<22} {'Count':>6} {'AvgPnL':>9} {'WinRate':>9} {'AvgRelVol':>10}")
print("-" * 60)
for cat in ["Very Low (<0.5x)", "Low (0.5-0.8x)", "Normal (0.8-1.2x)",
            "High (1.2-2x)", "Surge (>2x)"]:
    subset = vp_df[vp_df["vol_category"] == cat]
    if len(subset) == 0:
        continue
    print(f"{cat:<22} {len(subset):>6} {subset['pnl_pct'].mean():>8.2f}% "
          f"{subset['is_winner'].mean()*100:>7.1f}% {subset['relative_volume'].mean():>9.2f}x")

print(f"\nWinners vs Losers volume profile:")
print(f"  Winners avg relative volume: {vp_df.loc[vp_df['is_winner'], 'relative_volume'].mean():.2f}x")
print(f"  Losers avg relative volume:  {vp_df.loc[~vp_df['is_winner'], 'relative_volume'].mean():.2f}x")

# Volume surge analysis
surge = vp_df[vp_df["relative_volume"] > 2.0]
normal = vp_df[(vp_df["relative_volume"] >= 0.8) & (vp_df["relative_volume"] <= 1.2)]
print(f"\nVolume surge (>2x) entries: {len(surge)} trades")
if len(surge) > 0:
    print(f"  Avg PnL: {surge['pnl_pct'].mean():+.2f}%, Win rate: {surge['is_winner'].mean()*100:.1f}%")
print(f"Normal volume (0.8-1.2x) entries: {len(normal)} trades")
if len(normal) > 0:
    print(f"  Avg PnL: {normal['pnl_pct'].mean():+.2f}%, Win rate: {normal['is_winner'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# ANALYSIS 6: Open-to-Entry Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ANALYSIS 6: OPEN-TO-ENTRY ANALYSIS")
print("=" * 80)

ote_records = []
for _, row in entries.iterrows():
    sym = row["symbol"]
    dt = row["date"]
    bars = get_day_bars(sym, dt)
    if bars is None or len(bars) < 7:  # need at least 30min of data
        continue

    day_open = bars.iloc[0]["open"]
    entry_price = row["bar_close"]

    # Distance from open to entry
    open_to_entry_pct = (entry_price - day_open) / day_open * 100

    # Gap analysis: need previous day's close
    # Find previous trading day for this symbol
    all_dates = sorted(ohlcv_sorted[ohlcv_sorted["symbol"] == sym]["date_str"].unique())
    dt_idx = None
    for i, d in enumerate(all_dates):
        if d == dt:
            dt_idx = i
            break
    prev_close = np.nan
    gap_pct = np.nan
    if dt_idx is not None and dt_idx > 0:
        prev_dt = all_dates[dt_idx - 1]
        prev_bars = get_day_bars(sym, prev_dt)
        if prev_bars is not None and len(prev_bars) > 0:
            prev_close = prev_bars.iloc[-1]["close"]
            gap_pct = (day_open - prev_close) / prev_close * 100

    # Opening range (first 30 min = 6 bars)
    first_30min = bars.iloc[:6]
    or_high = first_30min["high"].max()
    or_low = first_30min["low"].min()
    or_range_pct = (or_high - or_low) / or_low * 100

    # Was entry above/below opening range?
    if entry_price > or_high:
        or_position = "Above OR"
    elif entry_price < or_low:
        or_position = "Below OR"
    else:
        or_position = "Within OR"

    ote_records.append({
        "symbol": sym,
        "date": dt,
        "day_open": day_open,
        "entry_price": entry_price,
        "open_to_entry_pct": open_to_entry_pct,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "or_high": or_high,
        "or_low": or_low,
        "or_range_pct": or_range_pct,
        "or_position": or_position,
        "pnl_pct": row["pnl_pct"],
        "is_winner": row["is_winner"],
        "outcome": row["outcome"],
    })

ote_df = pd.DataFrame(ote_records)
print(f"\nEntries with open-to-entry data: {len(ote_df)}")

# Open-to-entry distance
print(f"\nOpen-to-entry distance:")
print(f"  Mean: {ote_df['open_to_entry_pct'].mean():+.2f}%")
print(f"  Median: {ote_df['open_to_entry_pct'].median():+.2f}%")
print(f"  Std: {ote_df['open_to_entry_pct'].std():.2f}%")

print(f"\nWinners vs Losers open-to-entry:")
for outcome in ["Winner", "Loser"]:
    subset = ote_df[ote_df["outcome"] == outcome]
    print(f"  {outcome}s: mean={subset['open_to_entry_pct'].mean():+.2f}%, "
          f"median={subset['open_to_entry_pct'].median():+.2f}%")

# Gap analysis
gap_df = ote_df.dropna(subset=["gap_pct"])
print(f"\nGap analysis ({len(gap_df)} entries with prev-day data):")
gap_up = gap_df[gap_df["gap_pct"] > 0.5]
gap_down = gap_df[gap_df["gap_pct"] < -0.5]
gap_flat = gap_df[(gap_df["gap_pct"] >= -0.5) & (gap_df["gap_pct"] <= 0.5)]

print(f"  Gap up (>0.5%): {len(gap_up)} trades, avg PnL={gap_up['pnl_pct'].mean():+.2f}%, "
      f"win rate={gap_up['is_winner'].mean()*100:.1f}%")
print(f"  Flat gap:       {len(gap_flat)} trades, avg PnL={gap_flat['pnl_pct'].mean():+.2f}%, "
      f"win rate={gap_flat['is_winner'].mean()*100:.1f}%")
print(f"  Gap down (<-0.5%): {len(gap_down)} trades, avg PnL={gap_down['pnl_pct'].mean():+.2f}%, "
      f"win rate={gap_down['is_winner'].mean()*100:.1f}%")

# Opening range breakout
print(f"\nOpening range position at entry:")
print(f"{'Position':<15} {'Count':>6} {'Pct':>7} {'AvgPnL':>9} {'WinRate':>9}")
print("-" * 50)
for pos in ["Above OR", "Within OR", "Below OR"]:
    subset = ote_df[ote_df["or_position"] == pos]
    if len(subset) == 0:
        continue
    pct = len(subset) / len(ote_df) * 100
    print(f"{pos:<15} {len(subset):>6} {pct:>6.1f}% {subset['pnl_pct'].mean():>8.2f}% "
          f"{subset['is_winner'].mean()*100:>7.1f}%")

# Opening range size vs outcome
ote_df["or_q"] = pd.qcut(ote_df["or_range_pct"], q=3, labels=["Tight OR", "Medium OR", "Wide OR"])
print(f"\nPnL by opening range width:")
print(f"{'OR Width':<15} {'Count':>6} {'AvgPnL':>9} {'WinRate':>9} {'AvgOR%':>9}")
print("-" * 52)
for q in ["Tight OR", "Medium OR", "Wide OR"]:
    subset = ote_df[ote_df["or_q"] == q]
    print(f"{q:<15} {len(subset):>6} {subset['pnl_pct'].mean():>8.2f}% "
          f"{subset['is_winner'].mean()*100:>7.1f}% {subset['or_range_pct'].mean():>8.2f}%")

# ---------------------------------------------------------------------------
# SUMMARY & KEY FINDINGS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY: KEY FINDINGS")
print("=" * 80)

findings = []

# Finding 1: Best entry time
best_time_w = entries[entries["is_winner"]]["time_slot"].mode()
best_time_l = entries[~entries["is_winner"]]["time_slot"].mode()
f1 = (f"1. Peak entry time: {peak_slot}. "
      f"Winners cluster at: {best_time_w.iloc[0] if len(best_time_w) > 0 else 'N/A'}, "
      f"Losers cluster at: {best_time_l.iloc[0] if len(best_time_l) > 0 else 'N/A'}.")
findings.append(f1)
print(f"\n{f1}")

# Finding 2: Dip-buy vs momentum
f2 = (f"2. Dip-buy vs Momentum: Dip-buys ({len(dip_buys)}) avg PnL={dip_buys['pnl_pct'].mean():+.2f}%, "
      f"Momentum ({len(mom_buys)}) avg PnL={mom_buys['pnl_pct'].mean():+.2f}%. "
      f"{'Dip-buying' if dip_buys['pnl_pct'].mean() > mom_buys['pnl_pct'].mean() else 'Momentum-buying'} "
      f"produces better outcomes.")
findings.append(f2)
print(f2)

# Finding 3: Post-entry dynamics
w_cont = post_df[post_df["is_winner"]]["is_continuation"].mean() * 100
l_cont = post_df[~post_df["is_winner"]]["is_continuation"].mean() * 100
f3 = (f"3. Post-entry: Winners show {w_cont:.0f}% continuation rate vs "
      f"Losers {l_cont:.0f}%. "
      f"MFE/MAE ratio winners={post_df.loc[post_df['is_winner'], 'mfe_mae_ratio'].mean():.2f} "
      f"vs losers={post_df.loc[~post_df['is_winner'], 'mfe_mae_ratio'].mean():.2f}.")
findings.append(f3)
print(f3)

# Finding 4: Volatility
best_vol_q = vol_df.groupby("vol_quartile", observed=False)["pnl_pct"].mean().idxmax()
f4 = (f"4. Volatility: Best PnL quartile is '{best_vol_q}'. "
      f"Winners ATR%={vol_df.loc[vol_df['is_winner'], 'atr_pct'].mean():.3f}%, "
      f"Losers ATR%={vol_df.loc[~vol_df['is_winner'], 'atr_pct'].mean():.3f}%.")
findings.append(f4)
print(f4)

# Finding 5: Volume
f5 = (f"5. Volume: Entries at {vp_df['relative_volume'].mean():.2f}x avg volume. "
      f"Winners={vp_df.loc[vp_df['is_winner'], 'relative_volume'].mean():.2f}x, "
      f"Losers={vp_df.loc[~vp_df['is_winner'], 'relative_volume'].mean():.2f}x.")
findings.append(f5)
print(f5)

# Finding 6: Gap and opening range
best_or = ote_df.groupby("or_position")["pnl_pct"].mean().idxmax()
best_or_pnl = ote_df.groupby("or_position")["pnl_pct"].mean().max()
f6 = (f"6. Opening range: '{best_or}' entries avg PnL={best_or_pnl:+.2f}%. "
      f"Gap-down buys avg PnL={gap_down['pnl_pct'].mean():+.2f}% "
      f"vs gap-up avg PnL={gap_up['pnl_pct'].mean():+.2f}%.")
findings.append(f6)
print(f6)

# ---------------------------------------------------------------------------
# Save combined report CSV
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Saving report CSV...")
print("-" * 80)

# Build a comprehensive per-entry report
report_rows = []
for _, e_row in entries.iterrows():
    key = (e_row["symbol"], e_row["date"])

    rec = {
        "symbol": e_row["symbol"],
        "date": e_row["date"],
        "entry_time": e_row["entry_hhmm"],
        "time_slot": e_row["time_slot"],
        "price_usd": e_row["price_usd"],
        "bar_close": e_row["bar_close"],
        "price_diff": e_row["price_diff"],
        "pnl_pct": e_row["pnl_pct"],
        "outcome": e_row["outcome"],
        "hold_days": e_row["hold_days"],
        "amount_usd": e_row["amount_usd"],
    }

    # Pre-entry
    pre_match = pre_df[(pre_df["symbol"] == e_row["symbol"]) & (pre_df["date"] == e_row["date"])]
    if len(pre_match) > 0:
        pm = pre_match.iloc[0]
        rec["pre_change_pct"] = pm["pre_change_pct"]
        rec["entry_style"] = "dip_buy" if pm["is_dip_buy"] else "momentum_buy"
        rec["pre_range_pct"] = pm["pre_range_pct"]
    else:
        rec["pre_change_pct"] = np.nan
        rec["entry_style"] = np.nan
        rec["pre_range_pct"] = np.nan

    # Post-entry
    post_match = post_df[(post_df["symbol"] == e_row["symbol"]) & (post_df["date"] == e_row["date"])]
    if len(post_match) > 0:
        pm = post_match.iloc[0]
        rec["post_mfe_pct"] = pm["mfe_pct"]
        rec["post_mae_pct"] = pm["mae_pct"]
        rec["post_change_pct"] = pm["post_change_pct"]
        rec["post_continuation"] = pm["is_continuation"]
    else:
        rec["post_mfe_pct"] = np.nan
        rec["post_mae_pct"] = np.nan
        rec["post_change_pct"] = np.nan
        rec["post_continuation"] = np.nan

    # Volatility
    vol_match = vol_df[(vol_df["symbol"] == e_row["symbol"]) & (vol_df["date"] == e_row["date"])]
    if len(vol_match) > 0:
        vm = vol_match.iloc[0]
        rec["atr_pct"] = vm["atr_pct"]
        rec["day_range_pct"] = vm["day_range_pct"]
    else:
        rec["atr_pct"] = np.nan
        rec["day_range_pct"] = np.nan

    # Volume profile
    vp_match = vp_df[(vp_df["symbol"] == e_row["symbol"]) & (vp_df["date"] == e_row["date"])]
    if len(vp_match) > 0:
        vpm = vp_match.iloc[0]
        rec["relative_volume"] = vpm["relative_volume"]
    else:
        rec["relative_volume"] = np.nan

    # Open-to-entry
    ote_match = ote_df[(ote_df["symbol"] == e_row["symbol"]) & (ote_df["date"] == e_row["date"])]
    if len(ote_match) > 0:
        om = ote_match.iloc[0]
        rec["open_to_entry_pct"] = om["open_to_entry_pct"]
        rec["gap_pct"] = om["gap_pct"]
        rec["or_position"] = om["or_position"]
        rec["or_range_pct"] = om["or_range_pct"]
    else:
        rec["open_to_entry_pct"] = np.nan
        rec["gap_pct"] = np.nan
        rec["or_position"] = np.nan
        rec["or_range_pct"] = np.nan

    report_rows.append(rec)

report_df = pd.DataFrame(report_rows)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
report_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(report_df)} rows to {OUTPUT_CSV}")
print(f"Columns: {report_df.columns.tolist()}")

print(f"\nReport sample (first 5 rows):")
print(report_df.head().to_string(max_colwidth=20))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

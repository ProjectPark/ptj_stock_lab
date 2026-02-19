"""
Technical Indicator Analysis on Trading Entries
================================================
Analyzes RSI, MACD, Bollinger Bands, ATR, and Volume conditions
at the time of each buy entry, then compares winning vs losing trades.

Data sources:
  - data/market/daily/market_daily.parquet  (MultiIndex columns)
  - data/results/analysis/round_trips.csv   (buy/sell round trips with pnl)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MARKET_PATH = BASE_DIR / "data" / "market" / "daily" / "market_daily.parquet"
ROUND_TRIPS_PATH = BASE_DIR / "data" / "results" / "analysis" / "round_trips.csv"
OUTPUT_DIR = BASE_DIR / "data" / "results" / "analysis"
OUTPUT_CSV = OUTPUT_DIR / "technical_indicators_report.csv"
SUMMARY_TXT = OUTPUT_DIR / "technical_indicators_summary.txt"

MAJOR_TICKERS = ["MSTU", "CONL", "ROBN", "AMDL", "NVDL", "BITU"]

# ---------------------------------------------------------------------------
# Indicator computations (manual — no external TA lib required)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands: upper, middle, lower, %B position."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (series - lower) / (upper - lower)  # 0 = lower band, 1 = upper band
    return upper, middle, lower, pct_b


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume relative to N-day moving average."""
    avg_vol = volume.rolling(period).mean()
    return volume / avg_vol


# ---------------------------------------------------------------------------
# Build full indicator DataFrame for each ticker
# ---------------------------------------------------------------------------

def build_indicators(market: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a DataFrame indexed by Date with all indicators for one ticker."""
    try:
        close = market[(ticker, "Close")]
        high = market[(ticker, "High")]
        low = market[(ticker, "Low")]
        volume = market[(ticker, "Volume")]
        open_ = market[(ticker, "Open")]
    except KeyError:
        return pd.DataFrame()

    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["open"] = open_
    df["high"] = high
    df["low"] = low
    df["volume"] = volume

    # RSI
    df["rsi_14"] = compute_rsi(close, 14)

    # MACD
    macd_line, signal_line, histogram = compute_macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    df["macd_above_signal"] = (macd_line > signal_line).astype(int)

    # Bollinger Bands
    upper, middle, lower, pct_b = compute_bollinger(close)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["bb_pct_b"] = pct_b

    # ATR
    df["atr_14"] = compute_atr(high, low, close, 14)
    df["atr_pct"] = df["atr_14"] / close * 100  # ATR as % of price

    # Volume ratio
    df["vol_ratio_20"] = compute_volume_ratio(volume, 20)

    return df


# ---------------------------------------------------------------------------
# Merge trades with indicators
# ---------------------------------------------------------------------------

def merge_trades_with_indicators(
    round_trips: pd.DataFrame, market: pd.DataFrame, tickers: list
) -> pd.DataFrame:
    """For each buy trade in the given tickers, look up indicator values on buy_date."""
    rows = []
    for ticker in tickers:
        ind = build_indicators(market, ticker)
        if ind.empty:
            continue
        ticker_trades = round_trips[round_trips["ticker"] == ticker].copy()
        if ticker_trades.empty:
            continue

        ticker_trades["buy_date"] = pd.to_datetime(ticker_trades["buy_date"])
        for _, trade in ticker_trades.iterrows():
            bdate = trade["buy_date"]
            if bdate not in ind.index:
                continue
            row = ind.loc[bdate].to_dict()
            row["ticker"] = ticker
            row["buy_date"] = bdate
            row["sell_date"] = trade["sell_date"]
            row["buy_price"] = trade["buy_price"]
            row["sell_price"] = trade["sell_price"]
            row["pnl_pct"] = trade["pnl_pct"]
            row["hold_days"] = trade["hold_days"]
            row["win"] = 1 if trade["pnl_pct"] > 0 else 0
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def safe_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
    return {
        "count": len(s),
        "mean": round(s.mean(), 4),
        "median": round(s.median(), 4),
        "std": round(s.std(), 4),
    }


def win_rate(subset: pd.DataFrame) -> float:
    if len(subset) == 0:
        return np.nan
    return round(subset["win"].mean() * 100, 2)


def avg_pnl(subset: pd.DataFrame) -> float:
    if len(subset) == 0:
        return np.nan
    return round(subset["pnl_pct"].mean(), 4)


def group_analysis(df: pd.DataFrame, col: str, bins: list, labels: list):
    """Bin a column and compute win rate / avg pnl per bin."""
    df = df.copy()
    df["_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    results = []
    for label in labels:
        sub = df[df["_bin"] == label]
        results.append({
            "range": label,
            "n_trades": len(sub),
            "win_rate_%": win_rate(sub),
            "avg_pnl_%": avg_pnl(sub),
            "median_pnl_%": round(sub["pnl_pct"].median(), 4) if len(sub) > 0 else np.nan,
        })
    return pd.DataFrame(results)


def separator(title: str, char: str = "=", width: int = 80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    market = pd.read_parquet(MARKET_PATH)
    round_trips = pd.read_csv(ROUND_TRIPS_PATH)
    round_trips["buy_date"] = pd.to_datetime(round_trips["buy_date"])
    round_trips["sell_date"] = pd.to_datetime(round_trips["sell_date"])

    # Determine tickers to analyse (major + fallback if some have no data)
    available_tickers = [t for t in MAJOR_TICKERS if t in market.columns.get_level_values(0).unique()]
    # Add tickers that actually have round trips
    rt_tickers = round_trips["ticker"].unique().tolist()
    analysis_tickers = [t for t in available_tickers if t in rt_tickers]

    # If a major ticker has market data but no round trips, note it
    missing_rt = [t for t in available_tickers if t not in rt_tickers]
    if missing_rt:
        print(f"  Note: {missing_rt} have market data but no round trips — skipped.")

    # Also include any ticker with >= 10 round trips for broader analysis
    all_tickers_for_broad = [t for t in rt_tickers
                             if t in market.columns.get_level_values(0).unique()
                             and len(round_trips[round_trips["ticker"] == t]) >= 10]

    print(f"  Major tickers for analysis: {analysis_tickers}")
    print(f"  All tickers (>= 10 trades) for broad analysis: {all_tickers_for_broad}")

    # ------------------------------------------------------------------
    # Merge: attach indicator values at entry
    # ------------------------------------------------------------------
    print("\nComputing indicators and merging with trades...")
    df_all = merge_trades_with_indicators(round_trips, market, all_tickers_for_broad)
    df_major = df_all[df_all["ticker"].isin(analysis_tickers)].copy()

    print(f"  Total trades with indicators (broad): {len(df_all)}")
    print(f"  Major-ticker trades with indicators:  {len(df_major)}")

    # Overall stats
    total_trades = len(df_all)
    total_wins = df_all["win"].sum()
    overall_wr = round(total_wins / total_trades * 100, 2) if total_trades > 0 else 0
    overall_pnl = round(df_all["pnl_pct"].mean(), 4) if total_trades > 0 else 0

    # Collection for CSV export
    report_rows = []

    # ==================================================================
    separator("OVERALL BASELINE")
    # ==================================================================
    print(f"  Total trades:     {total_trades}")
    print(f"  Wins:             {int(total_wins)}  |  Losses: {total_trades - int(total_wins)}")
    print(f"  Win rate:         {overall_wr}%")
    print(f"  Avg PnL:          {overall_pnl}%")
    print(f"  Median PnL:       {round(df_all['pnl_pct'].median(), 4)}%")

    report_rows.append({
        "section": "BASELINE", "metric": "overall",
        "n_trades": total_trades, "win_rate_%": overall_wr, "avg_pnl_%": overall_pnl,
        "median_pnl_%": round(df_all["pnl_pct"].median(), 4),
    })

    # Per-ticker baseline
    print("\n  Per-ticker baseline:")
    print(f"  {'Ticker':<8} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'MedPnL':>8}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    for t in sorted(all_tickers_for_broad):
        sub = df_all[df_all["ticker"] == t]
        wr = win_rate(sub)
        ap = avg_pnl(sub)
        mp = round(sub["pnl_pct"].median(), 4) if len(sub) > 0 else np.nan
        print(f"  {t:<8} {len(sub):>7} {wr:>7.1f}% {ap:>7.2f}% {mp:>7.2f}%")
        report_rows.append({
            "section": "BASELINE", "metric": f"ticker_{t}",
            "n_trades": len(sub), "win_rate_%": wr, "avg_pnl_%": ap, "median_pnl_%": mp,
        })

    # ==================================================================
    separator("1. RSI(14) AT ENTRY")
    # ==================================================================
    rsi_col = "rsi_14"
    valid_rsi = df_all.dropna(subset=[rsi_col])
    winners = valid_rsi[valid_rsi["win"] == 1]
    losers = valid_rsi[valid_rsi["win"] == 0]

    print(f"\n  RSI distribution at entry (all trades, n={len(valid_rsi)}):")
    print(f"    Mean: {valid_rsi[rsi_col].mean():.2f}  Median: {valid_rsi[rsi_col].median():.2f}")
    print(f"  Winners (n={len(winners)}):  Mean RSI = {winners[rsi_col].mean():.2f}  Median = {winners[rsi_col].median():.2f}")
    print(f"  Losers  (n={len(losers)}):   Mean RSI = {losers[rsi_col].mean():.2f}  Median = {losers[rsi_col].median():.2f}")

    # RSI bins
    rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    rsi_labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-100"]
    rsi_table = group_analysis(valid_rsi, rsi_col, rsi_bins, rsi_labels)
    print(f"\n  RSI zone performance:")
    print(rsi_table.to_string(index=False))
    for _, r in rsi_table.iterrows():
        report_rows.append({
            "section": "RSI_ZONES", "metric": f"rsi_{r['range']}",
            "n_trades": r["n_trades"], "win_rate_%": r["win_rate_%"],
            "avg_pnl_%": r["avg_pnl_%"], "median_pnl_%": r["median_pnl_%"],
        })

    # RSI < 30 test
    rsi_oversold = valid_rsi[valid_rsi[rsi_col] < 30]
    rsi_not_oversold = valid_rsi[valid_rsi[rsi_col] >= 30]
    print(f"\n  RSI < 30 (oversold) test:")
    print(f"    RSI < 30:  n={len(rsi_oversold):>4}  WR={win_rate(rsi_oversold):>6.1f}%  AvgPnL={avg_pnl(rsi_oversold):>7.2f}%")
    print(f"    RSI >= 30: n={len(rsi_not_oversold):>4}  WR={win_rate(rsi_not_oversold):>6.1f}%  AvgPnL={avg_pnl(rsi_not_oversold):>7.2f}%")

    report_rows.append({
        "section": "RSI_THRESHOLD", "metric": "rsi_lt_30",
        "n_trades": len(rsi_oversold), "win_rate_%": win_rate(rsi_oversold),
        "avg_pnl_%": avg_pnl(rsi_oversold), "median_pnl_%": round(rsi_oversold["pnl_pct"].median(), 4) if len(rsi_oversold) > 0 else np.nan,
    })
    report_rows.append({
        "section": "RSI_THRESHOLD", "metric": "rsi_gte_30",
        "n_trades": len(rsi_not_oversold), "win_rate_%": win_rate(rsi_not_oversold),
        "avg_pnl_%": avg_pnl(rsi_not_oversold), "median_pnl_%": round(rsi_not_oversold["pnl_pct"].median(), 4) if len(rsi_not_oversold) > 0 else np.nan,
    })

    # RSI < 40 test
    rsi_under40 = valid_rsi[valid_rsi[rsi_col] < 40]
    rsi_above40 = valid_rsi[valid_rsi[rsi_col] >= 40]
    print(f"\n  RSI < 40 test:")
    print(f"    RSI < 40:  n={len(rsi_under40):>4}  WR={win_rate(rsi_under40):>6.1f}%  AvgPnL={avg_pnl(rsi_under40):>7.2f}%")
    print(f"    RSI >= 40: n={len(rsi_above40):>4}  WR={win_rate(rsi_above40):>6.1f}%  AvgPnL={avg_pnl(rsi_above40):>7.2f}%")

    # ==================================================================
    separator("2. MACD SIGNAL AT ENTRY")
    # ==================================================================
    macd_valid = df_all.dropna(subset=["macd_line", "macd_signal", "macd_hist"])

    macd_above = macd_valid[macd_valid["macd_above_signal"] == 1]
    macd_below = macd_valid[macd_valid["macd_above_signal"] == 0]

    print(f"\n  MACD line vs Signal line at entry (n={len(macd_valid)}):")
    print(f"    MACD > Signal (bullish): n={len(macd_above):>4}  WR={win_rate(macd_above):>6.1f}%  AvgPnL={avg_pnl(macd_above):>7.2f}%  MedPnL={round(macd_above['pnl_pct'].median(),2) if len(macd_above)>0 else 'N/A':>7}%")
    print(f"    MACD < Signal (bearish): n={len(macd_below):>4}  WR={win_rate(macd_below):>6.1f}%  AvgPnL={avg_pnl(macd_below):>7.2f}%  MedPnL={round(macd_below['pnl_pct'].median(),2) if len(macd_below)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "MACD_SIGNAL", "metric": "macd_above_signal",
        "n_trades": len(macd_above), "win_rate_%": win_rate(macd_above),
        "avg_pnl_%": avg_pnl(macd_above),
        "median_pnl_%": round(macd_above["pnl_pct"].median(), 4) if len(macd_above) > 0 else np.nan,
    })
    report_rows.append({
        "section": "MACD_SIGNAL", "metric": "macd_below_signal",
        "n_trades": len(macd_below), "win_rate_%": win_rate(macd_below),
        "avg_pnl_%": avg_pnl(macd_below),
        "median_pnl_%": round(macd_below["pnl_pct"].median(), 4) if len(macd_below) > 0 else np.nan,
    })

    # MACD histogram positive vs negative
    hist_pos = macd_valid[macd_valid["macd_hist"] > 0]
    hist_neg = macd_valid[macd_valid["macd_hist"] <= 0]
    print(f"\n  MACD histogram at entry:")
    print(f"    Histogram > 0: n={len(hist_pos):>4}  WR={win_rate(hist_pos):>6.1f}%  AvgPnL={avg_pnl(hist_pos):>7.2f}%")
    print(f"    Histogram <= 0: n={len(hist_neg):>4}  WR={win_rate(hist_neg):>6.1f}%  AvgPnL={avg_pnl(hist_neg):>7.2f}%")

    report_rows.append({
        "section": "MACD_HISTOGRAM", "metric": "hist_positive",
        "n_trades": len(hist_pos), "win_rate_%": win_rate(hist_pos),
        "avg_pnl_%": avg_pnl(hist_pos),
        "median_pnl_%": round(hist_pos["pnl_pct"].median(), 4) if len(hist_pos) > 0 else np.nan,
    })
    report_rows.append({
        "section": "MACD_HISTOGRAM", "metric": "hist_negative",
        "n_trades": len(hist_neg), "win_rate_%": win_rate(hist_neg),
        "avg_pnl_%": avg_pnl(hist_neg),
        "median_pnl_%": round(hist_neg["pnl_pct"].median(), 4) if len(hist_neg) > 0 else np.nan,
    })

    # MACD histogram bins for finer analysis
    macd_valid_h = macd_valid.copy()
    macd_valid_h["macd_hist_norm"] = macd_valid_h["macd_hist"] / macd_valid_h["close"] * 100  # normalize by price
    print(f"\n  MACD histogram (price-normalized %) distribution:")
    print(f"    Winners mean: {macd_valid_h[macd_valid_h['win']==1]['macd_hist_norm'].mean():.4f}%")
    print(f"    Losers  mean: {macd_valid_h[macd_valid_h['win']==0]['macd_hist_norm'].mean():.4f}%")

    # ==================================================================
    separator("3. BOLLINGER BAND POSITION AT ENTRY")
    # ==================================================================
    bb_valid = df_all.dropna(subset=["bb_pct_b"])

    print(f"\n  Bollinger Band %B at entry (n={len(bb_valid)}):")
    print(f"    0 = lower band, 0.5 = middle, 1 = upper band")
    bb_winners = bb_valid[bb_valid["win"] == 1]
    bb_losers = bb_valid[bb_valid["win"] == 0]
    print(f"    Winners: Mean %B = {bb_winners['bb_pct_b'].mean():.3f}  Median = {bb_winners['bb_pct_b'].median():.3f}")
    print(f"    Losers:  Mean %B = {bb_losers['bb_pct_b'].mean():.3f}  Median = {bb_losers['bb_pct_b'].median():.3f}")

    # BB zone analysis
    bb_bins = [-999, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 999]
    bb_labels = ["<0 (below LB)", "0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1.0 (above UB)"]
    bb_table = group_analysis(bb_valid, "bb_pct_b", bb_bins, bb_labels)
    print(f"\n  Bollinger Band %B zone performance:")
    print(bb_table.to_string(index=False))
    for _, r in bb_table.iterrows():
        report_rows.append({
            "section": "BB_ZONES", "metric": f"bb_{r['range']}",
            "n_trades": r["n_trades"], "win_rate_%": r["win_rate_%"],
            "avg_pnl_%": r["avg_pnl_%"], "median_pnl_%": r["median_pnl_%"],
        })

    # Mean reversion: price below lower BB
    below_lb = bb_valid[bb_valid["bb_pct_b"] < 0]
    above_ub = bb_valid[bb_valid["bb_pct_b"] > 1]
    in_band = bb_valid[(bb_valid["bb_pct_b"] >= 0) & (bb_valid["bb_pct_b"] <= 1)]
    print(f"\n  Mean-reversion test:")
    print(f"    Below lower BB (%B<0):  n={len(below_lb):>4}  WR={win_rate(below_lb):>6.1f}%  AvgPnL={avg_pnl(below_lb):>7.2f}%")
    print(f"    Within BB (0<=%B<=1):   n={len(in_band):>4}  WR={win_rate(in_band):>6.1f}%  AvgPnL={avg_pnl(in_band):>7.2f}%")
    print(f"    Above upper BB (%B>1):  n={len(above_ub):>4}  WR={win_rate(above_ub):>6.1f}%  AvgPnL={avg_pnl(above_ub):>7.2f}%")

    # ==================================================================
    separator("4. ATR(14) AT ENTRY — VOLATILITY STATE")
    # ==================================================================
    atr_valid = df_all.dropna(subset=["atr_pct"])

    # Compute ATR percentile for each entry relative to its ticker's ATR history
    atr_valid = atr_valid.copy()
    # We compute ATR percentile across all dates for that ticker
    atr_percentiles = []
    for _, row in atr_valid.iterrows():
        ticker = row["ticker"]
        ind = build_indicators(market, ticker)
        if ind.empty or row["buy_date"] not in ind.index:
            atr_percentiles.append(np.nan)
            continue
        atr_series = ind["atr_pct"].dropna()
        current_atr = row["atr_pct"]
        pctile = (atr_series < current_atr).mean() * 100
        atr_percentiles.append(round(pctile, 2))
    atr_valid["atr_percentile"] = atr_percentiles

    atr_winners = atr_valid[atr_valid["win"] == 1]
    atr_losers = atr_valid[atr_valid["win"] == 0]

    print(f"\n  ATR (% of price) at entry (n={len(atr_valid)}):")
    print(f"    Winners: Mean ATR% = {atr_winners['atr_pct'].mean():.2f}%  Median = {atr_winners['atr_pct'].median():.2f}%")
    print(f"    Losers:  Mean ATR% = {atr_losers['atr_pct'].mean():.2f}%  Median = {atr_losers['atr_pct'].median():.2f}%")

    print(f"\n  ATR percentile at entry:")
    print(f"    Winners: Mean = {atr_winners['atr_percentile'].mean():.1f}th  Median = {atr_winners['atr_percentile'].median():.1f}th")
    print(f"    Losers:  Mean = {atr_losers['atr_percentile'].mean():.1f}th  Median = {atr_losers['atr_percentile'].median():.1f}th")

    # ATR percentile buckets
    atr_bins = [0, 25, 50, 75, 100]
    atr_labels = ["Q1 (0-25)", "Q2 (25-50)", "Q3 (50-75)", "Q4 (75-100)"]
    atr_table = group_analysis(atr_valid, "atr_percentile", atr_bins, atr_labels)
    print(f"\n  ATR percentile quartile performance:")
    print(atr_table.to_string(index=False))
    for _, r in atr_table.iterrows():
        report_rows.append({
            "section": "ATR_QUARTILES", "metric": f"atr_{r['range']}",
            "n_trades": r["n_trades"], "win_rate_%": r["win_rate_%"],
            "avg_pnl_%": r["avg_pnl_%"], "median_pnl_%": r["median_pnl_%"],
        })

    # High vs Low ATR
    atr_high = atr_valid[atr_valid["atr_percentile"] >= 75]
    atr_low = atr_valid[atr_valid["atr_percentile"] < 25]
    atr_mid = atr_valid[(atr_valid["atr_percentile"] >= 25) & (atr_valid["atr_percentile"] < 75)]
    print(f"\n  High vs Low ATR:")
    print(f"    High ATR (>=75th pctile): n={len(atr_high):>4}  WR={win_rate(atr_high):>6.1f}%  AvgPnL={avg_pnl(atr_high):>7.2f}%")
    print(f"    Mid  ATR (25-75th):       n={len(atr_mid):>4}  WR={win_rate(atr_mid):>6.1f}%  AvgPnL={avg_pnl(atr_mid):>7.2f}%")
    print(f"    Low  ATR (<25th pctile):  n={len(atr_low):>4}  WR={win_rate(atr_low):>6.1f}%  AvgPnL={avg_pnl(atr_low):>7.2f}%")

    # ==================================================================
    separator("5. VOLUME ANALYSIS — RELATIVE TO 20-DAY AVERAGE")
    # ==================================================================
    vol_valid = df_all.dropna(subset=["vol_ratio_20"])

    vol_winners = vol_valid[vol_valid["win"] == 1]
    vol_losers = vol_valid[vol_valid["win"] == 0]

    print(f"\n  Volume ratio (vol / 20d avg) at entry (n={len(vol_valid)}):")
    print(f"    Winners: Mean = {vol_winners['vol_ratio_20'].mean():.2f}x  Median = {vol_winners['vol_ratio_20'].median():.2f}x")
    print(f"    Losers:  Mean = {vol_losers['vol_ratio_20'].mean():.2f}x  Median = {vol_losers['vol_ratio_20'].median():.2f}x")

    # High volume (>1.5x) vs normal
    vol_high = vol_valid[vol_valid["vol_ratio_20"] > 1.5]
    vol_normal = vol_valid[(vol_valid["vol_ratio_20"] >= 0.5) & (vol_valid["vol_ratio_20"] <= 1.5)]
    vol_low = vol_valid[vol_valid["vol_ratio_20"] < 0.5]

    print(f"\n  Volume threshold analysis:")
    print(f"    High volume (>1.5x):    n={len(vol_high):>4}  WR={win_rate(vol_high):>6.1f}%  AvgPnL={avg_pnl(vol_high):>7.2f}%  MedPnL={round(vol_high['pnl_pct'].median(),2) if len(vol_high)>0 else 'N/A':>7}%")
    print(f"    Normal  (0.5-1.5x):     n={len(vol_normal):>4}  WR={win_rate(vol_normal):>6.1f}%  AvgPnL={avg_pnl(vol_normal):>7.2f}%  MedPnL={round(vol_normal['pnl_pct'].median(),2) if len(vol_normal)>0 else 'N/A':>7}%")
    print(f"    Low volume (<0.5x):     n={len(vol_low):>4}  WR={win_rate(vol_low):>6.1f}%  AvgPnL={avg_pnl(vol_low):>7.2f}%  MedPnL={round(vol_low['pnl_pct'].median(),2) if len(vol_low)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "VOLUME", "metric": "vol_high_gt1.5x",
        "n_trades": len(vol_high), "win_rate_%": win_rate(vol_high),
        "avg_pnl_%": avg_pnl(vol_high),
        "median_pnl_%": round(vol_high["pnl_pct"].median(), 4) if len(vol_high) > 0 else np.nan,
    })
    report_rows.append({
        "section": "VOLUME", "metric": "vol_normal_0.5-1.5x",
        "n_trades": len(vol_normal), "win_rate_%": win_rate(vol_normal),
        "avg_pnl_%": avg_pnl(vol_normal),
        "median_pnl_%": round(vol_normal["pnl_pct"].median(), 4) if len(vol_normal) > 0 else np.nan,
    })
    report_rows.append({
        "section": "VOLUME", "metric": "vol_low_lt0.5x",
        "n_trades": len(vol_low), "win_rate_%": win_rate(vol_low),
        "avg_pnl_%": avg_pnl(vol_low),
        "median_pnl_%": round(vol_low["pnl_pct"].median(), 4) if len(vol_low) > 0 else np.nan,
    })

    # Volume bins
    vol_bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 999]
    vol_labels = ["<0.5x", "0.5-0.8x", "0.8-1.0x", "1.0-1.2x", "1.2-1.5x", "1.5-2.0x", "2.0-3.0x", ">3.0x"]
    vol_table = group_analysis(vol_valid, "vol_ratio_20", vol_bins, vol_labels)
    print(f"\n  Volume ratio bucket performance:")
    print(vol_table.to_string(index=False))

    # ==================================================================
    separator("6. MULTI-INDICATOR COMBOS")
    # ==================================================================
    combo_valid = df_all.dropna(subset=[rsi_col, "macd_above_signal", "bb_pct_b", "vol_ratio_20"])

    # Combo 1: RSI<40 + MACD bearish (below signal) → "oversold dip buy"
    combo1 = combo_valid[(combo_valid[rsi_col] < 40) & (combo_valid["macd_above_signal"] == 0)]
    combo1_inv = combo_valid[~((combo_valid[rsi_col] < 40) & (combo_valid["macd_above_signal"] == 0))]
    print(f"\n  Combo 1: RSI<40 + MACD below signal ('oversold dip buy'):")
    print(f"    Match:     n={len(combo1):>4}  WR={win_rate(combo1):>6.1f}%  AvgPnL={avg_pnl(combo1):>7.2f}%  MedPnL={round(combo1['pnl_pct'].median(),2) if len(combo1)>0 else 'N/A':>7}%")
    print(f"    No match:  n={len(combo1_inv):>4}  WR={win_rate(combo1_inv):>6.1f}%  AvgPnL={avg_pnl(combo1_inv):>7.2f}%  MedPnL={round(combo1_inv['pnl_pct'].median(),2) if len(combo1_inv)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "COMBO", "metric": "rsi_lt40_macd_bearish",
        "n_trades": len(combo1), "win_rate_%": win_rate(combo1),
        "avg_pnl_%": avg_pnl(combo1),
        "median_pnl_%": round(combo1["pnl_pct"].median(), 4) if len(combo1) > 0 else np.nan,
    })

    # Combo 2: Price below lower BB + high volume → "capitulation buy"
    combo2 = combo_valid[(combo_valid["bb_pct_b"] < 0) & (combo_valid["vol_ratio_20"] > 1.5)]
    combo2_inv = combo_valid[~((combo_valid["bb_pct_b"] < 0) & (combo_valid["vol_ratio_20"] > 1.5))]
    print(f"\n  Combo 2: Below lower BB + Volume>1.5x ('capitulation buy'):")
    print(f"    Match:     n={len(combo2):>4}  WR={win_rate(combo2):>6.1f}%  AvgPnL={avg_pnl(combo2):>7.2f}%  MedPnL={round(combo2['pnl_pct'].median(),2) if len(combo2)>0 else 'N/A':>7}%")
    print(f"    No match:  n={len(combo2_inv):>4}  WR={win_rate(combo2_inv):>6.1f}%  AvgPnL={avg_pnl(combo2_inv):>7.2f}%  MedPnL={round(combo2_inv['pnl_pct'].median(),2) if len(combo2_inv)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "COMBO", "metric": "below_lb_high_vol",
        "n_trades": len(combo2), "win_rate_%": win_rate(combo2),
        "avg_pnl_%": avg_pnl(combo2),
        "median_pnl_%": round(combo2["pnl_pct"].median(), 4) if len(combo2) > 0 else np.nan,
    })

    # Combo 3: RSI<30 + below lower BB → "extreme oversold"
    combo3 = combo_valid[(combo_valid[rsi_col] < 30) & (combo_valid["bb_pct_b"] < 0)]
    combo3_inv = combo_valid[~((combo_valid[rsi_col] < 30) & (combo_valid["bb_pct_b"] < 0))]
    print(f"\n  Combo 3: RSI<30 + Below lower BB ('extreme oversold'):")
    print(f"    Match:     n={len(combo3):>4}  WR={win_rate(combo3):>6.1f}%  AvgPnL={avg_pnl(combo3):>7.2f}%  MedPnL={round(combo3['pnl_pct'].median(),2) if len(combo3)>0 else 'N/A':>7}%")
    print(f"    No match:  n={len(combo3_inv):>4}  WR={win_rate(combo3_inv):>6.1f}%  AvgPnL={avg_pnl(combo3_inv):>7.2f}%  MedPnL={round(combo3_inv['pnl_pct'].median(),2) if len(combo3_inv)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "COMBO", "metric": "rsi_lt30_below_lb",
        "n_trades": len(combo3), "win_rate_%": win_rate(combo3),
        "avg_pnl_%": avg_pnl(combo3),
        "median_pnl_%": round(combo3["pnl_pct"].median(), 4) if len(combo3) > 0 else np.nan,
    })

    # Combo 4: RSI<40 + MACD bearish + High volume → "strong dip buy"
    combo4 = combo_valid[
        (combo_valid[rsi_col] < 40)
        & (combo_valid["macd_above_signal"] == 0)
        & (combo_valid["vol_ratio_20"] > 1.5)
    ]
    combo4_inv = combo_valid[
        ~(
            (combo_valid[rsi_col] < 40)
            & (combo_valid["macd_above_signal"] == 0)
            & (combo_valid["vol_ratio_20"] > 1.5)
        )
    ]
    print(f"\n  Combo 4: RSI<40 + MACD bearish + Vol>1.5x ('strong dip buy'):")
    print(f"    Match:     n={len(combo4):>4}  WR={win_rate(combo4):>6.1f}%  AvgPnL={avg_pnl(combo4):>7.2f}%  MedPnL={round(combo4['pnl_pct'].median(),2) if len(combo4)>0 else 'N/A':>7}%")
    print(f"    No match:  n={len(combo4_inv):>4}  WR={win_rate(combo4_inv):>6.1f}%  AvgPnL={avg_pnl(combo4_inv):>7.2f}%  MedPnL={round(combo4_inv['pnl_pct'].median(),2) if len(combo4_inv)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "COMBO", "metric": "rsi_lt40_macd_bear_highvol",
        "n_trades": len(combo4), "win_rate_%": win_rate(combo4),
        "avg_pnl_%": avg_pnl(combo4),
        "median_pnl_%": round(combo4["pnl_pct"].median(), 4) if len(combo4) > 0 else np.nan,
    })

    # Combo 5: MACD bullish + RSI 40-60 + normal volume → "trend continuation"
    combo5 = combo_valid[
        (combo_valid["macd_above_signal"] == 1)
        & (combo_valid[rsi_col] >= 40) & (combo_valid[rsi_col] <= 60)
        & (combo_valid["vol_ratio_20"] >= 0.8) & (combo_valid["vol_ratio_20"] <= 1.5)
    ]
    combo5_inv = combo_valid[
        ~(
            (combo_valid["macd_above_signal"] == 1)
            & (combo_valid[rsi_col] >= 40) & (combo_valid[rsi_col] <= 60)
            & (combo_valid["vol_ratio_20"] >= 0.8) & (combo_valid["vol_ratio_20"] <= 1.5)
        )
    ]
    print(f"\n  Combo 5: MACD bullish + RSI 40-60 + normal vol ('trend continuation'):")
    print(f"    Match:     n={len(combo5):>4}  WR={win_rate(combo5):>6.1f}%  AvgPnL={avg_pnl(combo5):>7.2f}%  MedPnL={round(combo5['pnl_pct'].median(),2) if len(combo5)>0 else 'N/A':>7}%")
    print(f"    No match:  n={len(combo5_inv):>4}  WR={win_rate(combo5_inv):>6.1f}%  AvgPnL={avg_pnl(combo5_inv):>7.2f}%  MedPnL={round(combo5_inv['pnl_pct'].median(),2) if len(combo5_inv)>0 else 'N/A':>7}%")

    report_rows.append({
        "section": "COMBO", "metric": "macd_bull_rsi40-60_normvol",
        "n_trades": len(combo5), "win_rate_%": win_rate(combo5),
        "avg_pnl_%": avg_pnl(combo5),
        "median_pnl_%": round(combo5["pnl_pct"].median(), 4) if len(combo5) > 0 else np.nan,
    })

    # ==================================================================
    separator("7. PER-TICKER INDICATOR ANALYSIS (MAJOR TICKERS)")
    # ==================================================================
    for ticker in analysis_tickers:
        tsub = df_all[df_all["ticker"] == ticker]
        if len(tsub) < 5:
            continue
        print(f"\n  --- {ticker} (n={len(tsub)}, WR={win_rate(tsub):.1f}%, AvgPnL={avg_pnl(tsub):.2f}%) ---")

        # RSI
        tsub_rsi = tsub.dropna(subset=[rsi_col])
        tw = tsub_rsi[tsub_rsi["win"] == 1]
        tl = tsub_rsi[tsub_rsi["win"] == 0]
        if len(tsub_rsi) > 0:
            print(f"    RSI:  Winners mean={tw[rsi_col].mean():.1f}  Losers mean={tl[rsi_col].mean():.1f}")

        # MACD
        tsub_macd = tsub.dropna(subset=["macd_above_signal"])
        ma = tsub_macd[tsub_macd["macd_above_signal"] == 1]
        mb = tsub_macd[tsub_macd["macd_above_signal"] == 0]
        if len(tsub_macd) > 0:
            print(f"    MACD: Bullish n={len(ma)} WR={win_rate(ma):.1f}%  |  Bearish n={len(mb)} WR={win_rate(mb):.1f}%")

        # BB
        tsub_bb = tsub.dropna(subset=["bb_pct_b"])
        if len(tsub_bb) > 0:
            tw_bb = tsub_bb[tsub_bb["win"] == 1]
            tl_bb = tsub_bb[tsub_bb["win"] == 0]
            print(f"    BB %B: Winners mean={tw_bb['bb_pct_b'].mean():.3f}  Losers mean={tl_bb['bb_pct_b'].mean():.3f}")

        # Volume
        tsub_vol = tsub.dropna(subset=["vol_ratio_20"])
        if len(tsub_vol) > 0:
            tw_v = tsub_vol[tsub_vol["win"] == 1]
            tl_v = tsub_vol[tsub_vol["win"] == 0]
            print(f"    Vol:  Winners mean={tw_v['vol_ratio_20'].mean():.2f}x  Losers mean={tl_v['vol_ratio_20'].mean():.2f}x")

    # ==================================================================
    separator("8. KEY FINDINGS SUMMARY")
    # ==================================================================
    summary_lines = []

    def finding(text):
        print(f"  - {text}")
        summary_lines.append(text)

    # RSI finding
    rsi_best_zone = rsi_table.loc[rsi_table["avg_pnl_%"].idxmax()]
    finding(f"RSI: Best entry zone is RSI {rsi_best_zone['range']} "
            f"(n={int(rsi_best_zone['n_trades'])}, WR={rsi_best_zone['win_rate_%']}%, "
            f"AvgPnL={rsi_best_zone['avg_pnl_%']}%)")

    rsi30_better = avg_pnl(rsi_oversold) if len(rsi_oversold) > 0 else -999
    rsi30_baseline = avg_pnl(rsi_not_oversold) if len(rsi_not_oversold) > 0 else -999
    if rsi30_better > rsi30_baseline:
        finding(f"RSI<30 (oversold) outperforms: {avg_pnl(rsi_oversold):.2f}% vs {avg_pnl(rsi_not_oversold):.2f}%")
    else:
        finding(f"RSI<30 does NOT outperform: {avg_pnl(rsi_oversold):.2f}% vs {avg_pnl(rsi_not_oversold):.2f}%")

    # MACD finding
    macd_above_pnl = avg_pnl(macd_above)
    macd_below_pnl = avg_pnl(macd_below)
    if macd_above_pnl > macd_below_pnl:
        finding(f"MACD bullish entries outperform bearish: {macd_above_pnl:.2f}% vs {macd_below_pnl:.2f}%")
    else:
        finding(f"MACD bearish entries outperform bullish: {macd_below_pnl:.2f}% vs {macd_above_pnl:.2f}% (contrarian signal)")

    # BB finding
    if len(bb_winners) > 0 and len(bb_losers) > 0:
        if bb_winners["bb_pct_b"].mean() < bb_losers["bb_pct_b"].mean():
            finding(f"BB: Winners enter closer to lower band (mean %B: {bb_winners['bb_pct_b'].mean():.3f} vs {bb_losers['bb_pct_b'].mean():.3f}) — mean reversion works")
        else:
            finding(f"BB: Winners enter closer to upper band (mean %B: {bb_winners['bb_pct_b'].mean():.3f} vs {bb_losers['bb_pct_b'].mean():.3f}) — momentum works")

    # ATR finding
    if len(atr_high) > 0 and len(atr_low) > 0:
        if avg_pnl(atr_high) > avg_pnl(atr_low):
            finding(f"ATR: High volatility entries better: {avg_pnl(atr_high):.2f}% vs low vol {avg_pnl(atr_low):.2f}%")
        else:
            finding(f"ATR: Low volatility entries better: {avg_pnl(atr_low):.2f}% vs high vol {avg_pnl(atr_high):.2f}%")

    # Volume finding
    if len(vol_high) > 0 and len(vol_normal) > 0:
        if avg_pnl(vol_high) > avg_pnl(vol_normal):
            finding(f"Volume: High vol entries (>1.5x) outperform: {avg_pnl(vol_high):.2f}% vs normal {avg_pnl(vol_normal):.2f}%")
        else:
            finding(f"Volume: Normal vol entries outperform high vol: {avg_pnl(vol_normal):.2f}% vs {avg_pnl(vol_high):.2f}%")

    # Best combo
    combo_results = [
        ("RSI<40 + MACD bearish", combo1),
        ("Below LB + High vol", combo2),
        ("RSI<30 + Below LB", combo3),
        ("RSI<40 + MACD bear + High vol", combo4),
        ("MACD bull + RSI 40-60 + normal vol", combo5),
    ]
    best_combo = max(combo_results, key=lambda x: avg_pnl(x[1]) if len(x[1]) > 2 else -999)
    best_combo_wr_leader = max(combo_results, key=lambda x: win_rate(x[1]) if len(x[1]) > 2 else -999)
    if len(best_combo[1]) > 2:
        finding(f"Best combo by AvgPnL: '{best_combo[0]}' — n={len(best_combo[1])}, "
                f"WR={win_rate(best_combo[1]):.1f}%, AvgPnL={avg_pnl(best_combo[1]):.2f}%")
    if len(best_combo_wr_leader[1]) > 2 and best_combo_wr_leader[0] != best_combo[0]:
        finding(f"Best combo by WinRate: '{best_combo_wr_leader[0]}' — n={len(best_combo_wr_leader[1])}, "
                f"WR={win_rate(best_combo_wr_leader[1]):.1f}%, AvgPnL={avg_pnl(best_combo_wr_leader[1]):.2f}%")

    # ==================================================================
    # Save outputs
    # ==================================================================
    separator("SAVING OUTPUTS")

    # CSV report
    report_df = pd.DataFrame(report_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV report saved to: {OUTPUT_CSV}")

    # Summary text
    with open(SUMMARY_TXT, "w") as f:
        f.write("Technical Indicator Analysis — Key Findings\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: 2026-02-19\n")
        f.write(f"Total trades analysed: {total_trades}\n")
        f.write(f"Overall win rate: {overall_wr}%\n")
        f.write(f"Overall avg PnL: {overall_pnl}%\n\n")
        f.write("Key Findings:\n")
        for line in summary_lines:
            f.write(f"  - {line}\n")
    print(f"  Summary saved to: {SUMMARY_TXT}")

    # Also save the full per-trade indicator data for further analysis
    trade_detail_path = OUTPUT_DIR / "technical_indicators_per_trade.csv"
    export_cols = [
        "ticker", "buy_date", "sell_date", "buy_price", "sell_price",
        "pnl_pct", "hold_days", "win",
        "rsi_14", "macd_line", "macd_signal", "macd_hist", "macd_above_signal",
        "bb_pct_b", "bb_upper", "bb_middle", "bb_lower",
        "atr_14", "atr_pct", "vol_ratio_20",
    ]
    available_cols = [c for c in export_cols if c in df_all.columns]
    df_all[available_cols].to_csv(trade_detail_path, index=False)
    print(f"  Per-trade detail saved to: {trade_detail_path}")

    print(f"\nDone. Analysed {total_trades} trades across {len(all_tickers_for_broad)} tickers.")


if __name__ == "__main__":
    main()

"""
Calendar Effects, Cross-Correlation & Lead-Lag Analysis
========================================================
Analyzes:
  Part 1 — Calendar effects (day-of-week, week-of-month, monthly trends, big-move reactions)
  Part 2 — Cross-correlation & lead-lag (return correlation, lead-lag, pair spread stationarity)
  Part 3 — Conditional returns (SPY regime, GLD-BTC divergence, sector rotation)

Data sources:
  - data/market/daily/market_daily.parquet
  - data/results/analysis/decision_log.csv
  - data/results/analysis/decision_log_trades.csv
  - data/results/analysis/round_trips.csv

Outputs:
  - data/results/analysis/correlation_matrix.csv
  - data/results/analysis/calendar_effects.csv
  - data/results/analysis/lead_lag_report.csv
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results" / "analysis"
RESULTS.mkdir(parents=True, exist_ok=True)

MARKET_PATH = DATA / "market" / "daily" / "market_daily.parquet"
DECISION_LOG_PATH = RESULTS / "decision_log.csv"
TRADES_PATH = RESULTS / "decision_log_trades.csv"
ROUND_TRIPS_PATH = RESULTS / "round_trips.csv"


def sep(title: str, char: str = "=", width: int = 80) -> None:
    """Print a section separator."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def sub_sep(title: str, char: str = "-", width: int = 60) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    """Load all data sources and return them."""
    print("Loading data...")

    # Market daily — MultiIndex columns
    market = pd.read_parquet(MARKET_PATH)
    print(f"  market_daily: {market.shape[0]} days x {market.shape[1]} cols "
          f"({len(set(c[0] for c in market.columns))} tickers)")

    # Decision log
    dlog = pd.read_csv(DECISION_LOG_PATH)
    dlog["date"] = pd.to_datetime(dlog["date"])
    print(f"  decision_log: {len(dlog)} rows")

    # Trades
    trades = pd.read_csv(TRADES_PATH)
    trades["date"] = pd.to_datetime(trades["date"])
    print(f"  trades: {len(trades)} rows")

    # Round trips
    rt = pd.read_csv(ROUND_TRIPS_PATH)
    rt["buy_date"] = pd.to_datetime(rt["buy_date"])
    rt["sell_date"] = pd.to_datetime(rt["sell_date"])
    print(f"  round_trips: {len(rt)} rows")

    return market, dlog, trades, rt


# ===========================================================================
# PART 1: Calendar Effects
# ===========================================================================

def part1_calendar_effects(dlog: pd.DataFrame, trades: pd.DataFrame,
                           rt: pd.DataFrame) -> pd.DataFrame:
    """Analyze calendar patterns in buying/selling behavior."""
    sep("PART 1: CALENDAR EFFECTS")

    # -- Prepare --
    dlog = dlog.copy()
    rt = rt.copy()
    trades = trades.copy()

    # Add weekday number (0=Mon, 4=Fri)
    dlog["weekday_num"] = dlog["date"].dt.weekday
    dlog["weekday_name"] = dlog["date"].dt.day_name()
    dlog["is_buy_day"] = dlog["action"].isin(["BUY", "BUY+SELL"])

    rt["buy_weekday_num"] = rt["buy_date"].dt.weekday
    rt["buy_weekday_name"] = rt["buy_date"].dt.day_name()
    rt["is_win"] = rt["pnl_pct"] > 0

    # -----------------------------------------------------------------------
    # 1. Day of Week
    # -----------------------------------------------------------------------
    sub_sep("1. Day of Week Analysis")

    # Buy probability by day
    dow_buy = dlog.groupby("weekday_num").agg(
        total_days=("is_buy_day", "count"),
        buy_days=("is_buy_day", "sum"),
        avg_buy_count=("buy_count", "mean"),
        avg_buy_amount=("buy_amount_usd", "mean"),
    )
    dow_buy["buy_prob"] = (dow_buy["buy_days"] / dow_buy["total_days"] * 100).round(1)
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    dow_buy.index = dow_buy.index.map(day_names)

    # Win rate from round trips by buy weekday
    dow_rt = rt.groupby("buy_weekday_num").agg(
        trades=("pnl_pct", "count"),
        wins=("is_win", "sum"),
        avg_pnl=("pnl_pct", "mean"),
        median_pnl=("pnl_pct", "median"),
    )
    dow_rt["win_rate"] = (dow_rt["wins"] / dow_rt["trades"] * 100).round(1)
    dow_rt.index = dow_rt.index.map(day_names)

    # Merge into one table
    dow = dow_buy.join(dow_rt[["trades", "win_rate", "avg_pnl", "median_pnl"]])
    print(dow.to_string())

    best_buy_day = dow["buy_prob"].idxmax()
    best_win_day = dow["win_rate"].idxmax()
    print(f"\n  Highest buy probability: {best_buy_day} ({dow.loc[best_buy_day, 'buy_prob']}%)")
    print(f"  Highest win rate:        {best_win_day} ({dow.loc[best_win_day, 'win_rate']}%)")

    # Chi-square test: is day-of-week distribution of buy days non-uniform?
    observed = dlog.groupby("weekday_num")["is_buy_day"].sum().values
    expected_freq = np.full(len(observed), observed.mean())
    chi2, p_chi2 = stats.chisquare(observed, f_exp=expected_freq)
    print(f"\n  Chi-square test (day-of-week buy distribution):")
    print(f"    chi2 = {chi2:.4f}, p = {p_chi2:.4f}")
    print(f"    {'Significant (p<0.05): buy patterns differ by weekday' if p_chi2 < 0.05 else 'Not significant: no strong weekday preference'}")

    # -----------------------------------------------------------------------
    # 2. Week of Month
    # -----------------------------------------------------------------------
    sub_sep("2. Week of Month Analysis")

    dlog["day_of_month"] = dlog["date"].dt.day
    dlog["week_of_month"] = pd.cut(
        dlog["day_of_month"],
        bins=[0, 7, 14, 21, 32],
        labels=["W1 (1-7)", "W2 (8-14)", "W3 (15-21)", "W4 (22+)"],
    )

    wom = dlog.groupby("week_of_month", observed=True).agg(
        total_days=("is_buy_day", "count"),
        buy_days=("is_buy_day", "sum"),
        total_buy_count=("buy_count", "sum"),
        avg_buy_amount=("buy_amount_usd", "mean"),
    )
    wom["buy_prob"] = (wom["buy_days"] / wom["total_days"] * 100).round(1)
    wom["buy_share"] = (wom["total_buy_count"] / wom["total_buy_count"].sum() * 100).round(1)
    print(wom.to_string())

    # -----------------------------------------------------------------------
    # 3. Month-over-Month Trend
    # -----------------------------------------------------------------------
    sub_sep("3. Month-over-Month Trend")

    dlog["year_month"] = dlog["date"].dt.to_period("M")
    rt["buy_year_month"] = rt["buy_date"].dt.to_period("M")

    monthly_activity = dlog.groupby("year_month").agg(
        trading_days=("date", "count"),
        buy_days=("is_buy_day", "sum"),
        total_buys=("buy_count", "sum"),
        total_buy_amt=("buy_amount_usd", "sum"),
    )
    monthly_activity["buy_rate"] = (monthly_activity["buy_days"] / monthly_activity["trading_days"] * 100).round(1)

    monthly_rt = rt.groupby("buy_year_month").agg(
        trades=("pnl_pct", "count"),
        wins=("is_win", "sum"),
        avg_pnl=("pnl_pct", "mean"),
    )
    monthly_rt["win_rate"] = (monthly_rt["wins"] / monthly_rt["trades"] * 100).round(1)

    monthly = monthly_activity.join(monthly_rt[["trades", "win_rate", "avg_pnl"]])
    print(monthly.to_string())

    # Regime shift analysis: front-half vs back-half
    all_months = sorted(dlog["year_month"].unique())
    mid = len(all_months) // 2
    front_months = set(all_months[:mid])
    back_months = set(all_months[mid:])

    front_mask = dlog["year_month"].isin(front_months)
    back_mask = dlog["year_month"].isin(back_months)

    front_buys = dlog.loc[front_mask, "buy_count"].mean()
    back_buys = dlog.loc[back_mask, "buy_count"].mean()
    front_amt = dlog.loc[front_mask, "buy_amount_usd"].mean()
    back_amt = dlog.loc[back_mask, "buy_amount_usd"].mean()

    front_rt = rt[rt["buy_year_month"].isin(front_months)]
    back_rt = rt[rt["buy_year_month"].isin(back_months)]

    print(f"\n  Regime Shift: Front-half ({all_months[0]}~{all_months[mid-1]}) "
          f"vs Back-half ({all_months[mid]}~{all_months[-1]})")
    print(f"    Front avg daily buys: {front_buys:.2f}  |  Back: {back_buys:.2f}  "
          f"(change: {(back_buys - front_buys) / front_buys * 100:+.1f}%)")
    print(f"    Front avg daily $:    ${front_amt:.0f}  |  Back: ${back_amt:.0f}  "
          f"(change: {(back_amt - front_amt) / front_amt * 100:+.1f}%)")
    if len(front_rt) > 0 and len(back_rt) > 0:
        f_wr = front_rt["is_win"].mean() * 100
        b_wr = back_rt["is_win"].mean() * 100
        print(f"    Front win rate:       {f_wr:.1f}%  |  Back: {b_wr:.1f}%")

    # -----------------------------------------------------------------------
    # 4. Day After Big SPY Moves
    # -----------------------------------------------------------------------
    sub_sep("4. Day After Big SPY Moves (±1%)")

    dlog_sorted = dlog.sort_values("date").reset_index(drop=True)
    dlog_sorted["prev_SPY_pct"] = dlog_sorted["SPY_pct"].shift(1)

    # After SPY drop >= 1%
    after_drop = dlog_sorted[dlog_sorted["prev_SPY_pct"] <= -1.0]
    after_rise = dlog_sorted[dlog_sorted["prev_SPY_pct"] >= 1.0]

    if len(after_drop) > 0:
        drop_buy_prob = after_drop["is_buy_day"].mean() * 100
        drop_avg_buys = after_drop["buy_count"].mean()
        print(f"  After SPY drop >= 1% ({len(after_drop)} days):")
        print(f"    Buy probability: {drop_buy_prob:.1f}%  |  Avg buys: {drop_avg_buys:.2f}")
    if len(after_rise) > 0:
        rise_buy_prob = after_rise["is_buy_day"].mean() * 100
        rise_avg_buys = after_rise["buy_count"].mean()
        print(f"  After SPY rise >= 1% ({len(after_rise)} days):")
        print(f"    Buy probability: {rise_buy_prob:.1f}%  |  Avg buys: {rise_avg_buys:.2f}")

    # Baseline for comparison
    baseline_buy_prob = dlog_sorted["is_buy_day"].mean() * 100
    print(f"  Baseline buy probability (all days): {baseline_buy_prob:.1f}%")

    # 2-day SPY decline
    dlog_sorted["prev_SPY_pct_2"] = dlog_sorted["SPY_pct"].shift(2)
    two_day_decline = dlog_sorted[
        (dlog_sorted["prev_SPY_pct"] < 0) & (dlog_sorted["prev_SPY_pct_2"] < 0)
    ]
    if len(two_day_decline) > 0:
        td_prob = two_day_decline["is_buy_day"].mean() * 100
        td_buys = two_day_decline["buy_count"].mean()
        print(f"\n  After 2 consecutive SPY declines ({len(two_day_decline)} days):")
        print(f"    Buy probability: {td_prob:.1f}%  |  Avg buys: {td_buys:.2f}")

    # Build calendar effects summary for CSV
    calendar_rows = []
    for day_name, row in dow.iterrows():
        calendar_rows.append({
            "category": "day_of_week",
            "label": day_name,
            "buy_prob_pct": row["buy_prob"],
            "win_rate_pct": row.get("win_rate", np.nan),
            "avg_pnl_pct": row.get("avg_pnl", np.nan),
            "sample_size": row.get("trades", 0),
        })
    for week_label, row in wom.iterrows():
        calendar_rows.append({
            "category": "week_of_month",
            "label": str(week_label),
            "buy_prob_pct": row["buy_prob"],
            "win_rate_pct": np.nan,
            "avg_pnl_pct": np.nan,
            "sample_size": row["total_days"],
        })

    calendar_df = pd.DataFrame(calendar_rows)
    return calendar_df


# ===========================================================================
# PART 2: Cross-Correlation & Lead-Lag
# ===========================================================================

def adf_test_manual(series: pd.Series, max_lag: int = 5):
    """
    Manual Augmented Dickey-Fuller test using OLS regression via numpy.

    Tests H0: series has a unit root (non-stationary)
    vs   H1: series is stationary

    Returns: (adf_stat, approximate_p_value, used_lag)
    """
    y = series.dropna().values
    n = len(y)
    if n < max_lag + 5:
        return np.nan, np.nan, 0

    dy = np.diff(y)  # first difference
    y_lag = y[:-1]   # y_{t-1}

    # Determine lag using simple BIC approximation
    best_bic = np.inf
    best_lag = 0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            # dy_t = alpha + gamma * y_{t-1} + e_t
            start = 0
            Y = dy[start:]
            X = np.column_stack([np.ones(len(Y)), y_lag[start:start + len(Y)]])
        else:
            start = lag
            Y = dy[start:]
            cols = [np.ones(len(Y)), y_lag[start:start + len(Y)]]
            for j in range(1, lag + 1):
                cols.append(dy[start - j:start - j + len(Y)])
            X = np.column_stack(cols)

        if len(Y) < X.shape[1] + 2:
            continue
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            resid = Y - X @ beta
            n_obs = len(Y)
            k = X.shape[1]
            sse = np.sum(resid ** 2)
            bic = n_obs * np.log(sse / n_obs) + k * np.log(n_obs)
            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except np.linalg.LinAlgError:
            continue

    # Run final regression with chosen lag
    lag = best_lag
    if lag == 0:
        start = 0
        Y = dy[start:]
        X = np.column_stack([np.ones(len(Y)), y_lag[start:start + len(Y)]])
    else:
        start = lag
        Y = dy[start:]
        cols = [np.ones(len(Y)), y_lag[start:start + len(Y)]]
        for j in range(1, lag + 1):
            cols.append(dy[start - j:start - j + len(Y)])
        X = np.column_stack(cols)

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ beta
        n_obs = len(Y)
        sse = np.sum(resid ** 2)
        se2 = sse / (n_obs - X.shape[1])
        # Covariance of beta
        XtX_inv = np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(se2 * XtX_inv[1, 1])
        adf_stat = beta[1] / se_gamma

        # Approximate p-value using MacKinnon critical values for constant, no trend
        # Critical values: 1%: -3.43, 5%: -2.86, 10%: -2.57
        if adf_stat < -3.43:
            p_approx = 0.005
        elif adf_stat < -2.86:
            p_approx = 0.03
        elif adf_stat < -2.57:
            p_approx = 0.07
        elif adf_stat < -1.94:
            p_approx = 0.15
        elif adf_stat < -1.62:
            p_approx = 0.30
        else:
            p_approx = 0.50 + min(0.49, max(0, (adf_stat + 1.62) * 0.1))

        return adf_stat, p_approx, lag
    except np.linalg.LinAlgError:
        return np.nan, np.nan, lag


def half_life_ols(spread: pd.Series) -> float:
    """Compute half-life of mean reversion via OLS on spread."""
    spread = spread.dropna()
    if len(spread) < 10:
        return np.nan
    y = spread.values
    dy = np.diff(y)
    y_lag = y[:-1]

    # dy = a + b * y_lag + eps
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        b = beta[1]
        if b >= 0:
            return np.nan  # not mean-reverting
        hl = -np.log(2) / b
        return hl
    except np.linalg.LinAlgError:
        return np.nan


def part2_cross_correlation(market: pd.DataFrame) -> tuple:
    """Compute return correlations, lead-lag, and pair spread analysis."""
    sep("PART 2: CROSS-CORRELATION & LEAD-LAG")

    focus_tickers = ["MSTU", "CONL", "ROBN", "AMDL", "NVDL", "BITU", "SPY", "QQQ", "GLD"]

    # Compute daily returns from Close prices
    closes = pd.DataFrame(
        {t: market[(t, "Close")] for t in focus_tickers if (t, "Close") in market.columns},
        index=market.index,
    )
    returns = closes.pct_change().dropna()
    print(f"  Computing returns for {list(returns.columns)}, {len(returns)} trading days\n")

    # -----------------------------------------------------------------------
    # 5. Return Correlation Matrix
    # -----------------------------------------------------------------------
    sub_sep("5. Return Correlation Matrix (Pearson)")

    corr = returns.corr().round(4)
    print(corr.to_string())

    # Most and least correlated pairs
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 5 most correlated pairs:")
    for a, b, c in pairs[:5]:
        print(f"    {a:6s} - {b:6s}: {c:+.4f}")
    print(f"\n  Top 5 least correlated pairs:")
    for a, b, c in pairs[-5:]:
        print(f"    {a:6s} - {b:6s}: {c:+.4f}")

    low_corr_pairs = [p for p in pairs if abs(p[2]) < 0.3]
    if low_corr_pairs:
        print(f"\n  Diversification benefit (|corr| < 0.3): {len(low_corr_pairs)} pairs")
        for a, b, c in low_corr_pairs[:8]:
            print(f"    {a:6s} - {b:6s}: {c:+.4f}")

    # -----------------------------------------------------------------------
    # 6. Lead-Lag Analysis
    # -----------------------------------------------------------------------
    sub_sep("6. Lead-Lag Analysis (cross-correlation at lags -2 to +2)")

    lead_lag_rows = []
    leader_tickers = ["SPY", "QQQ", "GLD", "BITU"]
    follower_tickers = ["MSTU", "CONL", "ROBN", "AMDL", "NVDL"]
    lags = [-2, -1, 0, 1, 2]

    for leader in leader_tickers:
        if leader not in returns.columns:
            continue
        for follower in follower_tickers:
            if follower not in returns.columns or leader == follower:
                continue
            row = {"leader": leader, "follower": follower}
            for lag in lags:
                if lag == 0:
                    c = returns[leader].corr(returns[follower])
                elif lag > 0:
                    # leader leads: leader at t correlated with follower at t+lag
                    c = returns[leader].iloc[:-lag].reset_index(drop=True).corr(
                        returns[follower].iloc[lag:].reset_index(drop=True)
                    )
                else:
                    # negative lag: follower leads
                    alag = abs(lag)
                    c = returns[leader].iloc[alag:].reset_index(drop=True).corr(
                        returns[follower].iloc[:-alag].reset_index(drop=True)
                    )
                row[f"lag_{lag}"] = round(c, 4)
            # Find best leading lag
            lag_vals = {l: row[f"lag_{l}"] for l in lags}
            best_lag = max(lag_vals, key=lambda l: abs(lag_vals[l]))
            row["best_lag"] = best_lag
            row["best_corr"] = lag_vals[best_lag]
            lead_lag_rows.append(row)

    lead_lag_df = pd.DataFrame(lead_lag_rows)
    print(lead_lag_df.to_string(index=False))

    # Highlight actionable pairs
    actionable = lead_lag_df[
        (lead_lag_df["best_lag"] != 0) &
        (lead_lag_df["best_corr"].abs() > 0.3)
    ]
    if len(actionable) > 0:
        print(f"\n  Actionable lead-lag pairs (|corr|>0.3 at non-zero lag):")
        for _, r in actionable.iterrows():
            direction = "leads" if r["best_lag"] > 0 else "follows"
            print(f"    {r['leader']} {direction} {r['follower']} by "
                  f"{abs(int(r['best_lag']))} day(s), corr={r['best_corr']:+.4f}")
    else:
        print("\n  No strongly actionable lead-lag pairs found (all best at lag 0).")

    # -----------------------------------------------------------------------
    # 7. Pair Spread Stationarity
    # -----------------------------------------------------------------------
    sub_sep("7. Pair Spread Stationarity (Twin Pairs)")

    twin_pairs = [
        ("BITU", "MSTU", "Crypto twins"),
        ("ROBN", "CONL", "Fintech/Coinbase twins"),
        ("NVDL", "AMDL", "Semi twins"),
    ]

    for lead, follow, label in twin_pairs:
        if lead not in returns.columns or follow not in returns.columns:
            print(f"  {label} ({lead}-{follow}): data not available")
            continue

        spread = returns[lead] - returns[follow]
        spread_mean = spread.mean()
        spread_std = spread.std()

        adf_stat, adf_p, used_lag = adf_test_manual(spread)
        hl = half_life_ols(spread)

        print(f"\n  {label} ({lead} - {follow}):")
        print(f"    Spread mean:    {spread_mean:.6f}")
        print(f"    Spread std:     {spread_std:.6f}")
        print(f"    ADF statistic:  {adf_stat:.4f}  (approx p={adf_p:.3f}, lag={used_lag})")
        if adf_p < 0.05:
            print(f"    => STATIONARY: spread mean-reverts (reject unit root at 5%)")
        elif adf_p < 0.10:
            print(f"    => WEAKLY STATIONARY: marginal evidence of mean-reversion")
        else:
            print(f"    => NON-STATIONARY: spread does NOT reliably mean-revert")
        print(f"    Half-life:      {hl:.1f} days" if not np.isnan(hl) else "    Half-life:      N/A (no mean reversion)")

    return corr, lead_lag_df


# ===========================================================================
# PART 3: Conditional Returns
# ===========================================================================

def part3_conditional_returns(dlog: pd.DataFrame, rt: pd.DataFrame,
                              trades: pd.DataFrame) -> None:
    """Analyze returns conditioned on market regime."""
    sep("PART 3: CONDITIONAL RETURNS")

    dlog = dlog.copy().sort_values("date").reset_index(drop=True)
    rt = rt.copy()
    trades = trades.copy()

    dlog["is_buy_day"] = dlog["action"].isin(["BUY", "BUY+SELL"])

    # Merge round trips with buy-day context
    rt["buy_weekday_num"] = rt["buy_date"].dt.weekday
    rt["is_win"] = rt["pnl_pct"] > 0

    # -----------------------------------------------------------------------
    # 8. SPY Regime + Entry Returns
    # -----------------------------------------------------------------------
    sub_sep("8. SPY Streak Regime & Entry Outcomes")

    # SPY consecutive streak: count consecutive up/down days
    spy_pct = dlog[["date", "SPY_pct"]].dropna()
    spy_sign = np.sign(spy_pct["SPY_pct"])

    # Compute streak length
    streak = np.zeros(len(spy_sign), dtype=int)
    for i in range(len(spy_sign)):
        if i == 0:
            streak[i] = int(spy_sign.iloc[i])
        else:
            if spy_sign.iloc[i] == spy_sign.iloc[i - 1] and spy_sign.iloc[i] != 0:
                streak[i] = streak[i - 1] + int(spy_sign.iloc[i])
            else:
                streak[i] = int(spy_sign.iloc[i])
    dlog["spy_streak"] = streak

    # 3-day streak up / down on the day BEFORE a buy
    dlog["prev_streak"] = dlog["spy_streak"].shift(1)

    for streak_val, label in [(3, "3+ day SPY UP streak"), (-3, "3+ day SPY DOWN streak")]:
        if streak_val > 0:
            mask = dlog["prev_streak"] >= streak_val
        else:
            mask = dlog["prev_streak"] <= streak_val

        subset = dlog[mask]
        if len(subset) == 0:
            print(f"\n  {label}: no occurrences found")
            continue

        buy_prob = subset["is_buy_day"].mean() * 100
        n_days = len(subset)

        # Find round trips initiated on these days
        subset_dates = set(subset["date"].dt.date)
        rt_matched = rt[rt["buy_date"].dt.date.isin(subset_dates)]
        if len(rt_matched) > 0:
            wr = rt_matched["is_win"].mean() * 100
            avg_pnl = rt_matched["pnl_pct"].mean()
            n_trades = len(rt_matched)
        else:
            wr = avg_pnl = n_trades = np.nan

        print(f"\n  {label} ({n_days} occurrences):")
        print(f"    Buy probability: {buy_prob:.1f}%")
        if not np.isnan(wr):
            print(f"    Win rate of entries: {wr:.1f}%  (n={n_trades})")
            print(f"    Avg P&L: {avg_pnl:+.2f}%")
        else:
            print(f"    No round-trip data for these entry days")

    # Also check using existing SPY_3d_streak column
    if "SPY_3d_streak" in dlog.columns:
        print(f"\n  Using pre-computed SPY_3d_streak column:")
        for streak_val in [0, 1, 2, 3]:
            mask = dlog["SPY_3d_streak"] == streak_val
            subset = dlog[mask]
            buy_prob = subset["is_buy_day"].mean() * 100
            print(f"    SPY_3d_streak={streak_val}: {len(subset)} days, buy_prob={buy_prob:.1f}%")

    # -----------------------------------------------------------------------
    # 9. GLD-BTC Divergence
    # -----------------------------------------------------------------------
    sub_sep("9. GLD-BTC Divergence")

    # Use GLD_pct and poly_btc_up (probability, >0.5 = bullish)
    has_both = dlog["GLD_pct"].notna() & dlog["poly_btc_up"].notna()
    div_data = dlog[has_both].copy()

    if len(div_data) > 0:
        # GLD up + BTC down (risk-off signal)
        risk_off = div_data[(div_data["GLD_pct"] > 0) & (div_data["poly_btc_up"] < 0.5)]
        # GLD down + BTC up (risk-on signal)
        risk_on = div_data[(div_data["GLD_pct"] < 0) & (div_data["poly_btc_up"] > 0.5)]

        for label, subset in [("GLD UP + BTC DOWN (risk-off)", risk_off),
                              ("GLD DOWN + BTC UP (risk-on)", risk_on)]:
            if len(subset) == 0:
                print(f"\n  {label}: no occurrences")
                continue
            buy_prob = subset["is_buy_day"].mean() * 100
            avg_buys = subset["buy_count"].mean()

            subset_dates = set(subset["date"].dt.date)
            rt_matched = rt[rt["buy_date"].dt.date.isin(subset_dates)]
            wr = rt_matched["is_win"].mean() * 100 if len(rt_matched) > 0 else np.nan
            avg_pnl = rt_matched["pnl_pct"].mean() if len(rt_matched) > 0 else np.nan
            n_trades = len(rt_matched)

            print(f"\n  {label} ({len(subset)} days):")
            print(f"    Buy probability: {buy_prob:.1f}%  |  Avg buys: {avg_buys:.2f}")
            if not np.isnan(wr):
                print(f"    Win rate: {wr:.1f}%  |  Avg P&L: {avg_pnl:+.2f}%  (n={n_trades})")
    else:
        print("  Insufficient GLD + BTC data for divergence analysis.")

    # Also analyze using GLD_pct direction as proxy for BTC (when poly data missing)
    # Use SPY as risk proxy instead
    sub_sep("9b. GLD-SPY Divergence (alternative, more data)")
    gld_spy = dlog[dlog["GLD_pct"].notna() & dlog["SPY_pct"].notna()].copy()
    if len(gld_spy) > 0:
        # GLD up + SPY down (flight to safety)
        flight_safety = gld_spy[(gld_spy["GLD_pct"] > 0) & (gld_spy["SPY_pct"] < 0)]
        # GLD down + SPY up (risk-on)
        risk_on_alt = gld_spy[(gld_spy["GLD_pct"] < 0) & (gld_spy["SPY_pct"] > 0)]

        for label, subset in [("GLD UP + SPY DOWN (flight to safety)", flight_safety),
                              ("GLD DOWN + SPY UP (risk-on)", risk_on_alt)]:
            buy_prob = subset["is_buy_day"].mean() * 100 if len(subset) > 0 else np.nan
            n = len(subset)
            subset_dates = set(subset["date"].dt.date)
            rt_m = rt[rt["buy_date"].dt.date.isin(subset_dates)]
            wr = rt_m["is_win"].mean() * 100 if len(rt_m) > 0 else np.nan
            avg_pnl = rt_m["pnl_pct"].mean() if len(rt_m) > 0 else np.nan

            print(f"\n  {label} ({n} days):")
            print(f"    Buy probability: {buy_prob:.1f}%")
            if not np.isnan(wr):
                print(f"    Win rate: {wr:.1f}%  |  Avg P&L: {avg_pnl:+.2f}%  (n={len(rt_m)})")

    # -----------------------------------------------------------------------
    # 10. Sector Rotation
    # -----------------------------------------------------------------------
    sub_sep("10. Sector Rotation (Crypto vs Semi co-movement)")

    # Classify tickers by sector
    crypto_tickers = {"MSTU", "MSTX", "BITU", "CONL", "ETHU", "WGMI", "BTOG", "XXRP", "CONY"}
    semi_tickers = {"NVDL", "AMDL", "SOXL", "TSMX", "SOLT", "IONQ", "IREN"}

    # For each buy day, determine which sectors were bought
    buy_days = trades[trades["action"] == "구매"].copy()
    buy_days["is_crypto"] = buy_days["yf_ticker"].isin(crypto_tickers)
    buy_days["is_semi"] = buy_days["yf_ticker"].isin(semi_tickers)

    day_sectors = buy_days.groupby("date").agg(
        crypto_bought=("is_crypto", "any"),
        semi_bought=("is_semi", "any"),
        crypto_count=("is_crypto", "sum"),
        semi_count=("is_semi", "sum"),
        total_count=("yf_ticker", "count"),
    )

    n_days = len(day_sectors)
    both = day_sectors["crypto_bought"] & day_sectors["semi_bought"]
    crypto_only = day_sectors["crypto_bought"] & ~day_sectors["semi_bought"]
    semi_only = ~day_sectors["crypto_bought"] & day_sectors["semi_bought"]
    neither = ~day_sectors["crypto_bought"] & ~day_sectors["semi_bought"]

    print(f"\n  Buy-day sector breakdown ({n_days} buy days):")
    print(f"    Both crypto + semi: {both.sum():4d} ({both.mean() * 100:.1f}%)")
    print(f"    Crypto only:        {crypto_only.sum():4d} ({crypto_only.mean() * 100:.1f}%)")
    print(f"    Semi only:          {semi_only.sum():4d} ({semi_only.mean() * 100:.1f}%)")
    print(f"    Neither:            {neither.sum():4d} ({neither.mean() * 100:.1f}%)")

    # Contingency table + chi-square
    contingency = pd.crosstab(day_sectors["crypto_bought"], day_sectors["semi_bought"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square test for sector co-movement:")
    print(f"    chi2={chi2:.4f}, p={p:.4f}, dof={dof}")
    if p < 0.05:
        print(f"    => SIGNIFICANT: crypto and semi buying are NOT independent")
    else:
        print(f"    => Not significant: no evidence of sector co-movement")

    # Correlation of daily crypto vs semi buy counts
    if n_days > 5:
        corr_sector = day_sectors["crypto_count"].corr(day_sectors["semi_count"])
        print(f"\n  Daily crypto buy count vs semi buy count correlation: {corr_sector:.4f}")

    # Win rate by sector combination
    rt_sector = rt.copy()
    rt_sector["is_crypto"] = rt_sector["ticker"].isin(crypto_tickers)
    rt_sector["is_semi"] = rt_sector["ticker"].isin(semi_tickers)

    for label, mask in [("Crypto tickers", rt_sector["is_crypto"]),
                        ("Semi tickers", rt_sector["is_semi"]),
                        ("Other tickers", ~rt_sector["is_crypto"] & ~rt_sector["is_semi"])]:
        subset = rt_sector[mask]
        if len(subset) > 0:
            wr = subset["is_win"].mean() * 100
            avg_pnl = subset["pnl_pct"].mean()
            print(f"\n    {label}: {len(subset)} trades, "
                  f"win_rate={wr:.1f}%, avg_pnl={avg_pnl:+.2f}%")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 80)
    print("  CALENDAR EFFECTS, CROSS-CORRELATION & LEAD-LAG ANALYSIS")
    print("=" * 80)

    market, dlog, trades, rt = load_data()

    # Part 1
    calendar_df = part1_calendar_effects(dlog, trades, rt)

    # Part 2
    corr_matrix, lead_lag_df = part2_cross_correlation(market)

    # Part 3
    part3_conditional_returns(dlog, rt, trades)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    sep("SAVING OUTPUTS")

    out_corr = RESULTS / "correlation_matrix.csv"
    corr_matrix.to_csv(out_corr)
    print(f"  Saved: {out_corr}")

    out_cal = RESULTS / "calendar_effects.csv"
    calendar_df.to_csv(out_cal, index=False)
    print(f"  Saved: {out_cal}")

    out_ll = RESULTS / "lead_lag_report.csv"
    lead_lag_df.to_csv(out_ll, index=False)
    print(f"  Saved: {out_ll}")

    sep("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Decision Log 구축 — 규칙 복원 실험 Step 1
==========================================
매 거래일마다 시장 상태 + Polymarket 확률 + 실제 행동을 결합하여
하나의 분석 테이블을 만든다.

출력:
  data/results/analysis/decision_log.csv       (일별 요약)
  data/results/analysis/decision_log_trades.csv (개별 거래 + 상태)

사용법:
  pyenv shell ptj_stock_lab && python experiments/build_decision_log.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent
HIST_CSV = BASE / "history" / "2025" / "거래내역서_20250218_20260217_1.csv"
MARKET_DAILY = BASE / "data" / "market" / "daily" / "market_daily.parquet"
POLY_DIR = BASE / "data" / "polymarket"
META_DIR = BASE / "data" / "meta"
OUT_DIR = BASE / "data" / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 1. 종목 매핑 로드
# ════════════════════════════════════════════════════════════════
def load_ticker_mapping() -> dict[str, dict]:
    """한글 종목명 → {yf, cat, ...} 매핑."""
    path = META_DIR / "ticker_mapping.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════
# 2. 거래내역 로드
# ════════════════════════════════════════════════════════════════
def load_trades(ticker_map: dict) -> pd.DataFrame:
    """거래내역 CSV를 로드하고 yf 티커를 매핑한다."""
    df = pd.read_csv(HIST_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["거래일자"], format="%Y.%m.%d").dt.date

    # 숫자 정리
    for col in ["거래수량", "거래대금_달러", "단가_달러", "수수료_달러", "제세금_달러"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)

    # yf 티커 매핑
    name_to_yf = {}
    name_to_cat = {}
    for name, info in ticker_map.items():
        name_to_yf[name] = info["yf"]
        name_to_cat[name] = info.get("cat", "기타")

    df["yf_ticker"] = df["종목명"].map(name_to_yf)
    df["category"] = df["종목명"].map(name_to_cat)

    unmapped = df[df["yf_ticker"].isna()]["종목명"].unique()
    if len(unmapped) > 0:
        print(f"  [WARN] 매핑 실패 종목: {unmapped}")

    print(f"  거래내역: {len(df)}건, {df['date'].nunique()}일")
    return df


# ════════════════════════════════════════════════════════════════
# 3. 시장 데이터 로드 (일봉)
# ════════════════════════════════════════════════════════════════
def load_market_daily() -> pd.DataFrame:
    """market_daily.parquet → long format + 전일 대비 변화율."""
    df = pd.read_parquet(MARKET_DAILY)

    # MultiIndex (Ticker, OHLCV) → long
    df_long = df.stack(level=0, future_stack=True).reset_index()
    col_map = {}
    for c in df_long.columns:
        cl = c.lower()
        if cl == "date":
            col_map[c] = "date"
        elif cl in ("ticker", "symbol", "level_1"):
            col_map[c] = "symbol"
        elif cl in ("open", "high", "low", "close", "volume"):
            col_map[c] = cl
        else:
            col_map[c] = c
    df_long = df_long.rename(columns=col_map)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    df_long = df_long.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 전일 대비 변화율
    df_long["prev_close"] = df_long.groupby("symbol")["close"].shift(1)
    df_long["pct_change"] = (
        (df_long["close"] / df_long["prev_close"] - 1) * 100
    ).round(4)

    print(f"  시장 데이터: {df_long['symbol'].nunique()}종목, "
          f"{df_long['date'].min()} ~ {df_long['date'].max()}")
    return df_long


# ════════════════════════════════════════════════════════════════
# 4. Polymarket 데이터 로드
# ════════════════════════════════════════════════════════════════
def _extract_prob(indicator: dict | None, outcome_key: str) -> float:
    """지표에서 확률 추출. 데이터 없으면 NaN."""
    if indicator is None or "error" in indicator:
        return np.nan

    # final_prices 먼저
    final = indicator.get("final_prices", {})
    if outcome_key in final:
        try:
            return float(final[outcome_key])
        except (ValueError, TypeError):
            pass

    # time series 마지막 값
    markets = indicator.get("markets", [])
    if markets:
        outcomes = markets[0].get("outcomes", {})
        series = outcomes.get(outcome_key, [])
        if series:
            last = series[-1]
            if isinstance(last, dict):
                return float(last.get("p", np.nan))
            elif isinstance(last, (list, tuple)) and len(last) >= 2:
                return float(last[1])

    return np.nan


def load_polymarket() -> dict[date, dict]:
    """Polymarket JSON → {date: {btc_up, ndx_up, eth_up, rate_hike}}."""
    result: dict[date, dict] = {}

    json_files = sorted(
        f for f in POLY_DIR.rglob("*.json")
        if "_cache" not in f.parts
    )

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        date_str = data.get("date", fp.stem.split("_")[0])
        try:
            dt = date.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        indicators = data.get("indicators", {})

        entry = {
            "btc_up": _extract_prob(indicators.get("btc_up_down"), "Up"),
            "btc_down": _extract_prob(indicators.get("btc_up_down"), "Down"),
            "ndx_up": _extract_prob(indicators.get("ndx_up_down"), "Up"),
            "eth_up": _extract_prob(indicators.get("eth_above_today"), "Yes"),
            "rate_hike": _extract_prob(indicators.get("fed_decision"), "Yes"),
        }

        # 기존보다 나은 데이터면 업데이트
        if dt not in result:
            result[dt] = entry
        else:
            for k, v in entry.items():
                if not np.isnan(v) and (np.isnan(result[dt].get(k, np.nan))):
                    result[dt][k] = v

    print(f"  Polymarket: {len(result)}일 ({min(result.keys())} ~ {max(result.keys())})")
    return result


# ════════════════════════════════════════════════════════════════
# 5. Decision Log 구축
# ════════════════════════════════════════════════════════════════
def build_decision_log(
    trades: pd.DataFrame,
    market: pd.DataFrame,
    poly: dict[date, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    매 거래일마다 시장 상태 + 행동을 결합.

    Returns
    -------
    daily_log : DataFrame  — 일별 요약 (1 row per trading day)
    trade_log : DataFrame  — 개별 거래 + 시장 상태 (1 row per trade)
    """
    # ── 거래일 범위 결정 ──
    trade_dates = set(trades["date"].unique())
    market_dates = sorted(market["date"].unique())
    min_date = min(min(trade_dates), min(market_dates))
    max_date = max(max(trade_dates), max(market_dates))

    # 시장 데이터를 pivot: {(date, symbol): row}
    market_idx = {}
    for _, row in market.iterrows():
        market_idx[(row["date"], row["symbol"])] = row

    # 시장 지표 종목들
    INDEX_TICKERS = ["SPY", "QQQ", "GLD"]

    # ── 일별 요약 ──
    daily_rows = []
    for dt in market_dates:
        if dt < min(trade_dates) or dt > max(trade_dates):
            continue

        row = {"date": dt}

        # 시장 지표
        for sym in INDEX_TICKERS:
            key = (dt, sym)
            if key in market_idx:
                m = market_idx[key]
                row[f"{sym}_close"] = round(m["close"], 2) if pd.notna(m["close"]) else np.nan
                row[f"{sym}_pct"] = round(m["pct_change"], 4) if pd.notna(m["pct_change"]) else np.nan
            else:
                row[f"{sym}_close"] = np.nan
                row[f"{sym}_pct"] = np.nan

        # Polymarket
        poly_day = poly.get(dt, {})
        row["poly_btc_up"] = poly_day.get("btc_up", np.nan)
        row["poly_ndx_up"] = poly_day.get("ndx_up", np.nan)
        row["poly_eth_up"] = poly_day.get("eth_up", np.nan)
        row["poly_rate_hike"] = poly_day.get("rate_hike", np.nan)

        # 행동 요약
        day_trades = trades[trades["date"] == dt]
        buys = day_trades[day_trades["거래구분"] == "구매"]
        sells = day_trades[day_trades["거래구분"] == "판매"]

        row["action"] = "HOLD"
        if len(buys) > 0 and len(sells) > 0:
            row["action"] = "BUY+SELL"
        elif len(buys) > 0:
            row["action"] = "BUY"
        elif len(sells) > 0:
            row["action"] = "SELL"

        row["buy_count"] = len(buys)
        row["sell_count"] = len(sells)
        row["buy_amount_usd"] = round(buys["거래대금_달러"].sum(), 2)
        row["sell_amount_usd"] = round(sells["거래대금_달러"].sum(), 2)
        row["buy_tickers"] = ",".join(sorted(buys["yf_ticker"].dropna().unique()))
        row["sell_tickers"] = ",".join(sorted(sells["yf_ticker"].dropna().unique()))
        row["unique_tickers_bought"] = buys["yf_ticker"].dropna().nunique()

        daily_rows.append(row)

    daily_log = pd.DataFrame(daily_rows)

    # ── 2차 변수 (맥락) ──
    if len(daily_log) > 0:
        # 최근 N일 연속 하락
        daily_log["SPY_3d_streak"] = _streak(daily_log["SPY_pct"], negative=True)
        daily_log["QQQ_3d_streak"] = _streak(daily_log["QQQ_pct"], negative=True)

        # 최근 3일 매수 횟수
        daily_log["buy_count_3d"] = (
            daily_log["buy_count"].rolling(3, min_periods=1).sum()
        )

        # Polymarket 변화 방향
        daily_log["poly_btc_up_delta"] = daily_log["poly_btc_up"].diff()

        # 5일 변동성 (SPY 기준)
        daily_log["SPY_vol_5d"] = (
            daily_log["SPY_pct"].rolling(5, min_periods=2).std().round(4)
        )

        # 복합 지표: 공포 (GLD↑ + QQQ↓ + SPY↓)
        daily_log["fear_signal"] = (
            (daily_log["GLD_pct"] > 0)
            & (daily_log["QQQ_pct"] < 0)
            & (daily_log["SPY_pct"] < 0)
        ).astype(int)

        # 복합 지표: 확신 (BTC up > 60% + NDX up > 55%)
        daily_log["confidence_signal"] = (
            (daily_log["poly_btc_up"] > 0.60)
            & (daily_log["poly_ndx_up"] > 0.55)
        ).astype(int)

    # ── 개별 거래 로그 ──
    trade_rows = []
    for _, t in trades.iterrows():
        dt = t["date"]
        ticker = t["yf_ticker"]

        row = {
            "date": dt,
            "action": t["거래구분"],
            "종목명": t["종목명"],
            "yf_ticker": ticker,
            "category": t["category"],
            "qty": t["거래수량"],
            "price_usd": t["단가_달러"],
            "amount_usd": t["거래대금_달러"],
            "fee_usd": t["수수료_달러"],
        }

        # 시장 지표
        for sym in INDEX_TICKERS:
            key = (dt, sym)
            if key in market_idx:
                m = market_idx[key]
                row[f"{sym}_pct"] = round(m["pct_change"], 4) if pd.notna(m["pct_change"]) else np.nan
            else:
                row[f"{sym}_pct"] = np.nan

        # 매매 종목의 당일 등락
        if ticker and (dt, ticker) in market_idx:
            m = market_idx[(dt, ticker)]
            row["ticker_pct"] = round(m["pct_change"], 4) if pd.notna(m["pct_change"]) else np.nan
            row["ticker_close"] = round(m["close"], 2) if pd.notna(m["close"]) else np.nan
        else:
            row["ticker_pct"] = np.nan
            row["ticker_close"] = np.nan

        # Polymarket
        poly_day = poly.get(dt, {})
        row["poly_btc_up"] = poly_day.get("btc_up", np.nan)
        row["poly_ndx_up"] = poly_day.get("ndx_up", np.nan)

        trade_rows.append(row)

    trade_log = pd.DataFrame(trade_rows)

    return daily_log, trade_log


def _streak(series: pd.Series, negative: bool = True) -> pd.Series:
    """연속 음(양)일 카운트."""
    result = []
    count = 0
    for val in series:
        if pd.isna(val):
            result.append(0)
            count = 0
            continue
        if (negative and val < 0) or (not negative and val > 0):
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=series.index)


# ════════════════════════════════════════════════════════════════
# 6. 콘솔 출력
# ════════════════════════════════════════════════════════════════
def print_summary(daily: pd.DataFrame, trade: pd.DataFrame):
    """핵심 통계 콘솔 출력."""
    print()
    print("=" * 70)
    print("  Decision Log — 요약 통계")
    print("=" * 70)

    # 행동 분포
    print("\n[1] 일별 행동 분포")
    print("-" * 40)
    action_counts = daily["action"].value_counts()
    total_days = len(daily)
    for action, cnt in action_counts.items():
        print(f"  {action:<10s}: {cnt:>4d}일 ({cnt/total_days*100:.1f}%)")

    # 매수일 vs 관망일 시장 상태 비교
    print("\n[2] 매수일 vs 관망일 — 시장 상태 비교")
    print("-" * 60)
    buy_days = daily[daily["action"].isin(["BUY", "BUY+SELL"])]
    hold_days = daily[daily["action"] == "HOLD"]

    features = [
        "SPY_pct", "QQQ_pct", "GLD_pct",
        "poly_btc_up", "poly_ndx_up",
    ]
    header = f"  {'지표':<16s} {'매수일 평균':>12s} {'관망일 평균':>12s} {'차이':>10s}"
    print(header)
    print("  " + "-" * 52)

    for feat in features:
        buy_mean = buy_days[feat].mean()
        hold_mean = hold_days[feat].mean()
        diff = buy_mean - hold_mean
        if pd.isna(buy_mean) or pd.isna(hold_mean):
            continue
        fmt = ".4f" if "poly" in feat else ".3f"
        print(f"  {feat:<16s} {buy_mean:>12{fmt}} {hold_mean:>12{fmt}} {diff:>+10{fmt}}")

    # 매수일 공포/확신 시그널
    print("\n[3] 복합 시그널 빈도")
    print("-" * 40)
    for sig in ["fear_signal", "confidence_signal"]:
        if sig in daily.columns:
            buy_rate = buy_days[sig].mean() * 100 if len(buy_days) > 0 else 0
            hold_rate = hold_days[sig].mean() * 100 if len(hold_days) > 0 else 0
            print(f"  {sig:<22s}: 매수일 {buy_rate:.1f}% / 관망일 {hold_rate:.1f}%")

    # 종목별 거래 빈도
    print("\n[4] 종목별 매수 빈도 (상위 10)")
    print("-" * 40)
    buy_trades = trade[trade["action"] == "구매"]
    ticker_counts = buy_trades["yf_ticker"].value_counts().head(10)
    for ticker, cnt in ticker_counts.items():
        cat = buy_trades[buy_trades["yf_ticker"] == ticker]["category"].iloc[0]
        amt = buy_trades[buy_trades["yf_ticker"] == ticker]["amount_usd"].sum()
        print(f"  {ticker:<8s} ({cat:<12s}): {cnt:>4d}건  ${amt:>10,.2f}")

    # 매수 종목의 당일 등락 분포
    print("\n[5] 매수 시점 종목 등락 분포")
    print("-" * 40)
    ticker_pcts = buy_trades["ticker_pct"].dropna()
    if len(ticker_pcts) > 0:
        print(f"  평균: {ticker_pcts.mean():+.3f}%")
        print(f"  중앙값: {ticker_pcts.median():+.3f}%")
        print(f"  하락 중 매수: {(ticker_pcts < 0).sum()}건 ({(ticker_pcts < 0).mean()*100:.1f}%)")
        print(f"  상승 중 매수: {(ticker_pcts > 0).sum()}건 ({(ticker_pcts > 0).mean()*100:.1f}%)")

    # 요일별 패턴
    print("\n[6] 요일별 행동 분포")
    print("-" * 40)
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name()
    wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for wd in wd_order:
        wd_data = daily[daily["weekday"] == wd]
        if len(wd_data) == 0:
            continue
        buy_pct = wd_data["action"].isin(["BUY", "BUY+SELL"]).mean() * 100
        avg_buys = wd_data["buy_count"].mean()
        print(f"  {wd:<10s}: {len(wd_data):>3d}일  매수일 {buy_pct:.0f}%  평균 {avg_buys:.1f}건/일")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  Decision Log 구축 — 규칙 복원 실험 Step 1  ║")
    print("  ╚══════════════════════════════════════════════╝")

    # ── 데이터 로드 ──
    print("\n[1/4] 데이터 로드...")
    ticker_map = load_ticker_mapping()
    trades = load_trades(ticker_map)
    market = load_market_daily()
    poly = load_polymarket()

    # ── Decision Log 구축 ──
    print("\n[2/4] Decision Log 구축...")
    daily_log, trade_log = build_decision_log(trades, market, poly)
    print(f"  일별 로그: {len(daily_log)}일")
    print(f"  거래 로그: {len(trade_log)}건")

    # ── 요약 출력 ──
    print("\n[3/4] 요약 통계...")
    print_summary(daily_log, trade_log)

    # ── 파일 저장 ──
    print("\n[4/4] 파일 저장...")
    daily_path = OUT_DIR / "decision_log.csv"
    trade_path = OUT_DIR / "decision_log_trades.csv"
    daily_log.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trade_log.to_csv(trade_path, index=False, encoding="utf-8-sig")
    print(f"  {daily_path}")
    print(f"  {trade_path}")

    print("\n  완료!")
    print()


if __name__ == "__main__":
    main()

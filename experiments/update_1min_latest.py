#!/usr/bin/env python3
"""
최신 1분봉 데이터 업데이트
===========================
backtest_1min_3y.parquet 와 backtest_1min_v2.parquet 의
마지막 날짜 다음 거래일 ~ 오늘(또는 지정일)까지 수집해 append 한다.

실행:
    pyenv shell ptj_stock_lab && python experiments/update_1min_latest.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

import config
from fetchers.alpaca_fetcher import get_alpaca_client

MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)

TICKER_MAP         = config.ALPACA_TICKER_MAP
TICKER_REVERSE_MAP = config.ALPACA_TICKER_REVERSE


def _fetch_ticker(client, api_ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 월별 청킹 수집. SIP 실패 시 IEX 피드로 자동 재시도."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame(1, TimeFrameUnit.Minute)
    chunks = []
    cur = start

    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        label = cur.strftime("%Y-%m")
        chunk_df = pd.DataFrame()

        for feed in [None, "iex"]:   # SIP 먼저, 실패 시 IEX 재시도
            feed_label = f"feed={feed}" if feed else "SIP"
            try:
                kwargs = dict(
                    symbol_or_symbols=[api_ticker],
                    timeframe=tf,
                    start=datetime.combine(cur, time()),
                    end=datetime.combine(chunk_end + timedelta(days=1), time()),
                )
                if feed:
                    kwargs["feed"] = feed
                req = StockBarsRequest(**kwargs)
                bars = client.get_stock_bars(req)
                chunk_df = bars.df.reset_index()
                print(f"      [{label}] {len(chunk_df):,} rows ({feed_label})", flush=True)
                break
            except Exception as e:
                err = str(e)
                if feed is None and "recent SIP" in err:
                    print(f"      [{label}] SIP 제한 → IEX 재시도", flush=True)
                    continue
                print(f"      [{label}] SKIP — {e}", flush=True)
                break

        if not chunk_df.empty:
            chunks.append(chunk_df)
        cur = nxt

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _postprocess(df: pd.DataFrame, ptj_ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()
    if df.empty:
        return df
    df["symbol"] = ptj_ticker
    df["date"] = df["timestamp"].dt.date.astype(str)
    keep = ["symbol", "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap", "date"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    return df[keep].reset_index(drop=True)


def update_file(path: Path, fetch_start: date, fetch_end: date):
    print(f"\n=== {path.name} 업데이트 ===")
    existing = pd.read_parquet(path)
    existing["date"] = existing["date"].astype(str)
    symbols = sorted(existing["symbol"].unique())
    print(f"  기존 최신: {existing['date'].max()}  →  수집: {fetch_start} ~ {fetch_end}")
    print(f"  종목: {symbols}")

    client = get_alpaca_client()
    collected = []

    for ptj_ticker in symbols:
        api_ticker = TICKER_MAP.get(ptj_ticker, ptj_ticker)
        print(f"  [{ptj_ticker}] 수집 중...", flush=True)
        raw = _fetch_ticker(client, api_ticker, fetch_start, fetch_end)
        processed = _postprocess(raw, ptj_ticker)
        if processed.empty:
            print(f"    → 데이터 없음 (미상장 또는 기간 외)")
            continue
        print(f"    → {len(processed):,} rows  ({processed['date'].min()} ~ {processed['date'].max()})")
        collected.append(processed)

    if not collected:
        print("  신규 데이터 없음. 종료.")
        return

    new_data = pd.concat(collected, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"]).reset_index(drop=True)
    combined = combined.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    dupes = before - len(combined)

    print(f"\n  신규 rows: {len(new_data):,}  (중복 제거: {dupes})")
    print(f"  합산 rows: {len(combined):,}  범위: {combined['date'].min()} ~ {combined['date'].max()}")

    combined.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  저장 완료: {path.name}  ({size_mb:.1f} MB)")


def main():
    # 마지막 거래일 계산 (오늘이 월요일이면 금요일, 아니면 어제까지)
    today = date.today()
    # 주말 제외: 가장 최근 금요일 또는 평일
    days_back = 1
    while True:
        candidate = today - timedelta(days=days_back)
        if candidate.weekday() < 5:   # 0=월 ~ 4=금
            fetch_end = candidate
            break
        days_back += 1

    # 3y 파일: 기존 최신일 + 1일부터
    path_3y = config.OHLCV_1MIN_3Y
    existing_3y = pd.read_parquet(path_3y)
    last_date_3y = pd.to_datetime(existing_3y["date"].max()).date()
    fetch_start = last_date_3y + timedelta(days=1)
    # 주말 skip
    while fetch_start.weekday() >= 5:
        fetch_start += timedelta(days=1)

    if fetch_start > fetch_end:
        print(f"이미 최신 상태입니다. (마지막: {last_date_3y}, 오늘: {today})")
        return

    print(f"수집 기간: {fetch_start} ~ {fetch_end}")

    # 3y 파일 업데이트
    update_file(path_3y, fetch_start, fetch_end)

    # v2 파일도 동일하게 업데이트
    path_v2 = config.OHLCV_1MIN_DEFAULT
    existing_v2 = pd.read_parquet(path_v2)
    last_date_v2 = pd.to_datetime(existing_v2["date"].max()).date()
    start_v2 = last_date_v2 + timedelta(days=1)
    while start_v2.weekday() >= 5:
        start_v2 += timedelta(days=1)

    if start_v2 <= fetch_end:
        update_file(path_v2, start_v2, fetch_end)
    else:
        print(f"\nbacktest_1min_v2.parquet 이미 최신 상태 (마지막: {last_date_v2})")


if __name__ == "__main__":
    main()

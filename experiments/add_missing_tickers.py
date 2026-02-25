#!/usr/bin/env python3
"""
누락 티커 1분봉 수집 → backtest_1min_3y.parquet 병합
=====================================================
전략 코드에서 사용 중이지만 parquet에 없는 티커들을 수집한다.

대상: TSLL, AMDL, GDXU, IAU, SOXX, TQQQ, MSTX

실행:
    pyenv shell ptj_stock_lab && python experiments/add_missing_tickers.py
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

# 수집할 신규 티커 목록
NEW_TICKERS = ["TSLL", "AMDL", "GDXU", "IAU", "SOXX", "TQQQ", "MSTX"]

FETCH_START = date(2023, 1, 3)
FETCH_END   = date.today() - timedelta(days=1)
# 주말이면 금요일로 당김
while FETCH_END.weekday() >= 5:
    FETCH_END -= timedelta(days=1)

OUTPUT_PATH = config.OHLCV_1MIN_3Y


def _fetch_ticker(client, ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 월별 청킹 수집. SIP 실패 시 IEX 자동 재시도."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame(1, TimeFrameUnit.Minute)
    chunks, cur = [], start

    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        label = cur.strftime("%Y-%m")
        chunk_df = pd.DataFrame()

        for feed in [None, "iex"]:
            feed_label = f"feed={feed}" if feed else "SIP"
            try:
                kwargs = dict(
                    symbol_or_symbols=[ticker],
                    timeframe=tf,
                    start=datetime.combine(cur, time()),
                    end=datetime.combine(chunk_end + timedelta(days=1), time()),
                )
                if feed:
                    kwargs["feed"] = feed
                bars = client.get_stock_bars(StockBarsRequest(**kwargs))
                chunk_df = bars.df.reset_index()
                cnt = len(chunk_df)
                if cnt > 0:
                    print(f"      [{label}] {cnt:,} rows ({feed_label})", flush=True)
                break
            except Exception as e:
                err = str(e)
                if feed is None and "recent SIP" in err:
                    print(f"      [{label}] SIP 제한 → IEX 재시도", flush=True)
                    continue
                if feed is None and ("not found" in err.lower() or "no data" in err.lower()):
                    # 미상장 구간 — 조용히 skip
                    break
                print(f"      [{label}] SKIP — {e}", flush=True)
                break

        if not chunk_df.empty:
            chunks.append(chunk_df)
        cur = nxt

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _postprocess(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()
    if df.empty:
        return df
    df["symbol"] = ticker
    df["date"] = df["timestamp"].dt.date.astype(str)
    keep = ["symbol", "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap", "date"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    return df[keep].reset_index(drop=True)


def main():
    print(f"=== 누락 티커 수집: {NEW_TICKERS} ===")
    print(f"기간: {FETCH_START} ~ {FETCH_END}\n")

    client = get_alpaca_client()
    collected, results = [], {}

    for ticker in NEW_TICKERS:
        print(f"[{ticker}] 수집 중...", flush=True)
        raw = _fetch_ticker(client, ticker, FETCH_START, FETCH_END)
        processed = _postprocess(raw, ticker)

        if processed.empty:
            print(f"  → 데이터 없음 (미상장 또는 기간 외)\n")
            results[ticker] = None
            continue

        date_min, date_max = processed["date"].min(), processed["date"].max()
        rows = len(processed)
        print(f"  → {rows:,} rows  ({date_min} ~ {date_max})\n")
        collected.append(processed)
        results[ticker] = (date_min, date_max, rows)

    # 결과 요약
    print("=== 수집 결과 요약 ===")
    for t, r in results.items():
        if r:
            print(f"  ✅ {t:6s}: {r[0]} ~ {r[1]}  ({r[2]:,} rows)")
        else:
            print(f"  ❌ {t:6s}: 데이터 없음")

    if not collected:
        print("\n병합할 데이터가 없습니다.")
        return

    # 기존 parquet 로드 & 병합
    print(f"\n기존 parquet 로드: {OUTPUT_PATH.name}")
    existing = pd.read_parquet(OUTPUT_PATH)
    existing["date"] = existing["date"].astype(str)
    print(f"  기존: {len(existing):,} rows  ({existing['date'].min()} ~ {existing['date'].max()})")

    new_data = pd.concat(collected, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"]).reset_index(drop=True)
    combined = combined.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)
    dupes = before - len(combined)

    print(f"\n합산 rows: {len(combined):,}  (신규: {len(new_data):,}, 중복 제거: {dupes})")
    print(f"포함 종목: {sorted(combined['symbol'].unique())}")
    print(f"범위: {combined['date'].min()} ~ {combined['date'].max()}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n저장 완료: {OUTPUT_PATH.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
2023~2024 1분봉 데이터 수집
============================
기존 backtest_1min_v2.parquet (2025~) 와 합쳐
backtest_1min_3y.parquet (2023~2026) 를 생성한다.

실행:
    pyenv shell ptj_stock_lab && python experiments/fetch_historical_1min.py

주요 동작:
1. 현재 parquet 의 종목 목록 확인
2. 종목별 2023-01-03 ~ 2024-12-31 개별 수집 (상장 이전 종목은 skip)
3. 기존 데이터와 concat → 정렬 → backtest_1min_3y.parquet 저장
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

# ============================================================
# 설정
# ============================================================
FETCH_START = date(2023, 1, 3)
FETCH_END   = date(2024, 12, 31)

EXISTING_PATH = config.OHLCV_DIR / "backtest_1min_v2.parquet"
OUTPUT_PATH   = config.OHLCV_DIR / "backtest_1min_3y.parquet"

MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)

# PTJ 티커 → Alpaca 티커 매핑 (config 에서 가져옴)
TICKER_MAP         = config.ALPACA_TICKER_MAP          # {"IRE": "IREN", ...}
TICKER_REVERSE_MAP = config.ALPACA_TICKER_REVERSE      # {"IREN": "IRE", ...}


# ============================================================
# 월별 청킹 수집 (단일 종목)
# ============================================================
def _fetch_ticker(client, api_ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 월별 청킹 수집. 실패 시 빈 DataFrame 반환."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame(1, TimeFrameUnit.Minute)
    chunks: list[pd.DataFrame] = []
    cur = start

    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(nxt - timedelta(days=1), end)
        label = cur.strftime("%Y-%m")

        try:
            req = StockBarsRequest(
                symbol_or_symbols=[api_ticker],
                timeframe=tf,
                start=datetime.combine(cur, time()),
                end=datetime.combine(chunk_end + timedelta(days=1), time()),
            )
            bars = client.get_stock_bars(req)
            chunk_df = bars.df.reset_index()
            row_cnt = len(chunk_df)
            print(f"      [{label}] {row_cnt:,} rows", flush=True)
            if row_cnt > 0:
                chunks.append(chunk_df)
        except Exception as e:
            print(f"      [{label}] SKIP — {e}", flush=True)

        cur = nxt

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _postprocess(df: pd.DataFrame, ptj_ticker: str) -> pd.DataFrame:
    """Alpaca 원시 데이터를 parquet 스키마에 맞게 변환."""
    if df.empty:
        return df

    # timestamp: UTC → US/Eastern
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")

    # 정규장 필터
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()

    if df.empty:
        return df

    # symbol: PTJ 티커로 통일
    df["symbol"] = ptj_ticker

    # date 컬럼
    df["date"] = df["timestamp"].dt.date.astype(str)

    # 컬럼 순서: 기존 parquet 스키마에 맞춤
    keep = ["symbol", "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap", "date"]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    df = df[keep].reset_index(drop=True)

    return df


# ============================================================
# 메인
# ============================================================
def main():
    # 1. 기존 데이터 로드 → 종목 목록 파악
    print(f"=== 기존 데이터 로드: {EXISTING_PATH.name} ===")
    existing = pd.read_parquet(EXISTING_PATH)
    symbols = sorted(existing["symbol"].unique())
    print(f"  종목 ({len(symbols)}개): {symbols}")
    print(f"  기존 범위: {existing['date'].min()} ~ {existing['date'].max()}")
    print(f"  기존 rows: {len(existing):,}")
    print()

    # 2. 종목별 수집
    client = get_alpaca_client()
    collected: list[pd.DataFrame] = []
    success_tickers: list[str] = []
    skip_tickers:    list[str] = []

    for ptj_ticker in symbols:
        api_ticker = TICKER_MAP.get(ptj_ticker, ptj_ticker)
        print(f"  [{ptj_ticker}] Alpaca={api_ticker} 수집 중 ({FETCH_START} ~ {FETCH_END})")

        raw = _fetch_ticker(client, api_ticker, FETCH_START, FETCH_END)
        if raw.empty:
            print(f"    → 데이터 없음 (미상장 또는 기간 외) — SKIP")
            skip_tickers.append(ptj_ticker)
            continue

        processed = _postprocess(raw, ptj_ticker)
        if processed.empty:
            print(f"    → 정규장 데이터 없음 — SKIP")
            skip_tickers.append(ptj_ticker)
            continue

        row_cnt = len(processed)
        date_min = processed["date"].min()
        date_max = processed["date"].max()
        print(f"    → {row_cnt:,} rows  ({date_min} ~ {date_max})")
        collected.append(processed)
        success_tickers.append(ptj_ticker)

    print()
    print(f"=== 수집 결과 ===")
    print(f"  성공 ({len(success_tickers)}): {success_tickers}")
    print(f"  스킵 ({len(skip_tickers)}): {skip_tickers}")

    if not collected:
        print("수집된 데이터가 없습니다. 종료.")
        return

    # 3. 합치기
    new_data = pd.concat(collected, ignore_index=True)
    print(f"\n신규 수집 rows: {len(new_data):,}")

    # date 컬럼 타입 통일 (object/str)
    existing["date"] = existing["date"].astype(str)
    new_data["date"] = new_data["date"].astype(str)

    combined = pd.concat([new_data, existing], ignore_index=True)
    combined = combined.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    # 중복 제거 (혹시 날짜 겹치는 경우)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"]).reset_index(drop=True)
    after = len(combined)
    if before != after:
        print(f"  중복 제거: {before - after:,} rows")

    print(f"\n합산 rows: {len(combined):,}")
    print(f"합산 범위: {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"포함 종목: {sorted(combined['symbol'].unique())}")

    # 4. 저장
    print(f"\n저장 중: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"저장 완료: {OUTPUT_PATH.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

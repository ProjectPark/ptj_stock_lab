#!/usr/bin/env python3
"""
Alpaca에서 SOXX/IREN 일봉 데이터 수집
- 기간: 2025-01-03 ~ 2026-02-17
- 저장: stock_history/soxx_iren_daily.parquet
"""
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

client = StockHistoricalDataClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_SECRET_KEY"),
)

TICKERS = ["SOXX", "IREN"]
START = date(2025, 1, 3)
END = date(2026, 2, 17)

print(f"Alpaca 일봉 수집: {', '.join(TICKERS)}")
print(f"기간: {START} ~ {END}")

request = StockBarsRequest(
    symbol_or_symbols=TICKERS,
    timeframe=TimeFrame(1, TimeFrameUnit.Day),
    start=datetime.combine(START, time()),
    end=datetime.combine(END + timedelta(days=1), time()),
    feed="iex",
)

bars = client.get_stock_bars(request)
df = bars.df.reset_index()

print(f"\n수집 완료: {len(df):,} rows")
print(f"컬럼: {list(df.columns)}")

# 종목별 요약
for ticker in TICKERS:
    sub = df[df["symbol"] == ticker]
    if len(sub) > 0:
        print(f"\n  {ticker}: {len(sub)}일")
        print(f"    기간: {sub['timestamp'].min()} ~ {sub['timestamp'].max()}")
        print(f"    종가 범위: ${sub['close'].min():.2f} ~ ${sub['close'].max():.2f}")
    else:
        print(f"\n  {ticker}: 데이터 없음!")

# parquet 저장
out_path = BASE_DIR / "soxx_iren_daily.parquet"
df.to_parquet(out_path, index=False)
print(f"\n저장: {out_path}")
print(f"파일 크기: {out_path.stat().st_size / 1024:.1f} KB")

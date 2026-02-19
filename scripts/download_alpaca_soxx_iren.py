#!/usr/bin/env python3
"""
Alpaca에서 SOXX/IREN 일봉 데이터 수집
- 기간: 2025-01-03 ~ 2026-02-17
- 저장: data/market/daily/soxx_iren_daily.parquet
"""
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config

from fetchers.alpaca_fetcher import fetch_daily

TICKERS = ["SOXX", "IREN"]
START = date(2025, 1, 3)
END = date(2026, 2, 17)

out_path = config.DAILY_DIR / "soxx_iren_daily.parquet"

df = fetch_daily(
    tickers=TICKERS,
    start_date=START,
    end_date=END,
    cache_path=out_path,
    use_cache=False,
    feed="iex",
)

# 종목별 요약
for ticker in TICKERS:
    sub = df[df["symbol"] == ticker]
    if len(sub) > 0:
        print(f"\n  {ticker}: {len(sub)}일")
        print(f"    기간: {sub['timestamp'].min()} ~ {sub['timestamp'].max()}")
        print(f"    종가 범위: ${sub['close'].min():.2f} ~ ${sub['close'].max():.2f}")
    else:
        print(f"\n  {ticker}: 데이터 없음!")

print(f"\n파일 크기: {out_path.stat().st_size / 1024:.1f} KB")

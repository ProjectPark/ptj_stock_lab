"""
yfinance 일봉 데이터 수집 스크립트
- ticker_mapping.json의 모든 종목 + PTJ 시스템 필수 종목
- 기간: 2025-02-01 ~ 2026-02-17
- 출력: market_daily.parquet
"""
import json
import sys
from pathlib import Path

import yfinance as yf
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config

# 1) ticker_mapping.json에서 yf_ticker 추출
with open(config.META_DIR / "ticker_mapping.json") as f:
    mapping = json.load(f)

mapped_tickers = set()
for info in mapping.values():
    t = info.get("yf_ticker", "")
    if t:
        mapped_tickers.add(t)

# 2) PTJ 시스템 필수 종목 추가
ptj_essentials = {"GLD", "SPY", "QQQ", "BITU", "ETHU", "SOLT"}
all_tickers = sorted(mapped_tickers | ptj_essentials)

print(f"총 {len(all_tickers)}개 종목 다운로드 시작:")
print(", ".join(all_tickers))

# 3) yfinance 일괄 다운로드
df = yf.download(
    tickers=all_tickers,
    start="2025-02-01",
    end="2026-02-18",  # end는 exclusive이므로 +1일
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    threads=True,
)

print(f"\n다운로드 완료. Shape: {df.shape}")
print(f"기간: {df.index.min()} ~ {df.index.max()}")

# 4) 데이터 확인 — 각 종목별 행 수
for ticker in all_tickers:
    if ticker in df.columns.get_level_values(0):
        sub = df[ticker].dropna(how="all")
        print(f"  {ticker}: {len(sub)}일")
    else:
        print(f"  {ticker}: 데이터 없음!")

# 5) parquet 저장
out_path = config.DAILY_DIR / "market_daily.parquet"
df.to_parquet(out_path)
print(f"\n저장 완료: {out_path}")
print(f"파일 크기: {out_path.stat().st_size / 1024:.1f} KB")

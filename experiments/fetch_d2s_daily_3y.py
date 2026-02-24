#!/usr/bin/env python3
"""
D2S 일봉 3년치 데이터 수집
============================
market_daily.parquet(1년) → market_daily_3y.parquet(3년) 생성.

- 대상 티커: 기존 market_daily.parquet의 모든 종목 (32개)
- 기간: 2023-01-03 ~ 2026-02-17
- 포맷: wide format (DatetimeIndex × MultiIndex(ticker, OHLCV)) — market_daily.parquet 동일
- 신규 티커(2023년 미상장)는 실제 상장일부터 포함, 없으면 skip

실행:
    pyenv shell ptj_stock_lab && python experiments/fetch_d2s_daily_3y.py
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
FETCH_END   = date(2026, 2, 17)

EXISTING_PATH = config.DAILY_DIR / "market_daily.parquet"
OUTPUT_PATH   = config.DAILY_DIR / "market_daily_3y.parquet"


# ============================================================
# 단일 종목 일봉 수집
# ============================================================
def _fetch_daily_ticker(client, ticker: str, start: date, end: date) -> pd.DataFrame:
    """단일 종목 일봉 수집. 미상장이면 빈 DataFrame 반환."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    try:
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime.combine(start, time()),
            end=datetime.combine(end + timedelta(days=1), time()),
        )
        bars = client.get_stock_bars(req)
        df = bars.df.reset_index()
        return df
    except Exception as e:
        print(f"    SKIP — {e}", flush=True)
        return pd.DataFrame()


# ============================================================
# long → wide 변환 (market_daily.parquet 포맷)
# ============================================================
def to_wide_format(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    long format (symbol, timestamp, open, high, low, close, volume, ...)
    → wide format (DatetimeIndex, MultiIndex columns (ticker, OHLCV))
    """
    # timestamp를 Date index로 (UTC timezone-aware → naive 변환 후 날짜 추출)
    long_df = long_df.copy()
    ts_col = pd.to_datetime(long_df["timestamp"])
    if ts_col.dt.tz is not None:
        ts_col = ts_col.dt.tz_convert("US/Eastern").dt.tz_localize(None)
    long_df["Date"] = ts_col.dt.normalize()
    long_df = long_df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    # 필요 컬럼만
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    keep_cols = ["symbol", "Date"] + [c for c in ohlcv_cols if c in long_df.columns]
    long_df = long_df[keep_cols]

    # pivot → wide
    wide = long_df.pivot_table(
        index="Date",
        columns="symbol",
        values=[c for c in ohlcv_cols if c in long_df.columns],
        aggfunc="last",
    )
    # columns: (OHLCV, ticker) → (ticker, OHLCV) 순서로 swap
    wide.columns = pd.MultiIndex.from_tuples(
        [(ticker, ohlcv) for ohlcv, ticker in wide.columns],
        names=["Ticker", "Price"],
    )
    wide = wide.sort_index(axis=1, level=0)
    wide.index.name = "Date"
    return wide


# ============================================================
# 메인
# ============================================================
def main():
    # 1. 기존 market_daily.parquet 로드 → 티커 목록 파악
    print(f"=== 기존 데이터 로드: {EXISTING_PATH.name} ===")
    existing_wide = pd.read_parquet(EXISTING_PATH)
    tickers = sorted(set(c[0] for c in existing_wide.columns))
    print(f"  티커 ({len(tickers)}개): {tickers}")
    print(f"  기존 범위: {existing_wide.index.min().date()} ~ {existing_wide.index.max().date()}")
    print()

    # 2. 종목별 Alpaca 수집 (long format)
    client = get_alpaca_client()
    collected: list[pd.DataFrame] = []
    success_tickers: list[str] = []
    skip_tickers:    list[str] = []

    for ticker in tickers:
        print(f"  [{ticker}] 수집 중 ({FETCH_START} ~ {FETCH_END}) ...", end=" ", flush=True)
        raw = _fetch_daily_ticker(client, ticker, FETCH_START, FETCH_END)

        if raw.empty:
            print("데이터 없음 — SKIP")
            skip_tickers.append(ticker)
            continue

        row_cnt = len(raw)
        dates = pd.to_datetime(raw["timestamp"])
        print(f"{row_cnt}일  ({dates.min().date()} ~ {dates.max().date()})")

        raw["symbol"] = ticker  # 혹시 symbol 컬럼 없으면 추가
        collected.append(raw)
        success_tickers.append(ticker)

    print()
    print(f"=== 수집 결과 ===")
    print(f"  성공 ({len(success_tickers)}개): {success_tickers}")
    print(f"  스킵 ({len(skip_tickers)}개): {skip_tickers}")

    if not collected:
        print("수집된 데이터가 없습니다. 종료.")
        return

    # 3. long → wide 변환
    print("\n=== wide 포맷 변환 중 ===")
    all_long = pd.concat(collected, ignore_index=True)
    print(f"  long rows: {len(all_long):,}")

    wide = to_wide_format(all_long)
    print(f"  wide shape: {wide.shape}  ({wide.index.min().date()} ~ {wide.index.max().date()})")
    print(f"  포함 티커: {sorted(set(c[0] for c in wide.columns))}")

    # 4. 저장
    print(f"\n=== 저장: {OUTPUT_PATH.name} ===")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"저장 완료: {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    # 5. 검증 — 핵심 티커 날짜 범위 출력
    print("\n=== 핵심 D2S 티커 범위 검증 ===")
    for t in ["SPY", "GLD", "MSTU", "ROBN", "CONL", "AMDL"]:
        if t in [c[0] for c in wide.columns]:
            sub = wide[t]["Close"].dropna()
            if len(sub) > 0:
                print(f"  {t}: {sub.index.min().date()} ~ {sub.index.max().date()}  ({len(sub)}일)")
            else:
                print(f"  {t}: 데이터 없음")
        else:
            print(f"  {t}: 미포함")


if __name__ == "__main__":
    main()

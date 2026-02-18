"""
PTJ 매매법 - yfinance 데이터 수집 + parquet 캐시
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


def _cache_path(ticker: str) -> Path:
    return config.DATA_DIR / f"{ticker}.parquet"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=config.CACHE_EXPIRE_HOURS)


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """단일 티커 데이터 수집 (캐시 우선)."""
    cache = _cache_path(ticker)
    if _is_cache_valid(cache):
        return pd.read_parquet(cache)

    df = yf.download(
        ticker,
        period=config.LOOKBACK_PERIOD,
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return df

    # yfinance MultiIndex 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["ticker"] = ticker
    df.to_parquet(cache, index=False)
    return df


def fetch_all() -> pd.DataFrame:
    """모든 티커 데이터를 수집하여 하나의 DataFrame으로 반환."""
    frames = []
    for ticker in config.TICKERS:
        try:
            df = fetch_ticker(ticker)
            if not df.empty:
                frames.append(df)
                print(f"  [OK] {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] {ticker} 수집 실패: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_latest_changes(data: pd.DataFrame) -> dict[str, dict]:
    """각 티커의 최근 거래일 등락률 계산.

    Returns:
        {ticker: {"close": float, "prev_close": float, "change_pct": float, "date": str}}
    """
    result = {}
    for ticker in data["ticker"].unique():
        tdf = data[data["ticker"] == ticker].sort_values("Date")
        if len(tdf) < 2:
            continue
        latest = tdf.iloc[-1]
        prev = tdf.iloc[-2]
        prev_close = float(prev["Close"])
        curr_close = float(latest["Close"])
        change_pct = ((curr_close - prev_close) / prev_close) * 100 if prev_close else 0.0
        result[ticker] = {
            "close": round(curr_close, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "date": str(latest["Date"])[:10],
        }
    return result

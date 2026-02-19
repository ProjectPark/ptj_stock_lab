"""
USD/KRW 환율 데이터 수집기
===========================
yfinance에서 1시간봉 환율 데이터를 수집한다.

사용법:
    from fetchers.fx_fetcher import fetch_usdkrw_hourly

    df = fetch_usdkrw_hourly(date(2025, 1, 3), date(2026, 2, 17))
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config


def fetch_usdkrw_hourly(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """yfinance에서 USD/KRW 1시간봉 수집. parquet 캐시 지원.

    yfinance 제한: 1분봉은 최근 7일만, 1시간봉은 최근 730일까지 가능.
    환율 백테스트 용도로 1시간봉이면 충분.

    Args:
        start_date: 시작일
        end_date: 종료일
        use_cache: 캐시 사용 여부
        cache_path: 캐시 경로. None이면 기본 경로 사용.
    """
    if cache_path is None:
        cache_path = config.FX_DIR / "usdkrw_hourly.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [USD/KRW] 캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    import yfinance as yf

    print(f"  [USD/KRW] yfinance에서 1시간봉 수집 중...")
    print(f"  기간: {start_date} ~ {end_date}")

    ticker = yf.Ticker("KRW=X")
    raw = ticker.history(
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),
        interval="1h",
    )

    if raw.empty:
        logger.warning("USD/KRW 데이터 없음!")
        print("  [USD/KRW] 데이터 없음!")
        return pd.DataFrame()

    df = raw.reset_index()
    df = df.rename(columns={
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # 타임존 → US/Eastern 변환 (주식 시간과 맞추기)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

    df["date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  [USD/KRW] 수집 완료: {len(df):,} rows → {cache_path.name}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  환율 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    return df

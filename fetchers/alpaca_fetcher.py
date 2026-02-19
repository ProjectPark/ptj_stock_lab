"""
통합 Alpaca 데이터 수집기
=========================
1분봉, 5분봉, 일봉 데이터를 통일된 인터페이스로 수집한다.

사용법:
    from fetchers.alpaca_fetcher import fetch_bars, fetch_1min_v1, fetch_1min_v2

    # 범용
    df = fetch_bars(["AAPL", "MSFT"], 5, date(2025,1,3), date(2026,2,17))

    # 편의 래퍼 (기존 호환)
    df = fetch_1min_v2(start_date=date(2025,1,3), end_date=date(2026,2,17))
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 정규장 시간 (US/Eastern)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# 싱글턴 클라이언트
_client = None


def get_alpaca_client():
    """Alpaca StockHistoricalDataClient 싱글턴."""
    global _client
    if _client is not None:
        return _client

    import sys
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    import config

    from alpaca.data.historical import StockHistoricalDataClient

    load_dotenv(config.PROJECT_ROOT / ".env")
    _client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )
    return _client


def _fetch_monthly_chunks(
    client,
    tickers: list[str],
    timeframe,
    start_date: date,
    end_date: date,
    verbose: bool = True,
) -> pd.DataFrame:
    """월 단위 청킹으로 데이터 수집. 1분봉/5분봉 등 대량 데이터 대응."""
    from alpaca.data.requests import StockBarsRequest

    chunks: list[pd.DataFrame] = []
    chunk_start = start_date

    while chunk_start <= end_date:
        if chunk_start.month == 12:
            next_month = date(chunk_start.year + 1, 1, 1)
        else:
            next_month = date(chunk_start.year, chunk_start.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end_date)

        label = chunk_start.strftime("%Y-%m")
        if verbose:
            print(f"    [{label}] {chunk_start} ~ {chunk_end} ...", end=" ", flush=True)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=timeframe,
                start=datetime.combine(chunk_start, time()),
                end=datetime.combine(chunk_end + timedelta(days=1), time()),
            )
            bars = client.get_stock_bars(request)
            chunk_df = bars.df.reset_index()
            if verbose:
                print(f"{len(chunk_df):,} rows")
            chunks.append(chunk_df)
        except Exception as e:
            logger.warning("Chunk %s failed: %s", label, e)
            if verbose:
                print(f"SKIP ({e})")

        chunk_start = next_month

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def fetch_bars(
    tickers: list[str],
    timeframe_minutes: int,
    start_date: date,
    end_date: date,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
    ticker_map: dict | None = None,
    ticker_reverse_map: dict | None = None,
    market_hours_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """통합 Alpaca 데이터 수집 함수.

    Args:
        tickers: PTJ 티커 리스트 (예: ["BITU", "MSTU", "IRE"])
        timeframe_minutes: 1(1분봉), 5(5분봉), 1440(일봉)
        start_date: 시작일
        end_date: 종료일
        cache_path: parquet 캐시 경로. None이면 캐시 미사용.
        use_cache: True이면 기존 캐시 로드
        ticker_map: PTJ→Alpaca 티커 매핑 (예: {"IRE": "IREN"})
        ticker_reverse_map: Alpaca→PTJ 역매핑 (예: {"IREN": "IRE"})
        market_hours_only: True이면 정규장(9:30~16:00 ET)만 필터
        verbose: 진행 상황 출력

    Returns:
        pd.DataFrame with columns: symbol, timestamp, open, high, low, close, volume, ...
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    # 캐시 확인
    if use_cache and cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        if verbose:
            print(f"  캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    client = get_alpaca_client()

    # 티커 매핑 (PTJ → Alpaca)
    if ticker_map:
        api_tickers = [ticker_map.get(t, t) for t in tickers]
    else:
        api_tickers = list(tickers)

    # Timeframe 결정
    if timeframe_minutes >= 1440:
        tf = TimeFrame(1, TimeFrameUnit.Day)
        is_daily = True
    elif timeframe_minutes >= 60:
        tf = TimeFrame(timeframe_minutes // 60, TimeFrameUnit.Hour)
        is_daily = False
    else:
        tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
        is_daily = False

    if verbose:
        print(f"  Alpaca에서 {len(api_tickers)}개 종목 {timeframe_minutes}분봉 수집 중...")
        print(f"  기간: {start_date} ~ {end_date}")

    # 일봉은 단일 호출, 분봉은 월별 청킹
    if is_daily:
        request = StockBarsRequest(
            symbol_or_symbols=api_tickers,
            timeframe=tf,
            start=datetime.combine(start_date, time()),
            end=datetime.combine(end_date + timedelta(days=1), time()),
        )
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()
    else:
        df = _fetch_monthly_chunks(
            client, api_tickers, tf, start_date, end_date, verbose=verbose,
        )

    if df.empty:
        logger.warning("No data returned from Alpaca")
        return df

    if verbose:
        print(f"  총 수집: {len(df):,} rows")

    # Alpaca 티커 → PTJ 티커 역매핑
    if ticker_reverse_map:
        df["symbol"] = df["symbol"].replace(ticker_reverse_map)

    # UTC → US/Eastern 변환 (일봉은 타임존이 다를 수 있으므로 안전하게 처리)
    if not is_daily:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")

        # 정규장 필터
        if market_hours_only:
            t = df["timestamp"].dt.time
            df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()

    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    # 캐시 저장
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"  필터 후: {len(df):,} rows → {cache_path.name}")

    return df


# ============================================================
# 편의 래퍼 (하위 호환)
# ============================================================

def fetch_5min_v1(
    start_date: date = date(2025, 2, 8),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """5분봉 데이터 수집 (v1 종목)."""
    import sys
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    import config

    return fetch_bars(
        tickers=list(config.TICKERS.keys()),
        timeframe_minutes=5,
        start_date=start_date,
        end_date=end_date,
        cache_path=config.OHLCV_DIR / "backtest_5min.parquet",
        use_cache=use_cache,
    )


def fetch_1min_v1(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """1분봉 데이터 수집 (v1 종목)."""
    import sys
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    import config

    return fetch_bars(
        tickers=list(config.TICKERS.keys()),
        timeframe_minutes=1,
        start_date=start_date,
        end_date=end_date,
        cache_path=config.OHLCV_DIR / "backtest_1min.parquet",
        use_cache=use_cache,
    )


def fetch_1min_v2(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """1분봉 데이터 수집 (v2 종목 + 티커 매핑)."""
    import sys
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    import config

    return fetch_bars(
        tickers=list(config.TICKERS_V2.keys()),
        timeframe_minutes=1,
        start_date=start_date,
        end_date=end_date,
        cache_path=config.OHLCV_DIR / "backtest_1min_v2.parquet",
        use_cache=use_cache,
        ticker_map=config.ALPACA_TICKER_MAP,
        ticker_reverse_map=config.ALPACA_TICKER_REVERSE,
    )


def fetch_daily(
    tickers: list[str],
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
    feed: str | None = None,
) -> pd.DataFrame:
    """일봉 데이터 수집.

    Args:
        tickers: 종목 리스트
        start_date: 시작일
        end_date: 종료일
        cache_path: 캐시 경로
        use_cache: 캐시 사용 여부
        feed: Alpaca 데이터 피드 (예: "iex"). None이면 기본값.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if use_cache and cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    client = get_alpaca_client()

    print(f"Alpaca 일봉 수집: {', '.join(tickers)}")
    print(f"기간: {start_date} ~ {end_date}")

    kwargs = dict(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=datetime.combine(start_date, time()),
        end=datetime.combine(end_date + timedelta(days=1), time()),
    )
    if feed:
        kwargs["feed"] = feed

    request = StockBarsRequest(**kwargs)
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    print(f"수집 완료: {len(df):,} rows")

    # date 컬럼 추가 (fetch_bars 스키마 통일)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"저장: {cache_path}")

    return df

"""
데이터 수집 패키지
==================
Alpaca, FX, 캐시 메타데이터를 통합 제공.

사용법:
    from fetchers import fetch_bars, fetch_1min_v2, fetch_usdkrw_hourly
    from fetchers import CacheMeta, is_cache_sufficient
"""

from fetchers.alpaca_fetcher import (
    fetch_bars,
    fetch_1min_v1,
    fetch_1min_v2,
    fetch_5min_v1,
    fetch_daily,
)
from fetchers.fx_fetcher import fetch_usdkrw_hourly
from fetchers.cache_meta import CacheMeta, save_meta, load_meta, is_cache_sufficient

__all__ = [
    "fetch_bars",
    "fetch_1min_v1",
    "fetch_1min_v2",
    "fetch_5min_v1",
    "fetch_daily",
    "fetch_usdkrw_hourly",
    "CacheMeta",
    "save_meta",
    "load_meta",
    "is_cache_sufficient",
]

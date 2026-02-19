"""
캐시 메타데이터 관리
====================
parquet 캐시 파일의 유효성을 검증하기 위한 메타데이터 시스템.

사용법:
    from fetchers.cache_meta import CacheMeta, save_meta, load_meta, is_cache_sufficient

    meta = CacheMeta(
        tickers=["BITU", "MSTU"],
        start_date="2025-01-03",
        end_date="2026-02-17",
        timeframe_minutes=1,
        created_at="2026-02-19T10:00:00",
        row_count=1234567,
    )
    save_meta(meta, Path("data/market/ohlcv/backtest_1min.parquet"))
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheMeta:
    """캐시 파일의 메타데이터."""
    tickers: list[str]
    start_date: str           # ISO format (YYYY-MM-DD)
    end_date: str             # ISO format (YYYY-MM-DD)
    timeframe_minutes: int    # 1, 5, 60, 1440
    created_at: str           # ISO datetime
    row_count: int


def _meta_path(cache_path: Path) -> Path:
    """캐시 파일에 대응하는 .meta.json 경로."""
    return cache_path.with_suffix(".meta.json")


def save_meta(meta: CacheMeta, cache_path: Path) -> Path:
    """메타데이터를 .meta.json 파일로 저장."""
    path = _meta_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    logger.debug("Cache meta saved: %s", path)
    return path


def load_meta(cache_path: Path) -> CacheMeta | None:
    """캐시 파일의 메타데이터를 로드. 없으면 None."""
    path = _meta_path(cache_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CacheMeta(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to load cache meta %s: %s", path, e)
        return None


def is_cache_sufficient(
    cache_path: Path,
    required_tickers: list[str],
    required_start: date,
    required_end: date,
) -> bool:
    """캐시가 요구 조건을 충족하는지 확인.

    조건:
    1. 캐시 파일 존재
    2. 메타데이터 존재
    3. 요구 티커가 모두 포함
    4. 요구 기간이 캐시 기간 내에 포함

    Args:
        cache_path: parquet 캐시 파일 경로
        required_tickers: 필요한 티커 리스트
        required_start: 필요한 시작일
        required_end: 필요한 종료일

    Returns:
        True if cache is sufficient
    """
    if not cache_path.exists():
        return False

    meta = load_meta(cache_path)
    if meta is None:
        return False

    # 티커 포함 확인
    cached_tickers = set(meta.tickers)
    if not set(required_tickers).issubset(cached_tickers):
        logger.debug("Cache missing tickers: %s",
                      set(required_tickers) - cached_tickers)
        return False

    # 기간 포함 확인
    cached_start = date.fromisoformat(meta.start_date)
    cached_end = date.fromisoformat(meta.end_date)
    if required_start < cached_start or required_end > cached_end:
        logger.debug("Cache period insufficient: cached %s~%s, required %s~%s",
                      cached_start, cached_end, required_start, required_end)
        return False

    return True

"""
Polymarket 과거 데이터 수집기 (1min / 5min / 1h / 1d)
=====================================================
CLOB API prices-history를 통해 지정 간격의 과거 확률 시계열을 수집한다.

사용법:
    from polymarket.poly_history import collect_history_for_date, collect_history_range

    # 단일 날짜, 1분봉
    data = collect_history_for_date(date(2026, 2, 15), fidelity=1)

    # 날짜 범위, 5분봉
    all_data = collect_history_range(date(2026, 2, 1), date(2026, 2, 15), fidelity=5)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import sys

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config

from polymarket.poly_config import (
    GAMMA_API_BASE,
    INDICATORS,
    REQUEST_TIMEOUT,
)
from polymarket.poly_common import fetch_event as _fetch_event, extract_final_prices
from polymarket.poly_fetcher import build_slug

logger = logging.getLogger(__name__)

CLOB_API_BASE = "https://clob.polymarket.com"
HISTORY_DIR = config.POLY_DATA_DIR

# rate limit 간격 (초)
API_DELAY = 0.4


# ------------------------------------------------------------------
# Token ID 추출
# ------------------------------------------------------------------

def extract_token_ids(event: dict) -> list[dict]:
    """이벤트에서 각 마켓의 outcome별 CLOB token ID를 추출.

    Returns:
        [
            {
                "market_question": "Bitcoin Up or Down ...?",
                "outcomes": [
                    {"label": "Up", "token_id": "3915848..."},
                    {"label": "Down", "token_id": "6452601..."},
                ],
            },
            ...
        ]
    """
    result = []
    for market in event.get("markets", []):
        outcomes = json.loads(market.get("outcomes", "[]"))
        token_ids = json.loads(market.get("clobTokenIds", "[]"))
        if not token_ids:
            continue
        result.append({
            "market_question": market.get("question", ""),
            "outcomes": [
                {"label": outcomes[i], "token_id": token_ids[i]}
                for i in range(min(len(outcomes), len(token_ids)))
            ],
        })
    return result


# ------------------------------------------------------------------
# CLOB API: prices-history
# ------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1,
) -> list[dict]:
    """CLOB API에서 특정 토큰의 가격 히스토리를 가져온다.

    Args:
        token_id: CLOB token ID (긴 숫자 문자열)
        start_ts: 시작 Unix timestamp (UTC)
        end_ts: 종료 Unix timestamp (UTC)
        fidelity: 데이터 해상도 (분 단위). 1=1분, 5=5분, 60=1시간

    Returns:
        [{"t": 1771261819, "p": 0.5}, ...]
    """
    url = f"{CLOB_API_BASE}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])
    except requests.RequestException as e:
        logger.error("CLOB prices-history failed for token %s...%s: %s",
                     token_id[:20], token_id[-10:], e)
        return []


# _fetch_event는 poly_common에서 import

# ------------------------------------------------------------------
# 날짜 → Unix timestamp 변환
# ------------------------------------------------------------------

def _date_to_ts_range(target_date: date, margin_hours: int = 48) -> tuple[int, int]:
    """날짜에 대한 타임스탬프 범위를 생성.

    이벤트는 보통 1~2일 전에 생성되므로 margin을 둔다.
    """
    dt = datetime(target_date.year, target_date.month, target_date.day,
                  tzinfo=timezone.utc)
    start = dt - timedelta(hours=margin_hours)
    end = dt + timedelta(hours=24)  # 당일 마감까지
    return int(start.timestamp()), int(end.timestamp())


# ------------------------------------------------------------------
# 단일 날짜 수집
# ------------------------------------------------------------------

def collect_history_for_date(
    target_date: date,
    fidelity: int = 1,
    indicators: dict | None = None,
) -> dict:
    """특정 날짜의 모든 지표에 대해 가격 히스토리를 수집.

    Args:
        target_date: 수집 대상 날짜
        fidelity: 데이터 해상도 (분). 1=1분봉, 5=5분봉
        indicators: 수집할 지표 딕셔너리. None이면 전체 INDICATORS 사용.

    Returns:
        {
            "date": "2026-02-15",
            "fidelity_minutes": 1,
            "indicators": {
                "btc_up_down": {
                    "slug": "bitcoin-up-or-down-on-february-15",
                    "title": "Bitcoin Up or Down on February 15?",
                    "resolved": true,
                    "final_prices": {"Up": "0", "Down": "1"},
                    "markets": [
                        {
                            "question": "Bitcoin Up or Down ...?",
                            "outcomes": {
                                "Up": [{"t": ..., "p": ...}, ...],
                                "Down": [{"t": ..., "p": ...}, ...],
                            }
                        }
                    ]
                },
                ...
            }
        }
    """
    if indicators is None:
        indicators = INDICATORS

    ref_dt = datetime(target_date.year, target_date.month, target_date.day)
    start_ts, end_ts = _date_to_ts_range(target_date)

    result = {
        "date": target_date.isoformat(),
        "fidelity_minutes": fidelity,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "indicators": {},
    }

    for name, indicator in indicators.items():
        logger.info("Collecting %s for %s (fidelity=%d min)", name, target_date, fidelity)
        try:
            slug = build_slug(name, indicator, ref_dt)
            if not slug:
                result["indicators"][name] = {"error": "slug 생성 실패"}
                continue

            event = _fetch_event(slug)
            time.sleep(API_DELAY)

            if not event:
                result["indicators"][name] = {
                    "slug": slug,
                    "error": "이벤트 조회 실패",
                }
                continue

            token_groups = extract_token_ids(event)
            if not token_groups:
                result["indicators"][name] = {
                    "slug": slug,
                    "error": "token ID 없음",
                }
                continue

            # 최종 가격 (정산 결과) 추출
            final_prices = extract_final_prices(event)

            # 각 마켓의 outcome별 가격 히스토리 수집
            markets_data = []
            for group in token_groups:
                outcomes_history = {}
                for item in group["outcomes"]:
                    history = fetch_price_history(
                        item["token_id"], start_ts, end_ts, fidelity,
                    )
                    outcomes_history[item["label"]] = history
                    time.sleep(API_DELAY)

                markets_data.append({
                    "question": group["market_question"],
                    "outcomes": outcomes_history,
                })

            result["indicators"][name] = {
                "slug": slug,
                "title": event.get("title", ""),
                "resolved": event.get("closed", False),
                "final_prices": final_prices,
                "volume": float(event.get("volume", 0)),
                "markets": markets_data,
            }
            logger.info("  → %s: %d markets collected", name, len(markets_data))

        except Exception as e:
            logger.error("Failed to collect %s: %s", name, e, exc_info=True)
            result["indicators"][name] = {"error": str(e)}

    return result


# ------------------------------------------------------------------
# 날짜 범위 수집
# ------------------------------------------------------------------

def collect_history_range(
    start_date: date,
    end_date: date,
    fidelity: int = 1,
    indicators: dict | None = None,
    save: bool = True,
) -> list[dict]:
    """날짜 범위에 대해 순회하며 히스토리를 수집.

    Args:
        start_date: 시작 날짜 (포함)
        end_date: 종료 날짜 (포함)
        fidelity: 데이터 해상도 (분)
        indicators: 수집할 지표. None이면 전체.
        save: True이면 JSON 파일로 저장.

    Returns:
        날짜별 수집 결과 리스트
    """
    results = []
    current = start_date

    while current <= end_date:
        logger.info("=" * 60)
        logger.info("Collecting %s (fidelity=%d min)", current, fidelity)
        logger.info("=" * 60)

        data = collect_history_for_date(current, fidelity, indicators)
        results.append(data)

        if save:
            save_history(data, fidelity)

        current += timedelta(days=1)

    return results


# ------------------------------------------------------------------
# 저장 / 로드
# ------------------------------------------------------------------

def save_history(data: dict, fidelity: int) -> Path:
    """수집 결과를 JSON 파일로 저장.

    파일 경로: polymarket/history/{date}_{fidelity}m.json
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{data['date']}_{fidelity}m.json"
    filepath = HISTORY_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s (%d bytes)", filepath, filepath.stat().st_size)
    return filepath


def load_history(target_date: date, fidelity: int = 1) -> dict | None:
    """저장된 JSON 파일에서 히스토리를 로드."""
    filename = f"{target_date.isoformat()}_{fidelity}m.json"
    filepath = HISTORY_DIR / filename

    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_available_dates(fidelity: int | None = None) -> list[dict]:
    """저장된 히스토리 파일 목록을 반환.

    Returns:
        [{"date": "2026-02-15", "fidelity": 1, "file": "2026-02-15_1m.json"}, ...]
    """
    if not HISTORY_DIR.exists():
        return []

    result = []
    for f in sorted(HISTORY_DIR.glob("*.json")):
        parts = f.stem.split("_")
        if len(parts) == 2:
            d = parts[0]
            fid = int(parts[1].rstrip("m"))
            if fidelity is None or fid == fidelity:
                result.append({"date": d, "fidelity": fid, "file": f.name})
    return result

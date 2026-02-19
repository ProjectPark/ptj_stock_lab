"""
Polymarket 공통 헬퍼
====================
여러 polymarket 모듈에서 공유하는 유틸리티.

- MONTH_NUM: 영문 월명 → 숫자 매핑
- fetch_event: Gamma API 이벤트 1건 조회
- extract_final_prices: 이벤트의 최종 가격(정산 결과) 추출
"""
from __future__ import annotations

import json
import logging

import requests

from polymarket.poly_config import GAMMA_API_BASE, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# 영문 월명 → 숫자 매핑 (FOMC 탐색, slug 생성 등에 사용)
MONTH_NUM: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def fetch_event(slug: str) -> dict | None:
    """Gamma API에서 이벤트 1건을 조회한다."""
    url = f"{GAMMA_API_BASE}/events"
    try:
        resp = requests.get(url, params={"slug": slug}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            logger.warning("Empty response for slug: %s", slug)
            return None
        return data[0]
    except requests.RequestException as e:
        logger.error("API request failed for slug %s: %s", slug, e)
        return None
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        logger.error("Failed to parse response for slug %s: %s", slug, e)
        return None


def extract_final_prices(event: dict) -> dict[str, str]:
    """이벤트의 최종 가격(정산 결과)을 추출한다.

    Returns:
        {"Up": "0", "Down": "1"} 형태의 outcome별 가격 딕셔너리.
    """
    final_prices: dict[str, str] = {}
    for market in event.get("markets", []):
        outcomes = json.loads(market.get("outcomes", "[]"))
        prices = json.loads(market.get("outcomePrices", "[]"))
        for i in range(min(len(outcomes), len(prices))):
            final_prices[outcomes[i]] = prices[i]
    return final_prices

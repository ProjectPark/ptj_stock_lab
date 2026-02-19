"""
Polymarket 데이터 수집기
========================
Gamma API를 통해 Polymarket 예측 시장 데이터를 수집하고 파싱한다.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import requests

from polymarket.poly_config import (
    FOMC_MONTHS,
    GAMMA_API_BASE,
    INDICATORS,
    REQUEST_TIMEOUT,
    IndicatorResult,
    MarketProb,
    MarketType,
    SlugType,
)
from polymarket.poly_common import MONTH_NUM, fetch_event as _fetch_event

logger = logging.getLogger(__name__)

# FOMC slug 캐시 (모듈 레벨)
_fomc_slug_cache: str | None = None


# ------------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------------

def _format_price_label(value: float) -> str:
    """가격을 읽기 쉬운 라벨로 변환. 예: 68000 → '$68K'"""
    if value >= 1000:
        k = value / 1000
        if k == int(k):
            return f"${int(k)}K"
        return f"${k:.1f}K"
    return f"${value:,.0f}"


# ------------------------------------------------------------------
# Slug 생성
# ------------------------------------------------------------------

def build_slug(name: str, indicator: dict, ref_date: datetime | None = None) -> str | None:
    """지표 설정 + 날짜 → Polymarket slug 생성.

    FOMC 타입은 ``_find_active_fomc_slug`` 를 호출하며, 실패 시 None 을 반환한다.
    """
    if ref_date is None:
        ref_date = datetime.now()

    # day_offset 적용 (예: btc_above_tomorrow → +1일)
    day_offset = indicator.get("day_offset", 0)
    if day_offset:
        ref_date = ref_date + timedelta(days=day_offset)

    slug_type = indicator["slug_type"]
    template = indicator["slug_template"]

    if slug_type == SlugType.DAILY:
        return template.format(
            month=ref_date.strftime("%B").lower(),
            day=ref_date.day,
        )

    if slug_type == SlugType.DAILY_WITH_YEAR:
        return template.format(
            month=ref_date.strftime("%B").lower(),
            day=ref_date.day,
            year=ref_date.year,
        )

    if slug_type == SlugType.WEEKLY:
        # Polymarket 주간: 월요일~일요일
        weekday = ref_date.weekday()  # Mon=0 … Sun=6
        monday = ref_date - timedelta(days=weekday)
        sunday = monday + timedelta(days=6)
        return template.format(
            month=monday.strftime("%B").lower(),
            week_start=monday.day,
            week_end=sunday.day,
        )

    if slug_type == SlugType.MONTHLY:
        return template.format(
            month=ref_date.strftime("%B").lower(),
            year=ref_date.year,
        )

    if slug_type == SlugType.FOMC:
        return _find_active_fomc_slug(template, ref_date)

    return None


# ------------------------------------------------------------------
# API 호출 (_fetch_event는 poly_common에서 import)
# ------------------------------------------------------------------

def _is_event_resolved(event: dict) -> bool:
    """이벤트가 이미 종료(resolved)되었는지 확인.

    공식 closed/active 외에, BINARY 마켓에서 한쪽 확률이 99% 이상이면
    사실상 종료된 것으로 판단한다 (예: 공휴일 NDX).
    """
    if event.get("closed") or not event.get("active"):
        return True
    # BINARY 마켓: 한쪽이 99% 이상이면 사실상 resolved
    markets = event.get("markets", [])
    if len(markets) == 1:
        try:
            prices = json.loads(markets[0].get("outcomePrices", "[]"))
            if any(float(p) >= 0.99 for p in prices):
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


# ------------------------------------------------------------------
# 마켓 파싱
# ------------------------------------------------------------------

def _parse_markets(event: dict, indicator: dict) -> tuple[list[MarketProb], float]:
    """이벤트의 markets 를 MarketType 에 따라 파싱한다."""
    market_type = indicator["market_type"]
    raw_markets = event.get("markets", [])
    volume = float(event.get("volume", 0))

    parsers = {
        MarketType.BINARY: lambda: _parse_binary(raw_markets),
        MarketType.MULTI_LEVEL: lambda: _parse_multi_level(raw_markets, indicator),
        MarketType.RANGE: lambda: _parse_range(raw_markets),
        MarketType.REACH_DIP: lambda: _parse_reach_dip(raw_markets),
        MarketType.FED: lambda: _parse_fed(raw_markets),
    }
    parser = parsers.get(market_type)
    if parser is None:
        return [], volume
    return parser(), volume


def _parse_binary(markets: list[dict]) -> list[MarketProb]:
    """BINARY: 단일 마켓의 Up/Down 확률."""
    if not markets:
        return []
    m = markets[0]
    outcomes = json.loads(m["outcomes"])
    prices = json.loads(m["outcomePrices"])
    return [
        MarketProb(label=outcomes[i], prob=float(prices[i]), value=None)
        for i in range(len(outcomes))
    ]


def _parse_multi_level(markets: list[dict], indicator: dict) -> list[MarketProb]:
    """MULTI_LEVEL: 'above $X?' 다중 마켓. closed 제외, value 기준 정렬."""
    price_regex = indicator.get("price_regex", r"above \$([0-9,]+)")
    results: list[MarketProb] = []
    for m in markets:
        if m.get("closed", False):
            continue
        question = m.get("question", "")
        match = re.search(price_regex, question, re.IGNORECASE)
        if not match:
            continue
        price_value = float(match.group(1).replace(",", ""))
        prices = json.loads(m["outcomePrices"])
        yes_prob = float(prices[0])
        results.append(MarketProb(
            label=_format_price_label(price_value),
            prob=yes_prob,
            value=price_value,
        ))
    results.sort(key=lambda x: x.value)
    return results


def _parse_range(markets: list[dict]) -> list[MarketProb]:
    """RANGE: 'between $X and $Y', 'less than $X', 'greater than $X'."""
    results: list[MarketProb] = []
    for m in markets:
        if m.get("closed", False):
            continue
        question = m.get("question", "")
        prices = json.loads(m["outcomePrices"])
        yes_prob = float(prices[0])

        between = re.search(r"between \$([0-9,]+) and \$([0-9,]+)", question, re.IGNORECASE)
        if between:
            low = float(between.group(1).replace(",", ""))
            high = float(between.group(2).replace(",", ""))
            label = f"{_format_price_label(low)}-{_format_price_label(high)}"
            sort_value = low
        else:
            less = re.search(r"less than \$([0-9,]+)", question, re.IGNORECASE)
            if less:
                val = float(less.group(1).replace(",", ""))
                label = f"< {_format_price_label(val)}"
                sort_value = val - 1
            else:
                greater = re.search(r"greater than \$([0-9,]+)", question, re.IGNORECASE)
                if greater:
                    val = float(greater.group(1).replace(",", ""))
                    label = f"> {_format_price_label(val)}"
                    sort_value = val + 1
                else:
                    continue

        results.append(MarketProb(label=label, prob=yes_prob, value=sort_value))

    results.sort(key=lambda x: x.value)
    return results


def _parse_reach_dip(markets: list[dict]) -> list[MarketProb]:
    """REACH_DIP: reach / dip 분리 파싱."""
    price_re = re.compile(r"\$([0-9,]+)")
    reach_items: list[MarketProb] = []
    dip_items: list[MarketProb] = []

    for m in markets:
        if m.get("closed", False):
            continue
        question = m.get("question", "")
        prices = json.loads(m["outcomePrices"])
        yes_prob = float(prices[0])

        price_match = price_re.search(question)
        if not price_match:
            continue
        price_value = float(price_match.group(1).replace(",", ""))

        q_lower = question.lower()
        if "reach" in q_lower:
            reach_items.append(MarketProb(
                label=f"reach {_format_price_label(price_value)}",
                prob=yes_prob,
                value=price_value,
            ))
        elif "dip" in q_lower:
            dip_items.append(MarketProb(
                label=f"dip {_format_price_label(price_value)}",
                prob=yes_prob,
                value=price_value,
            ))

    reach_items.sort(key=lambda x: x.value)
    dip_items.sort(key=lambda x: x.value, reverse=True)
    return reach_items + dip_items


def _parse_fed(markets: list[dict]) -> list[MarketProb]:
    """FED: 4개 마켓의 결정 유형 파싱."""
    label_map = [
        (r"50\+?\s*bps?", "50bp+ 인하"),
        (r"decrease.*?25\s*bps?", "25bp 인하"),
        (r"no change", "변경없음"),
        (r"increase", "25bp+ 인상"),
    ]
    results: list[MarketProb] = []
    for m in markets:
        question = m.get("question", "")
        prices = json.loads(m["outcomePrices"])
        yes_prob = float(prices[0])

        label = question[:30]  # fallback
        for pattern, lbl in label_map:
            if re.search(pattern, question, re.IGNORECASE):
                label = lbl
                break

        results.append(MarketProb(label=label, prob=yes_prob, value=None))
    return results


# ------------------------------------------------------------------
# FOMC 탐색
# ------------------------------------------------------------------

def _find_active_fomc_slug(template: str, ref_date: datetime) -> str | None:
    """FOMC 활성 이벤트 slug 탐색.

    현재 월부터 FOMC_MONTHS 를 순회하며 active & not closed 이벤트를 찾는다.
    찾으면 모듈 레벨 캐시에 저장.
    """
    global _fomc_slug_cache
    if _fomc_slug_cache is not None:
        return _fomc_slug_cache

    # 현재 월 이상인 첫 FOMC 월 인덱스 찾기
    start_idx = 0
    for i, month_name in enumerate(FOMC_MONTHS):
        if MONTH_NUM.get(month_name, 0) >= ref_date.month:
            start_idx = i
            break

    for attempt in range(12):
        idx = (start_idx + attempt) % len(FOMC_MONTHS)
        slug = template.format(month=FOMC_MONTHS[idx])

        logger.debug("FOMC search: trying slug %s", slug)
        event = _fetch_event(slug)
        if event and event.get("active") and not event.get("closed"):
            _fomc_slug_cache = slug
            logger.info("Found active FOMC event: %s", slug)
            return slug

    logger.warning("No active FOMC event found after 12 attempts")
    return None


# ------------------------------------------------------------------
# 메인 수집 함수
# ------------------------------------------------------------------

def fetch_all_indicators(ref_date: datetime | None = None) -> dict[str, IndicatorResult]:
    """모든 지표를 수집하여 name → IndicatorResult 딕셔너리로 반환.

    Args:
        ref_date: 기준 날짜. None 이면 현재 시각 사용.

    Returns:
        dict[str, IndicatorResult]. 실패한 지표도 error 필드와 함께 포함된다.
    """
    if ref_date is None:
        # Polymarket은 뉴욕 시간(ET) 기준으로 동작
        # UTC-5 (EST) / UTC-4 (EDT) — 간단히 UTC-5로 처리
        et_offset = timezone(timedelta(hours=-5))
        ref_date = datetime.now(et_offset).replace(tzinfo=None)

    results: dict[str, IndicatorResult] = {}

    for name, indicator in INDICATORS.items():
        try:
            slug = build_slug(name, indicator, ref_date)
            if slug is None:
                results[name] = IndicatorResult(
                    name=name,
                    title=indicator["label"],
                    slug="",
                    markets=[],
                    volume=0,
                    fetched_at=datetime.now(),
                    error="Slug 생성 실패",
                )
                continue

            event = _fetch_event(slug)

            # 일간 마켓이 resolved 상태면 다음 날 마켓으로 재시도 (최대 2일)
            slug_type = indicator["slug_type"]
            if slug_type in (SlugType.DAILY, SlugType.DAILY_WITH_YEAR):
                for retry_day in range(1, 3):
                    if event is None or not _is_event_resolved(event):
                        break
                    logger.info("%s: resolved, trying +%d day", name, retry_day)
                    next_date = ref_date + timedelta(days=retry_day)
                    slug = build_slug(name, indicator, next_date)
                    if slug is None:
                        break
                    event = _fetch_event(slug)

            if event is None:
                results[name] = IndicatorResult(
                    name=name,
                    title=indicator["label"],
                    slug=slug or "",
                    markets=[],
                    volume=0,
                    fetched_at=datetime.now(),
                    error=f"이벤트 조회 실패: {slug}",
                )
                continue

            if _is_event_resolved(event):
                results[name] = IndicatorResult(
                    name=name,
                    title=indicator["label"],
                    slug=slug,
                    markets=[],
                    volume=0,
                    fetched_at=datetime.now(),
                    error="마켓 종료됨 (resolved)",
                )
                continue

            markets, volume = _parse_markets(event, indicator)
            title = event.get("title", indicator["label"])

            results[name] = IndicatorResult(
                name=name,
                title=title,
                slug=slug,
                markets=markets,
                volume=volume,
                fetched_at=datetime.now(),
            )
            logger.info("Fetched %s: %d markets, volume=$%.0f", name, len(markets), volume)

        except Exception as e:
            logger.error("Failed to fetch %s: %s", name, e, exc_info=True)
            results[name] = IndicatorResult(
                name=name,
                title=indicator.get("label", name),
                slug="",
                markets=[],
                volume=0,
                fetched_at=datetime.now(),
                error=str(e),
            )

    return results

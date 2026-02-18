"""
Polymarket 비동기 최적화 수집기
================================
3단계 파이프라인으로 과거 데이터를 빠르게 수집한다.

Phase 1: Slug 생성 + 중복 제거 (CPU만)
Phase 2: Gamma API 비동기 이벤트 조회
Phase 3: CLOB API 비동기 가격 히스토리 수집

최적화:
  - 주간/월간/Fed 지표는 동일 토큰을 공유 → 중복 호출 제거 (54% 절감)
  - asyncio + httpx로 동시 50개 요청 → 16분 내 411일 수집

사용법:
    import asyncio
    from polymarket.poly_history_async import collect_range_async

    asyncio.run(collect_range_async(date(2025, 1, 3), date(2026, 2, 17), fidelity=1))
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from polymarket.poly_config import (
    GAMMA_API_BASE,
    INDICATORS,
    SlugType,
)
from polymarket.poly_fetcher import build_slug
from polymarket.poly_history import HISTORY_DIR, extract_token_ids

logger = logging.getLogger(__name__)

CLOB_API_BASE = "https://clob.polymarket.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # 초 (지수 백오프 기본값)


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class SlugTask:
    """하나의 유니크 slug에 대한 수집 작업."""
    indicator_name: str
    slug: str
    slug_type: SlugType
    ref_date: datetime       # slug 생성에 사용된 기준 날짜
    date_range: tuple[date, date]  # 이 slug가 커버하는 날짜 범위
    indicator_config: dict = field(repr=False)


@dataclass
class TokenTask:
    """하나의 유니크 토큰에 대한 CLOB 수집 작업."""
    token_id: str
    outcome_label: str       # "Up", "Down", "Above $68K" 등
    market_question: str
    indicator_name: str
    slug: str
    start_ts: int
    end_ts: int


# ============================================================
# Phase 1: Slug 생성 + 중복 제거
# ============================================================

def _get_week_key(d: date) -> tuple[int, int]:
    """날짜 → (year, iso_week) 키."""
    iso = d.isocalendar()
    return (iso[0], iso[1])


def _get_month_key(d: date) -> tuple[int, int]:
    """날짜 → (year, month) 키."""
    return (d.year, d.month)


def build_unique_slugs(
    start_date: date,
    end_date: date,
    indicators: dict | None = None,
) -> list[SlugTask]:
    """날짜 범위에서 유니크 slug만 추출한다.

    일간 지표: 날마다 고유 slug (중복 없음)
    주간 지표: 같은 주 → 같은 slug (7일분을 1개로)
    월간 지표: 같은 달 → 같은 slug (30일분을 1개로)
    FOMC:     회의별 → 같은 slug (수개월분을 1개로)
    """
    if indicators is None:
        indicators = INDICATORS

    seen_slugs: set[str] = set()
    tasks: list[SlugTask] = []
    current = start_date

    # 주간/월간/FOMC 범위 추적용
    weekly_seen: dict[str, set[tuple[int, int]]] = {}
    monthly_seen: dict[str, set[tuple[int, int]]] = {}
    fomc_seen: dict[str, set[str]] = {}

    while current <= end_date:
        ref_dt = datetime(current.year, current.month, current.day)

        for name, indicator in indicators.items():
            slug_type = indicator["slug_type"]

            # FOMC는 별도 처리 (탐색 기반이라 slug 예측 불가)
            if slug_type == SlugType.FOMC:
                # FOMC는 Phase 2에서 탐색하므로, 월 단위로 1회만 등록
                month_key = _get_month_key(current)
                fomc_key = fomc_seen.setdefault(name, set())
                if month_key not in fomc_key:
                    fomc_key.add(month_key)
                    tasks.append(SlugTask(
                        indicator_name=name,
                        slug="__fomc_search__",
                        slug_type=slug_type,
                        ref_date=ref_dt,
                        date_range=(current, current),
                        indicator_config=indicator,
                    ))
                continue

            slug = build_slug(name, indicator, ref_dt)
            if not slug:
                continue

            # 중복 체크
            if slug_type == SlugType.WEEKLY:
                week_key = _get_week_key(current)
                weekly_set = weekly_seen.setdefault(name, set())
                if week_key in weekly_set:
                    continue
                weekly_set.add(week_key)

            elif slug_type == SlugType.MONTHLY:
                month_key = _get_month_key(current)
                monthly_set = monthly_seen.setdefault(name, set())
                if month_key in monthly_set:
                    continue
                monthly_set.add(month_key)

            # 일간은 항상 유니크 (slug 자체가 날짜 포함)
            task_key = f"{name}:{slug}"
            if task_key in seen_slugs:
                continue
            seen_slugs.add(task_key)

            # 날짜 범위 계산
            if slug_type in (SlugType.DAILY, SlugType.DAILY_WITH_YEAR):
                d_range = (current, current)
            elif slug_type == SlugType.WEEKLY:
                weekday = current.weekday()
                monday = current - timedelta(days=weekday)
                sunday = monday + timedelta(days=6)
                d_range = (max(monday, start_date), min(sunday, end_date))
            elif slug_type == SlugType.MONTHLY:
                month_start = current.replace(day=1)
                next_month = (month_start + timedelta(days=32)).replace(day=1)
                month_end = next_month - timedelta(days=1)
                d_range = (max(month_start, start_date), min(month_end, end_date))
            else:
                d_range = (current, current)

            tasks.append(SlugTask(
                indicator_name=name,
                slug=slug,
                slug_type=slug_type,
                ref_date=ref_dt,
                date_range=d_range,
                indicator_config=indicator,
            ))

        current += timedelta(days=1)

    return tasks


# ============================================================
# Phase 2: Gamma API 비동기 이벤트 조회
# ============================================================

async def _fetch_event_async(
    client: httpx.AsyncClient,
    slug: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Gamma API에서 이벤트 1건을 비동기 조회 (재시도 포함)."""
    url = f"{GAMMA_API_BASE}/events"
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                resp = await client.get(url, params={"slug": slug}, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    return None
                return data[0]
            except (httpx.HTTPError, IndexError, json.JSONDecodeError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("Gamma retry %d for %s: %s (wait %.1fs)",
                                   attempt + 1, slug, e, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("Gamma failed after %d retries for %s: %s",
                                 MAX_RETRIES, slug, e)
                    return None


async def _search_fomc_slug_async(
    client: httpx.AsyncClient,
    template: str,
    ref_date: datetime,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """FOMC 활성 이벤트 slug 탐색 (비동기)."""
    from polymarket.poly_config import FOMC_MONTHS

    month_num = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    start_idx = 0
    for i, m in enumerate(FOMC_MONTHS):
        if month_num.get(m, 0) >= ref_date.month:
            start_idx = i
            break

    for attempt in range(len(FOMC_MONTHS)):
        idx = (start_idx + attempt) % len(FOMC_MONTHS)
        slug = template.format(month=FOMC_MONTHS[idx])
        event = await _fetch_event_async(client, slug, semaphore)
        if event and not event.get("closed"):
            return slug

    return None


async def fetch_events_async(
    slug_tasks: list[SlugTask],
    concurrency: int = 50,
) -> dict[str, dict | None]:
    """유니크 slug들의 이벤트를 병렬 조회.

    Returns: {slug: event_dict} (실패 시 None)
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, dict | None] = {}

    async with httpx.AsyncClient() as client:
        # FOMC와 일반 slug 분리
        fomc_tasks = [t for t in slug_tasks if t.slug == "__fomc_search__"]
        normal_tasks = [t for t in slug_tasks if t.slug != "__fomc_search__"]

        # FOMC 탐색 (순차 — 캐시가 중요하므로)
        fomc_cache: dict[str, str | None] = {}
        for task in fomc_tasks:
            template = task.indicator_config["slug_template"]
            if template not in fomc_cache:
                slug = await _search_fomc_slug_async(
                    client, template, task.ref_date, semaphore)
                fomc_cache[template] = slug
            found_slug = fomc_cache[template]
            if found_slug and found_slug not in results:
                event = await _fetch_event_async(client, found_slug, semaphore)
                results[found_slug] = event
                task.slug = found_slug
            elif found_slug:
                task.slug = found_slug

        # 일반 slug 병렬 조회
        unique_slugs = list({t.slug for t in normal_tasks})
        logger.info("Phase 2: %d 유니크 slug 조회 시작 (동시 %d)",
                     len(unique_slugs), concurrency)

        async def _fetch_one(slug: str):
            event = await _fetch_event_async(client, slug, semaphore)
            results[slug] = event

        batch_size = concurrency * 2
        for i in range(0, len(unique_slugs), batch_size):
            batch = unique_slugs[i:i + batch_size]
            await asyncio.gather(*[_fetch_one(s) for s in batch])
            done = min(i + batch_size, len(unique_slugs))
            logger.info("  Gamma: %d/%d 완료", done, len(unique_slugs))

    return results


# ============================================================
# Phase 3: CLOB API 비동기 가격 히스토리
# ============================================================

def _build_token_tasks(
    slug_tasks: list[SlugTask],
    events: dict[str, dict | None],
) -> list[TokenTask]:
    """이벤트에서 토큰을 추출하고 유니크 TokenTask 목록을 생성."""
    seen_tokens: set[str] = set()
    tasks: list[TokenTask] = []

    for st in slug_tasks:
        slug = st.slug
        if slug == "__fomc_search__" or slug not in events:
            continue
        event = events[slug]
        if event is None:
            continue

        token_groups = extract_token_ids(event)

        # 타임스탬프 범위 계산
        d_start, d_end = st.date_range
        dt_start = datetime(d_start.year, d_start.month, d_start.day,
                            tzinfo=timezone.utc) - timedelta(hours=48)
        dt_end = datetime(d_end.year, d_end.month, d_end.day,
                          tzinfo=timezone.utc) + timedelta(hours=24)
        start_ts = int(dt_start.timestamp())
        end_ts = int(dt_end.timestamp())

        for group in token_groups:
            for item in group["outcomes"]:
                tid = item["token_id"]
                if tid in seen_tokens:
                    continue
                seen_tokens.add(tid)
                tasks.append(TokenTask(
                    token_id=tid,
                    outcome_label=item["label"],
                    market_question=group["market_question"],
                    indicator_name=st.indicator_name,
                    slug=slug,
                    start_ts=start_ts,
                    end_ts=end_ts,
                ))

    return tasks


async def _fetch_price_history_async(
    client: httpx.AsyncClient,
    task: TokenTask,
    fidelity: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """단일 토큰의 가격 히스토리를 비동기 수집 (재시도 포함)."""
    url = f"{CLOB_API_BASE}/prices-history"
    params = {
        "market": task.token_id,
        "startTs": task.start_ts,
        "endTs": task.end_ts,
        "fidelity": fidelity,
    }
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                resp = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                return task.token_id, data.get("history", [])
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("CLOB retry %d for %s...%s: %s",
                                   attempt + 1, task.token_id[:15],
                                   task.token_id[-8:], e)
                    await asyncio.sleep(wait)
                else:
                    logger.error("CLOB failed for token %s...%s: %s",
                                 task.token_id[:15], task.token_id[-8:], e)
                    return task.token_id, []


async def fetch_histories_async(
    token_tasks: list[TokenTask],
    fidelity: int = 1,
    concurrency: int = 50,
) -> dict[str, list[dict]]:
    """유니크 토큰들의 가격 히스토리를 병렬 수집.

    Returns: {token_id: [{"t": ..., "p": ...}, ...]}
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, list[dict]] = {}

    logger.info("Phase 3: %d 유니크 토큰 조회 시작 (동시 %d, fidelity=%d)",
                len(token_tasks), concurrency, fidelity)

    async with httpx.AsyncClient() as client:
        batch_size = concurrency * 2
        for i in range(0, len(token_tasks), batch_size):
            batch = token_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[_fetch_price_history_async(client, t, fidelity, semaphore)
                  for t in batch]
            )
            for tid, history in batch_results:
                results[tid] = history

            done = min(i + batch_size, len(token_tasks))
            logger.info("  CLOB: %d/%d 완료", done, len(token_tasks))

    return results


# ============================================================
# Phase 4: 조립 + 저장
# ============================================================

def assemble_and_save(
    slug_tasks: list[SlugTask],
    events: dict[str, dict | None],
    histories: dict[str, list[dict]],
    start_date: date,
    end_date: date,
    fidelity: int,
) -> int:
    """수집 결과를 날짜별 JSON으로 조립하여 저장.

    기존 poly_history.py와 동일한 파일 형식을 유지한다.

    Returns: 저장된 파일 수
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # slug → (event, slug_task) 매핑
    slug_to_event: dict[str, dict] = {}
    slug_to_tasks: dict[str, list[SlugTask]] = {}
    for st in slug_tasks:
        if st.slug != "__fomc_search__" and st.slug in events and events[st.slug]:
            slug_to_event[st.slug] = events[st.slug]
            slug_to_tasks.setdefault(st.slug, []).append(st)

    # 날짜별로 조립
    saved = 0
    current = start_date
    while current <= end_date:
        ref_dt = datetime(current.year, current.month, current.day)
        result = {
            "date": current.isoformat(),
            "fidelity_minutes": fidelity,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "indicators": {},
        }

        indicators = INDICATORS
        for name, indicator in indicators.items():
            slug_type = indicator["slug_type"]

            # slug 결정
            if slug_type == SlugType.FOMC:
                # FOMC: slug_tasks에서 해당 월의 slug 찾기
                slug = None
                for st in slug_tasks:
                    if st.indicator_name == name and st.slug != "__fomc_search__":
                        slug = st.slug
                        break
            else:
                slug = build_slug(name, indicator, ref_dt)

            if not slug or slug not in slug_to_event:
                result["indicators"][name] = {"error": "이벤트 없음"}
                continue

            event = slug_to_event[slug]
            token_groups = extract_token_ids(event)
            if not token_groups:
                result["indicators"][name] = {"slug": slug, "error": "token ID 없음"}
                continue

            # 최종 가격
            final_prices = {}
            for market in event.get("markets", []):
                outcomes = json.loads(market.get("outcomes", "[]"))
                prices = json.loads(market.get("outcomePrices", "[]"))
                for i in range(min(len(outcomes), len(prices))):
                    final_prices[outcomes[i]] = prices[i]

            # 마켓별 히스토리 조립
            markets_data = []
            for group in token_groups:
                outcomes_history = {}
                for item in group["outcomes"]:
                    tid = item["token_id"]
                    outcomes_history[item["label"]] = histories.get(tid, [])
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

        # 저장
        filename = f"{current.isoformat()}_{fidelity}m.json"
        filepath = HISTORY_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        saved += 1

        current += timedelta(days=1)

    logger.info("Phase 4: %d개 파일 저장 완료", saved)
    return saved


# ============================================================
# 이벤트 캐시 (재실행 시 Phase 2 스킵)
# ============================================================

CACHE_DIR = HISTORY_DIR / "_cache"


def save_events_cache(events: dict[str, dict | None], slug_tasks: list[SlugTask]):
    """Phase 2 결과를 캐시로 저장."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {
        "events": {k: v for k, v in events.items() if v is not None},
        "slug_task_slugs": [t.slug for t in slug_tasks],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = CACHE_DIR / "events_cache.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    logger.info("이벤트 캐시 저장: %s (%d events)", path, len(cache["events"]))


def load_events_cache() -> dict[str, dict] | None:
    """캐시된 이벤트를 로드."""
    path = CACHE_DIR / "events_cache.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    logger.info("이벤트 캐시 로드: %d events", len(cache.get("events", {})))
    return cache.get("events", {})


# ============================================================
# 메인 진입점
# ============================================================

async def collect_range_async(
    start_date: date,
    end_date: date,
    fidelity: int = 1,
    concurrency: int = 50,
    indicators: dict | None = None,
    use_cache: bool = True,
) -> dict:
    """비동기 최적화 수집 메인 함수.

    Returns:
        {"total_days": ..., "saved_files": ..., "errors": ..., "elapsed": ...}
    """
    t0 = time.time()

    # Phase 1: Slug 생성 + 중복 제거
    logger.info("=" * 60)
    logger.info("Phase 1: Slug 생성 + 중복 제거")
    logger.info("=" * 60)
    slug_tasks = build_unique_slugs(start_date, end_date, indicators)

    total_days = (end_date - start_date).days + 1
    total_indicators = len(indicators or INDICATORS)
    naive_calls = total_days * total_indicators
    logger.info("  날짜 범위: %s ~ %s (%d일)", start_date, end_date, total_days)
    logger.info("  유니크 slug: %d개 (중복 제거 전 %d개, %.0f%% 절감)",
                len(slug_tasks), naive_calls,
                (1 - len(slug_tasks) / naive_calls) * 100 if naive_calls else 0)

    # Phase 2: Gamma API 이벤트 조회
    logger.info("=" * 60)
    logger.info("Phase 2: Gamma API 이벤트 조회")
    logger.info("=" * 60)

    events: dict[str, dict | None] = {}
    cached_events = load_events_cache() if use_cache else None
    if cached_events:
        # 캐시에서 있는 것은 재사용
        needed_slugs = []
        for st in slug_tasks:
            if st.slug in cached_events:
                events[st.slug] = cached_events[st.slug]
            elif st.slug != "__fomc_search__":
                needed_slugs.append(st)
        logger.info("  캐시 히트: %d, 신규 조회 필요: %d",
                     len(events), len(needed_slugs))
        if needed_slugs:
            new_events = await fetch_events_async(needed_slugs, concurrency)
            events.update(new_events)
    else:
        events = await fetch_events_async(slug_tasks, concurrency)

    # 캐시 저장
    save_events_cache(events, slug_tasks)

    success_events = sum(1 for v in events.values() if v is not None)
    logger.info("  이벤트 조회 완료: %d 성공 / %d 실패",
                success_events, len(events) - success_events)

    # Phase 3: CLOB API 가격 히스토리
    logger.info("=" * 60)
    logger.info("Phase 3: CLOB API 가격 히스토리 수집")
    logger.info("=" * 60)

    token_tasks = _build_token_tasks(slug_tasks, events)
    logger.info("  유니크 토큰: %d개", len(token_tasks))

    histories = await fetch_histories_async(token_tasks, fidelity, concurrency)

    success_tokens = sum(1 for v in histories.values() if v)
    logger.info("  히스토리 수집 완료: %d 성공 / %d 빈 결과",
                success_tokens, len(histories) - success_tokens)

    # Phase 4: 조립 + 저장
    logger.info("=" * 60)
    logger.info("Phase 4: 조립 + 저장")
    logger.info("=" * 60)

    saved = assemble_and_save(
        slug_tasks, events, histories, start_date, end_date, fidelity)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("수집 완료! %d일 / %d파일 / %.1f분 소요", total_days, saved, elapsed / 60)
    logger.info("=" * 60)

    return {
        "total_days": total_days,
        "unique_slugs": len(slug_tasks),
        "unique_tokens": len(token_tasks),
        "saved_files": saved,
        "elapsed_seconds": elapsed,
    }

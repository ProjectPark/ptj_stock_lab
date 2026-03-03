#!/usr/bin/env python3
"""
Study Step 1 — Polymarket BTC/NDX 장 시작 확률 수집
=====================================================
목적: Study 18/19 (Sector-Rotate / Short-Fundamental M28 게이트 비교) 공용 데이터셋 구축.
     각 날짜 NYSE 장 시작(9:30 ET) 직전 BTC/NDX 상승확률을 CLOB API에서 수집.

출력: data/study/poly_probs.csv
     columns: date, btc_up_prob, ndx_up_prob, btc_resolved, ndx_resolved

columns 설명:
    date          YYYY-MM-DD
    btc_up_prob   NYSE open 시점 BTC 상승확률 (0.0~1.0). 수집 실패 시 빈칸
    ndx_up_prob   NYSE open 시점 NDX 상승확률 (0.0~1.0). 수집 실패 시 빈칸
    btc_resolved  당일 BTC 정산 결과 (1.0=Up, 0.0=Down). 미정산 시 빈칸
    ndx_resolved  당일 NDX 정산 결과 (1.0=Up, 0.0=Down). 미정산 시 빈칸

사용법:
    # 기본: 2024-01-01 ~ 2025-12-31 수집 (이미 수집된 날짜 스킵)
    pyenv shell ptj_stock_lab && python experiments/study_step1_poly_prob_collect.py

    # 날짜 범위 지정
    pyenv shell ptj_stock_lab && python experiments/study_step1_poly_prob_collect.py \\
        --start 2025-01-01 --end 2025-12-31

    # 5일만 수집해서 구조 확인 (먼저 이걸로 테스트)
    pyenv shell ptj_stock_lab && python experiments/study_step1_poly_prob_collect.py --probe

    # NDX 최초 이용 가능 날짜 탐색
    pyenv shell ptj_stock_lab && python experiments/study_step1_poly_prob_collect.py --probe-ndx

    # 이미 수집된 날짜 강제 재수집
    pyenv shell ptj_stock_lab && python experiments/study_step1_poly_prob_collect.py --force

주의:
    - 날짜당 ~2~4초 소요 (CLOB API rate limit)
    - 전체 2년치: 약 730일 × 3초 = ~35분 예상
    - NDX 마켓은 Polymarket 개설 시점 이전 날짜는 수집 불가 → ndx_up_prob 빈칸
    - 이미 수집된 날짜는 자동 스킵 (재실행 안전)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from polymarket.poly_common import fetch_event, extract_final_prices
from polymarket.poly_history import fetch_price_history, extract_token_ids
from polymarket.poly_fetcher import build_slug
from polymarket.poly_config import INDICATORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── 설정 ────────────────────────────────────────────────────────────────────

TARGET_INDICATORS = {
    "btc_up_down": INDICATORS["btc_up_down"],
    "ndx_up_down": INDICATORS["ndx_up_down"],
}

# NYSE open = 9:30 ET.  ET = UTC-5 (EST, Nov~Mar) / UTC-4 (EDT, Mar~Nov).
# 보수적으로 UTC-5 고정 사용 → 9:30 ET = 14:30 UTC
NYSE_OPEN_HOUR_UTC = 14
NYSE_OPEN_MIN_UTC  = 30

# CLOB API rate limit 간격 (초)
API_DELAY = 0.5

# 수집 기본 범위
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END   = date(2025, 12, 31)

# 출력 경로
STUDY_DIR  = _PROJECT_ROOT / "data" / "study"
OUTPUT_CSV = STUDY_DIR / "poly_probs.csv"
CSV_FIELDS = ["date", "btc_up_prob", "ndx_up_prob", "btc_resolved", "ndx_resolved"]


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def nyopen_ts(d: date) -> int:
    """날짜의 NYSE open 유닉스 타임스탬프 (UTC)."""
    dt = datetime(d.year, d.month, d.day,
                  NYSE_OPEN_HOUR_UTC, NYSE_OPEN_MIN_UTC, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def prob_at(series: list[dict], target_ts: int) -> float | None:
    """timeseries에서 target_ts 이전 가장 가까운 확률값 반환.

    series: [{"t": unix_ts, "p": 0.62}, ...]
    target_ts 이전 데이터가 없으면 None 반환.
    """
    best: dict | None = None
    for pt in series:
        t = pt.get("t", 0)
        if t <= target_ts:
            if best is None or t > best["t"]:
                best = pt
    return best["p"] if best else None


def load_done_dates(csv_path: Path) -> set[str]:
    """이미 수집 완료된 날짜 집합 로드."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["date"] for row in reader}


def append_row(csv_path: Path, row: dict) -> None:
    """CSV에 row 한 줄 추가 (헤더는 파일 없을 때만 기록)."""
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def fmt(v: float | None) -> str:
    """확률값 표시용 포맷."""
    return f"{v:.3f}" if v is not None else "None"


# ─── 핵심: 하루 수집 ─────────────────────────────────────────────────────────

def collect_one(target_date: date) -> dict:
    """한 날짜의 BTC/NDX NYSE open 시점 확률 수집.

    Returns:
        CSV_FIELDS 키를 가진 dict.
        확률 수집 실패 시 해당 필드 None.
    """
    ts        = nyopen_ts(target_date)
    start_ts  = ts - 24 * 3600   # 하루 전까지 (이벤트 생성 시점 커버)
    end_ts    = ts + 8  * 3600   # 장 마감 직후까지

    ref_dt = datetime(target_date.year, target_date.month, target_date.day)

    row: dict = {f: None for f in CSV_FIELDS}
    row["date"] = target_date.isoformat()

    for key, indicator in TARGET_INDICATORS.items():
        slug = build_slug(key, indicator, ref_dt)
        if not slug:
            logger.debug("%s | %s: slug 생성 실패", target_date, key)
            continue

        event = fetch_event(slug)
        time.sleep(API_DELAY)

        if not event:
            logger.debug("%s | %s: 이벤트 없음 (slug=%s)", target_date, key, slug)
            continue

        # 정산 결과
        final_prices = extract_final_prices(event)

        # CLOB token ID 추출
        token_groups = extract_token_ids(event)
        if not token_groups:
            logger.debug("%s | %s: token ID 없음", target_date, key)
            continue

        # BINARY 마켓 → 첫 번째 그룹의 Up/Down 히스토리 수집
        group            = token_groups[0]
        outcomes_hist: dict[str, list] = {}

        for item in group["outcomes"]:
            hist = fetch_price_history(item["token_id"], start_ts, end_ts, fidelity=5)
            outcomes_hist[item["label"]] = hist
            time.sleep(API_DELAY)

        # NYSE open 직전 Up 확률
        up_series = outcomes_hist.get("Up", [])
        up_prob   = prob_at(up_series, ts)

        # 정산 결과 → 0.0 / 1.0
        up_final = final_prices.get("Up")
        resolved = float(up_final) if up_final is not None else None

        if key == "btc_up_down":
            row["btc_up_prob"] = round(up_prob, 4) if up_prob is not None else None
            row["btc_resolved"] = resolved
        else:  # ndx_up_down
            row["ndx_up_prob"] = round(up_prob, 4) if up_prob is not None else None
            row["ndx_resolved"] = resolved

    return row


# ─── NDX 최초 가용 날짜 탐색 ────────────────────────────────────────────────

def probe_ndx_availability(search_start: date = date(2026, 1, 1),
                            max_days: int = 120) -> None:
    """NDX 마켓이 최초로 존재하는 날짜를 역순으로 탐색.

    search_start부터 오늘까지 날짜를 순서대로 체크,
    첫 번째 이벤트 발견 시 날짜 출력 후 종료.
    """
    logger.info("NDX 최초 이용 가능 날짜 탐색 시작 (%s ~)", search_start)
    indicator = TARGET_INDICATORS["ndx_up_down"]

    current = search_start
    today   = date.today()
    checked = 0

    while current <= today and checked < max_days:
        ref_dt = datetime(current.year, current.month, current.day)
        slug   = build_slug("ndx_up_down", indicator, ref_dt)

        if slug:
            event = fetch_event(slug)
            time.sleep(API_DELAY)
            if event:
                logger.info("✅ NDX 최초 발견: %s (slug=%s)", current, slug)
                return
            else:
                logger.info("  %s → 이벤트 없음 (slug=%s)", current, slug)

        current += timedelta(days=1)
        checked += 1

    logger.info("탐색 완료: NDX 마켓이 %s ~ %d일 범위에서 발견되지 않음",
                search_start, max_days)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket BTC/NDX 장 시작 확률 수집")
    p.add_argument("--start", default=DEFAULT_START.isoformat(),
                   help=f"수집 시작 날짜 YYYY-MM-DD (기본: {DEFAULT_START})")
    p.add_argument("--end",   default=DEFAULT_END.isoformat(),
                   help=f"수집 종료 날짜 YYYY-MM-DD (기본: {DEFAULT_END})")
    p.add_argument("--probe", action="store_true",
                   help="첫 5일만 수집해서 데이터 구조/API 동작 확인")
    p.add_argument("--probe-ndx", action="store_true",
                   help="NDX 마켓 최초 이용 가능 날짜 탐색")
    p.add_argument("--force", action="store_true",
                   help="이미 수집된 날짜도 강제 재수집")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # NDX 탐색 모드
    if args.probe_ndx:
        probe_ndx_availability()
        return

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    done      = set() if args.force else load_done_dates(OUTPUT_CSV)
    total     = (end - start).days + 1
    remaining = sum(1 for d in (start + timedelta(i) for i in range(total))
                    if d.isoformat() not in done)

    logger.info("수집 범위: %s ~ %s (%d일)", start, end, total)
    logger.info("이미 완료: %d일  |  수집 예정: %d일", len(done), remaining)
    if remaining > 0:
        logger.info("예상 소요: ~%d분", max(1, remaining * 3 // 60))

    collected = 0
    skipped   = 0
    failed    = 0
    probe_cnt = 0

    current = start
    while current <= end:
        date_str = current.isoformat()

        if date_str in done and not args.force:
            skipped += 1
            current += timedelta(days=1)
            continue

        try:
            logger.info("▶ %s 수집 중...", date_str)
            row = collect_one(current)
            append_row(OUTPUT_CSV, row)
            collected += 1

            logger.info(
                "  BTC prob=%s resolved=%s  |  NDX prob=%s resolved=%s",
                fmt(row["btc_up_prob"]),  row["btc_resolved"],
                fmt(row["ndx_up_prob"]),  row["ndx_resolved"],
            )

            # --probe: 5일 후 종료
            if args.probe:
                probe_cnt += 1
                if probe_cnt >= 5:
                    logger.info("─ probe 완료 (5일). 구조 확인 후 --probe 없이 실행하면 전체 수집. ─")
                    break

        except KeyboardInterrupt:
            logger.info("중단됨. 이미 수집된 데이터는 보존됩니다.")
            break
        except Exception as e:
            logger.error("%s 수집 실패: %s", date_str, e, exc_info=True)
            failed += 1

        current += timedelta(days=1)

    # ─── 완료 요약 ────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("완료: 수집=%d  스킵=%d  실패=%d", collected, skipped, failed)
    logger.info("저장: %s", OUTPUT_CSV)

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1  # 헤더 제외
        logger.info("총 누적 레코드: %d일", rows)

        # BTC/NDX 수집 현황 요약
        btc_ok, ndx_ok = 0, 0
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["btc_up_prob"]:
                    btc_ok += 1
                if r["ndx_up_prob"]:
                    ndx_ok += 1
        logger.info("BTC prob 수집 성공: %d일  |  NDX prob 수집 성공: %d일",
                    btc_ok, ndx_ok)


if __name__ == "__main__":
    main()

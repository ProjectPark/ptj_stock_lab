#!/usr/bin/env python3
"""
Polymarket 과거 데이터 수집 스크립트
====================================

사용법:
    # 단일 날짜, 1분봉
    python collect_poly_history.py --date 2026-02-15 --fidelity 1

    # 단일 날짜, 5분봉
    python collect_poly_history.py --date 2026-02-15 --fidelity 5

    # 날짜 범위, 1분봉
    python collect_poly_history.py --start 2026-02-01 --end 2026-02-15 --fidelity 1

    # 특정 지표만 수집
    python collect_poly_history.py --date 2026-02-15 --fidelity 1 --indicators btc_up_down,ndx_up_down

    # 저장된 파일 목록 확인
    python collect_poly_history.py --list

    # 저장된 파일 요약 보기
    python collect_poly_history.py --show 2026-02-15 --fidelity 1
"""
import argparse
import json
import logging
import sys
from datetime import date, datetime

from polymarket.poly_config import INDICATORS
from polymarket.poly_history import (
    collect_history_for_date,
    collect_history_range,
    list_available_dates,
    load_history,
    save_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_collect_single(args):
    """단일 날짜 수집."""
    indicators = None
    if args.indicators:
        names = [n.strip() for n in args.indicators.split(",")]
        indicators = {n: INDICATORS[n] for n in names if n in INDICATORS}
        if not indicators:
            logger.error("유효한 지표가 없습니다: %s", args.indicators)
            logger.info("사용 가능: %s", ", ".join(INDICATORS.keys()))
            return

    target = parse_date(args.date)
    logger.info("수집 시작: %s (fidelity=%d분)", target, args.fidelity)

    data = collect_history_for_date(target, args.fidelity, indicators)
    filepath = save_history(data, args.fidelity)

    # 요약 출력
    print(f"\n{'='*60}")
    print(f"수집 완료: {target} ({args.fidelity}분봉)")
    print(f"저장: {filepath}")
    print(f"{'='*60}")
    for name, ind_data in data["indicators"].items():
        if "error" in ind_data:
            print(f"  ✗ {name}: {ind_data['error']}")
        else:
            total_points = sum(
                len(h)
                for m in ind_data.get("markets", [])
                for h in m.get("outcomes", {}).values()
            )
            print(f"  ✓ {name}: {total_points} data points, "
                  f"resolved={ind_data.get('resolved', '?')}")


def cmd_collect_range(args):
    """날짜 범위 수집."""
    indicators = None
    if args.indicators:
        names = [n.strip() for n in args.indicators.split(",")]
        indicators = {n: INDICATORS[n] for n in names if n in INDICATORS}

    start = parse_date(args.start)
    end = parse_date(args.end)

    logger.info("범위 수집 시작: %s ~ %s (fidelity=%d분)", start, end, args.fidelity)
    results = collect_history_range(start, end, args.fidelity, indicators)

    print(f"\n{'='*60}")
    print(f"수집 완료: {start} ~ {end} ({len(results)}일)")
    print(f"{'='*60}")
    for data in results:
        ok = sum(1 for v in data["indicators"].values() if "error" not in v)
        fail = sum(1 for v in data["indicators"].values() if "error" in v)
        print(f"  {data['date']}: {ok} 성공, {fail} 실패")


def cmd_list(args):
    """저장된 파일 목록."""
    fidelity = args.fidelity if args.fidelity else None
    files = list_available_dates(fidelity)
    if not files:
        print("저장된 히스토리 파일이 없습니다.")
        return
    print(f"{'날짜':<14} {'해상도':<10} {'파일'}")
    print("-" * 50)
    for f in files:
        print(f"{f['date']:<14} {f['fidelity']}분{'':<6} {f['file']}")


def cmd_show(args):
    """저장된 파일 요약."""
    target = parse_date(args.show)
    data = load_history(target, args.fidelity)
    if not data:
        print(f"파일 없음: {target} ({args.fidelity}분봉)")
        return

    print(f"\n{'='*60}")
    print(f"{data['date']} — {data['fidelity_minutes']}분봉")
    print(f"수집 시각: {data.get('collected_at', '?')}")
    print(f"{'='*60}")

    for name, ind_data in data["indicators"].items():
        if "error" in ind_data:
            print(f"\n  ✗ {name}: {ind_data['error']}")
            continue

        print(f"\n  ✓ {name}")
        print(f"    slug: {ind_data.get('slug', '?')}")
        print(f"    title: {ind_data.get('title', '?')}")
        print(f"    resolved: {ind_data.get('resolved', '?')}")
        print(f"    final_prices: {ind_data.get('final_prices', {})}")

        for m in ind_data.get("markets", []):
            print(f"    market: {m['question'][:60]}")
            for label, history in m.get("outcomes", {}).items():
                if history:
                    first_t = datetime.utcfromtimestamp(history[0]["t"])
                    last_t = datetime.utcfromtimestamp(history[-1]["t"])
                    first_p = history[0]["p"]
                    last_p = history[-1]["p"]
                    print(f"      {label}: {len(history)} points "
                          f"({first_t:%m/%d %H:%M} → {last_t:%m/%d %H:%M}) "
                          f"p={first_p} → {last_p}")


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket 과거 데이터 수집기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fidelity", type=int, default=1,
                        help="데이터 해상도 (분). 1=1분봉, 5=5분봉 (기본: 1)")
    parser.add_argument("--indicators", type=str, default=None,
                        help="수집할 지표 (쉼표 구분). 예: btc_up_down,ndx_up_down")

    # 모드 선택
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=str, help="단일 날짜 수집 (YYYY-MM-DD)")
    group.add_argument("--start", type=str, help="범위 시작 날짜 (--end와 함께)")
    group.add_argument("--list", action="store_true", help="저장된 파일 목록")
    group.add_argument("--show", type=str, help="저장된 파일 요약 (YYYY-MM-DD)")

    parser.add_argument("--end", type=str, help="범위 종료 날짜")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.show:
        cmd_show(args)
    elif args.date:
        cmd_collect_single(args)
    elif args.start:
        if not args.end:
            parser.error("--start에는 --end가 필요합니다")
        cmd_collect_range(args)


if __name__ == "__main__":
    main()

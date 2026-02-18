#!/usr/bin/env python3
"""
Polymarket 비동기 최적화 수집 스크립트
=======================================

사용법:
    # 전체 기간 1분봉 (기본)
    python collect_poly_history_async.py --start 2025-01-03 --end 2026-02-17

    # 1시간봉, 동시 30
    python collect_poly_history_async.py --start 2025-01-03 --end 2026-02-17 --fidelity 60 --concurrency 30

    # 특정 지표만
    python collect_poly_history_async.py --start 2025-01-03 --end 2026-02-17 --indicators btc_up_down,fed_decision

    # 캐시 무시 (이벤트 재조회)
    python collect_poly_history_async.py --start 2025-01-03 --end 2026-02-17 --no-cache

    # 테스트 (3일만)
    python collect_poly_history_async.py --start 2026-02-13 --end 2026-02-15 --fidelity 5
"""
import argparse
import asyncio
import logging
import sys
from datetime import date, datetime

from polymarket.poly_config import INDICATORS
from polymarket.poly_history_async import collect_range_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


async def main():
    parser = argparse.ArgumentParser(
        description="Polymarket 비동기 최적화 수집기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", type=str, required=True,
                        help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--fidelity", type=int, default=1,
                        help="데이터 해상도 (분). 1=1분봉, 5=5분봉, 60=1시간봉 (기본: 1)")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="동시 API 요청 수 (기본: 50)")
    parser.add_argument("--indicators", type=str, default=None,
                        help="수집할 지표 (쉼표 구분). 예: btc_up_down,ndx_up_down")
    parser.add_argument("--no-cache", action="store_true",
                        help="이벤트 캐시 무시 (전체 재조회)")

    args = parser.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    indicators = None
    if args.indicators:
        names = [n.strip() for n in args.indicators.split(",")]
        indicators = {n: INDICATORS[n] for n in names if n in INDICATORS}
        if not indicators:
            logger.error("유효한 지표가 없습니다: %s", args.indicators)
            logger.info("사용 가능: %s", ", ".join(INDICATORS.keys()))
            sys.exit(1)

    total_days = (end - start).days + 1
    print(f"\n{'='*60}")
    print(f"Polymarket 비동기 수집기")
    print(f"{'='*60}")
    print(f"  기간:     {start} ~ {end} ({total_days}일)")
    print(f"  해상도:   {args.fidelity}분봉")
    print(f"  동시성:   {args.concurrency}")
    print(f"  캐시:     {'사용안함' if args.no_cache else '사용'}")
    if indicators:
        print(f"  지표:     {', '.join(indicators.keys())}")
    else:
        print(f"  지표:     전체 ({len(INDICATORS)}개)")
    print(f"{'='*60}\n")

    result = await collect_range_async(
        start_date=start,
        end_date=end,
        fidelity=args.fidelity,
        concurrency=args.concurrency,
        indicators=indicators,
        use_cache=not args.no_cache,
    )

    print(f"\n{'='*60}")
    print(f"수집 완료!")
    print(f"{'='*60}")
    print(f"  총 일수:         {result['total_days']}일")
    print(f"  유니크 slug:     {result['unique_slugs']}개")
    print(f"  유니크 토큰:     {result['unique_tokens']}개")
    print(f"  저장 파일:       {result['saved_files']}개")
    print(f"  소요 시간:       {result['elapsed_seconds']:.1f}초 ({result['elapsed_seconds']/60:.1f}분)")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

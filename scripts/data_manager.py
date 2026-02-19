#!/usr/bin/env python3
"""
시뮬레이션 데이터 관리 도구
============================
캐시 상태 조회, 검증, 정리를 위한 CLI.

사용법:
    python scripts/data_manager.py status          # 전체 상태
    python scripts/data_manager.py status alpaca    # Alpaca만
    python scripts/data_manager.py validate         # 캐시 무결성 검증
    python scripts/data_manager.py clean --dry-run  # 정리 대상 확인
    python scripts/data_manager.py clean            # 실제 정리
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from fetchers.cache_meta import load_meta


# ============================================================
# 캐시 레지스트리: 관리 대상 캐시 파일 정의
# ============================================================
CACHE_REGISTRY = {
    "alpaca": {
        "label": "Alpaca (주식 분봉/일봉)",
        "files": {
            "1min_v1": {
                "path": config.OHLCV_DIR / "backtest_1min.parquet",
                "desc": "1분봉 v1 (TICKERS)",
                "timeframe": "1min",
            },
            "1min_v2": {
                "path": config.OHLCV_DIR / "backtest_1min_v2.parquet",
                "desc": "1분봉 v2 (TICKERS_V2 + 티커매핑)",
                "timeframe": "1min",
            },
            "5min_v1": {
                "path": config.OHLCV_DIR / "backtest_5min.parquet",
                "desc": "5분봉 v1 (TICKERS)",
                "timeframe": "5min",
            },
            "daily_soxx_iren": {
                "path": config.DAILY_DIR / "soxx_iren_daily.parquet",
                "desc": "일봉 SOXX/IREN",
                "timeframe": "daily",
            },
            "daily_market": {
                "path": config.DAILY_DIR / "market_daily.parquet",
                "desc": "일봉 전체 (yfinance)",
                "timeframe": "daily",
            },
        },
    },
    "fx": {
        "label": "FX (환율)",
        "files": {
            "usdkrw_hourly": {
                "path": config.FX_DIR / "usdkrw_hourly.parquet",
                "desc": "USD/KRW 1시간봉",
                "timeframe": "1h",
            },
        },
    },
    "polymarket": {
        "label": "Polymarket (예측 확률)",
        "files": {
            "history": {
                "path": config.POLY_DATA_DIR,
                "desc": "일별 확률 히스토리 (JSON)",
                "timeframe": "daily",
                "is_dir": True,
            },
            "events_cache": {
                "path": config.POLY_DATA_DIR / "_cache" / "events_cache.json",
                "desc": "이벤트 캐시 (Phase 2 결과)",
                "timeframe": "-",
            },
        },
    },
}


# ============================================================
# 상태 조회
# ============================================================
def _file_info(path: Path) -> dict:
    """파일 상태 정보."""
    if not path.exists():
        return {"exists": False}

    stat = path.stat()
    size_kb = stat.st_size / 1024
    mtime = datetime.fromtimestamp(stat.st_mtime)
    age_hours = (datetime.now() - mtime).total_seconds() / 3600

    info = {
        "exists": True,
        "size_kb": round(size_kb, 1),
        "size_display": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB",
        "modified": mtime.strftime("%Y-%m-%d %H:%M"),
        "age_hours": round(age_hours, 1),
    }

    # parquet 메타 추가
    if path.suffix == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            info["rows"] = len(df)
            info["columns"] = list(df.columns)
            if "symbol" in df.columns:
                info["tickers"] = sorted(df["symbol"].unique().tolist())
                info["ticker_count"] = len(info["tickers"])
            if "date" in df.columns:
                info["date_range"] = f"{df['date'].min()} ~ {df['date'].max()}"
            elif "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"])
                info["date_range"] = f"{ts.dt.date.min()} ~ {ts.dt.date.max()}"
        except Exception as e:
            info["read_error"] = str(e)

    # cache_meta 추가
    meta = load_meta(path)
    if meta:
        info["meta"] = {
            "tickers": meta.tickers,
            "period": f"{meta.start_date} ~ {meta.end_date}",
            "timeframe": f"{meta.timeframe_minutes}min",
            "row_count": meta.row_count,
            "created_at": meta.created_at,
        }

    return info


def _dir_info(path: Path) -> dict:
    """디렉토리 상태 정보 (Polymarket history 등)."""
    if not path.exists():
        return {"exists": False}

    # 직접 하위 + 서브디렉토리 JSON 모두 탐색
    json_files = sorted(path.glob("*.json")) + sorted(path.glob("*/*.json"))
    if not json_files:
        return {"exists": True, "file_count": 0, "size_display": "0 KB"}

    # 날짜 추출
    dates = []
    fidelities = set()
    for f in json_files:
        parts = f.stem.split("_")
        if len(parts) == 2:
            dates.append(parts[0])
            fidelities.add(parts[1])

    total_size = sum(f.stat().st_size for f in json_files)

    return {
        "exists": True,
        "file_count": len(json_files),
        "size_display": f"{total_size/1024/1024:.1f} MB" if total_size > 1024*1024 else f"{total_size/1024:.1f} KB",
        "date_range": f"{min(dates)} ~ {max(dates)}" if dates else "-",
        "fidelities": sorted(fidelities) if fidelities else [],
        "latest_file": json_files[-1].name if json_files else "-",
    }


def cmd_status(args):
    """캐시 상태 출력."""
    sources = args.source if args.source else list(CACHE_REGISTRY.keys())

    print()
    print("=" * 65)
    print("  시뮬레이션 데이터 상태")
    print("=" * 65)

    for source_key in sources:
        if source_key not in CACHE_REGISTRY:
            print(f"\n  [!] 알 수 없는 소스: {source_key}")
            continue

        source = CACHE_REGISTRY[source_key]
        print(f"\n  [{source['label']}]")
        print(f"  {'─' * 60}")

        for name, file_def in source["files"].items():
            path = file_def["path"]
            is_dir = file_def.get("is_dir", False)

            if is_dir:
                info = _dir_info(path)
            else:
                info = _file_info(path)

            if not info["exists"]:
                print(f"    {name:20s}  ✗ 없음")
                continue

            if is_dir:
                print(f"    {name:20s}  ✓ {info['file_count']}개 파일  {info['size_display']}")
                if info.get("date_range"):
                    print(f"    {'':20s}    기간: {info['date_range']}")
                if info.get("fidelities"):
                    print(f"    {'':20s}    해상도: {', '.join(info['fidelities'])}")
            else:
                size = info["size_display"]
                mod = info["modified"]
                age = info["age_hours"]
                age_str = f"{age:.0f}h" if age < 24 else f"{age/24:.0f}d"

                rows_str = f"  {info['rows']:,} rows" if "rows" in info else ""
                tickers_str = f"  {info.get('ticker_count', '?')}종목" if "tickers" in info else ""

                print(f"    {name:20s}  ✓ {size:>10s}  {mod}  ({age_str} ago){rows_str}{tickers_str}")

                if "date_range" in info:
                    print(f"    {'':20s}    기간: {info['date_range']}")
                if "tickers" in info:
                    tickers = info["tickers"]
                    if len(tickers) <= 8:
                        print(f"    {'':20s}    종목: {', '.join(tickers)}")
                    else:
                        print(f"    {'':20s}    종목: {', '.join(tickers[:6])}, ... +{len(tickers)-6}")

    print()
    print("=" * 65)


# ============================================================
# 검증
# ============================================================
def cmd_validate(args):
    """캐시 파일 무결성 검증."""
    print()
    print("=" * 65)
    print("  캐시 무결성 검증")
    print("=" * 65)

    issues = []

    for source_key, source in CACHE_REGISTRY.items():
        for name, file_def in source["files"].items():
            path = file_def["path"]
            is_dir = file_def.get("is_dir", False)

            if is_dir or not path.exists():
                continue

            label = f"{source_key}/{name}"

            # 1) 파일 읽기 가능?
            if path.suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    if df.empty:
                        issues.append((label, "WARN", "빈 파일"))
                        continue
                except Exception as e:
                    issues.append((label, "ERROR", f"읽기 실패: {e}"))
                    continue

                # 2) 필수 컬럼?
                if "symbol" not in df.columns:
                    issues.append((label, "WARN", "'symbol' 컬럼 없음"))
                if "timestamp" not in df.columns:
                    issues.append((label, "WARN", "'timestamp' 컬럼 없음"))

                # 3) meta.json 일치?
                meta = load_meta(path)
                if meta:
                    if meta.row_count != len(df):
                        issues.append((label, "WARN",
                            f"메타 row_count({meta.row_count}) != 실제({len(df)})"))
                    if "symbol" in df.columns:
                        actual_tickers = set(df["symbol"].unique())
                        meta_tickers = set(meta.tickers)
                        if actual_tickers != meta_tickers:
                            diff = actual_tickers.symmetric_difference(meta_tickers)
                            issues.append((label, "WARN",
                                f"메타 티커 불일치: {diff}"))

                print(f"  ✓ {label:35s}  {len(df):>10,} rows  OK")

            elif path.suffix == ".json":
                try:
                    with open(path) as f:
                        json.load(f)
                    print(f"  ✓ {label:35s}  OK")
                except Exception as e:
                    issues.append((label, "ERROR", f"JSON 파싱 실패: {e}"))

    if issues:
        print(f"\n  문제 발견: {len(issues)}건")
        print(f"  {'─' * 58}")
        for label, severity, msg in issues:
            icon = "⚠" if severity == "WARN" else "✗"
            print(f"  {icon} [{severity}] {label}: {msg}")
    else:
        print(f"\n  모든 캐시 정상!")

    print()
    print("=" * 65)


# ============================================================
# 정리
# ============================================================
def cmd_clean(args):
    """오래된/고아 캐시 정리."""
    dry_run = args.dry_run
    max_age_days = args.max_age or 30

    print()
    print("=" * 65)
    print(f"  캐시 정리 {'(DRY RUN)' if dry_run else ''}")
    print(f"  기준: {max_age_days}일 이상 된 파일")
    print("=" * 65)

    cutoff = datetime.now() - timedelta(days=max_age_days)
    targets = []

    # OHLCV 디렉토리의 parquet 중 레지스트리에 없는 것 = 고아 파일
    known_paths = set()
    for source in CACHE_REGISTRY.values():
        for file_def in source["files"].values():
            known_paths.add(file_def["path"])

    for d in [config.OHLCV_DIR, config.DAILY_DIR, config.FX_DIR]:
        if not d.exists():
            continue
        for f in d.glob("*.parquet"):
            if f not in known_paths:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    targets.append(("orphan", f, f.stat().st_size))

    # 오래된 meta.json
    for d in [config.OHLCV_DIR, config.DAILY_DIR, config.FX_DIR]:
        if not d.exists():
            continue
        for f in d.glob("*.meta.json"):
            parquet = f.with_suffix("").with_suffix(".parquet")
            if not parquet.exists():
                targets.append(("orphan_meta", f, f.stat().st_size))

    if not targets:
        print("\n  정리 대상 없음.")
    else:
        total_size = sum(t[2] for t in targets)
        print(f"\n  정리 대상: {len(targets)}개 파일 ({total_size/1024:.1f} KB)")
        for reason, path, size in targets:
            tag = "고아" if "orphan" in reason else "오래됨"
            action = "삭제 예정" if not dry_run else "삭제 대상"
            print(f"    [{tag}] {path.name} ({size/1024:.1f} KB) — {action}")

            if not dry_run:
                path.unlink()
                print(f"           → 삭제 완료")

    print()
    print("=" * 65)


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="시뮬레이션 데이터 관리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/data_manager.py status              전체 상태
  python scripts/data_manager.py status alpaca        Alpaca만
  python scripts/data_manager.py status polymarket    Polymarket만
  python scripts/data_manager.py validate             무결성 검증
  python scripts/data_manager.py clean --dry-run      정리 대상 확인
  python scripts/data_manager.py clean --max-age 14   14일 기준 정리
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # status
    p_status = sub.add_parser("status", help="캐시 상태 조회")
    p_status.add_argument("source", nargs="*", help="소스 필터 (alpaca, fx, polymarket)")

    # validate
    sub.add_parser("validate", help="캐시 무결성 검증")

    # clean
    p_clean = sub.add_parser("clean", help="오래된 캐시 정리")
    p_clean.add_argument("--dry-run", action="store_true", help="실제 삭제 없이 대상만 출력")
    p_clean.add_argument("--max-age", type=int, default=30, help="삭제 기준 일수 (기본 30일)")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        # 인자 없이 실행 → 전체 상태 출력
        args.source = []
        cmd_status(args)


if __name__ == "__main__":
    main()

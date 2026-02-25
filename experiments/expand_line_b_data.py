"""
Line B 데이터 확대 스크립트
============================
비활성 전략에 필요한 추가 종목/메타데이터를 수집한다.

실행:
    pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py
    pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --dry-run
    pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-1min
    pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-meta
    pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-crypto
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

import config
from fetchers.alpaca_fetcher import fetch_bars, get_alpaca_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# 대상 종목 정의
# ============================================================

# Group A: 1분봉 OHLCV가 필요한 신규 종목
# backtest_1min_v2.parquet 에 없는 종목들
GROUP_A_TICKERS = {
    # jab_soxl 전략 조건 (SOXX 실시간 데이터 필요)
    "SOXX": "iShares Semiconductor ETF",
    # jab_tsll 전략 (TSLA, TSLL)
    "TSLA": "Tesla",
    "TSLL": "Direxion Daily TSLA Bull 2x",
    # vix_gold 전략 (VIX 대리: UVXY)
    "UVXY": "ProShares Ultra VIX Short-Term Futures",
    # vix_gold 전략 직접 종목
    "IAU": "iShares Gold Trust",
    "GDXU": "U.S. Global GO GOLD and Precious Metal Miners ETF",
    # jab_etq 전략
    "ETQ": "Graniteshares 2x Short ETH ETP",
    # bank_conditional 전략 (BAC 감시)
    "BAC": "Bank of America",
    "JPM": "JPMorgan Chase",
    "HSBC": "HSBC Holdings",
    "WFC": "Wells Fargo",
    "C": "Citigroup",
    # sector_rotate 전략 (SOXX 포함됨, RBC 추가)
    "RBC": "RBC Bearings",
    # reit_risk 전략
    "VNQ": "Vanguard Real Estate ETF",
    # bear_regime 전략 (인버스 ETF)
    "SOXS": "Direxion Daily Semiconductor Bear 3x",
    "BITI": "ProShares Short Bitcoin Strategy ETF",
    # bargain_buy 전략 추가 종목 (기존에 없는 것만)
    "CONZ": "CONL inverse (N/A - placeholder)",
    "IREZ": "IRE inverse (N/A - placeholder)",
    "HOOZ": "HOOD inverse (N/A - placeholder)",
    "MSTZ": "MSTU inverse (N/A - placeholder)",
}

# Alpaca에서 지원 가능한 실제 종목 (역방향 ETF 제외)
GROUP_A_ALPACA_FETCHABLE = [
    "SOXX", "TSLA", "TSLL", "IAU", "GDXU",
    "BAC", "JPM", "WFC", "C",
    "VNQ", "SOXS", "BITI",
    "UVXY",
]

# Alpaca에서 지원 불확실한 종목 (ETQ는 ETP, HSBC/RBC는 확인 필요)
GROUP_A_VERIFY_NEEDED = ["ETQ", "HSBC", "RBC"]

# 이미 별도 파일로 존재하는 종목
GROUP_A_ALREADY_HAVE = ["SOXX"]  # soxx_1min.parquet

# Group B: 히스토리 메타데이터 (yfinance 일봉으로 계산)
# bargain_buy: 3년 최고가
# sector_rotate: 1년 저가
# short_macro: NDX/SPX ATH
# reit_risk: VNQ 120일 이동평균
GROUP_B_META_TICKERS = {
    "bargain_buy_3y_high": ["CONL", "SOXL", "AMDL", "NVDL", "ROBN", "ETHU", "BRKU"],
    "sector_rotate_1y_low": ["BITU", "SOXX", "ROBN", "GLD"],
    "reit_risk_vnq_ma120": ["VNQ"],
    "short_macro_ath": ["QQQ", "SPY"],  # NDX/SPX proxy
}

# Group C: 크립토 스팟 데이터 (jab_bitu 전략)
GROUP_C_CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
}

# Group D: Polymarket 확장 지표
# bear_regime_long 전략: btc_monthly_dip 확률
# poly_config.py의 btc_monthly 지표로 커버 가능 (기존 인프라)
GROUP_D_POLY = {
    "btc_monthly_dip": "btc_monthly 지표의 Dip 컴포넌트 (기존 poly_config.py에 존재)",
}


# ============================================================
# Step 1: 기존 데이터 커버리지 분석
# ============================================================

def analyze_coverage() -> dict:
    """현재 데이터 커버리지를 분석하여 부족한 부분을 식별한다."""
    log.info("=== 데이터 커버리지 분석 ===")

    # 현재 backtest_1min_v2.parquet 종목
    ohlcv_path = config.OHLCV_DIR / "backtest_1min_v2.parquet"
    if ohlcv_path.exists():
        df = pd.read_parquet(ohlcv_path)
        current_tickers = sorted(df["symbol"].unique().tolist())
        date_range = (str(df["date"].min()), str(df["date"].max()))
    else:
        current_tickers = []
        date_range = ("N/A", "N/A")

    # soxx_1min.parquet 별도 존재
    soxx_path = config.OHLCV_DIR / "soxx_1min.parquet"
    soxx_available = soxx_path.exists()

    # strategy_daily_3y.parquet 종목 (일봉)
    daily_path = config.DAILY_DIR / "strategy_daily_3y.parquet"
    if daily_path.exists():
        ddf = pd.read_parquet(daily_path)
        daily_tickers = sorted(ddf["symbol"].unique().tolist())
        daily_date_range = (str(ddf["date"].min()), str(ddf["date"].max()))
    else:
        daily_tickers = []
        daily_date_range = ("N/A", "N/A")

    # 히스토리 메타 파일
    meta_path = config.META_DIR / "history_metadata.json"
    meta_exists = meta_path.exists()

    result = {
        "current_1min_tickers": current_tickers,
        "current_1min_count": len(current_tickers),
        "current_1min_date_range": date_range,
        "soxx_1min_available": soxx_available,
        "daily_3y_tickers": daily_tickers,
        "daily_3y_count": len(daily_tickers),
        "daily_3y_date_range": daily_date_range,
        "history_metadata_exists": meta_exists,
    }

    log.info(f"1분봉 종목 수: {result['current_1min_count']} ({', '.join(current_tickers)})")
    log.info(f"1분봉 날짜 범위: {date_range[0]} ~ {date_range[1]}")
    log.info(f"SOXX 1분봉 별도 파일: {soxx_available}")
    log.info(f"일봉 종목 수 (3년): {result['daily_3y_count']}")
    log.info(f"히스토리 메타 존재: {meta_exists}")

    return result


# ============================================================
# Step 2: Alpaca 1분봉 추가 수집
# ============================================================

def fetch_additional_1min(
    tickers: list[str] | None = None,
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 20),
    dry_run: bool = False,
) -> dict:
    """Alpaca API로 추가 종목 1분봉 데이터를 수집한다."""
    log.info("=== Alpaca 추가 1분봉 수집 ===")

    if tickers is None:
        # SOXX는 이미 soxx_1min.parquet에 존재
        tickers = [t for t in GROUP_A_ALPACA_FETCHABLE if t not in GROUP_A_ALREADY_HAVE]

    results = {}

    for ticker in tickers:
        cache_path = config.OHLCV_DIR / f"{ticker.lower()}_1min.parquet"

        if cache_path.exists():
            df_existing = pd.read_parquet(cache_path)
            log.info(f"  [SKIP] {ticker}: 이미 존재 ({len(df_existing):,} rows)")
            results[ticker] = {"status": "exists", "rows": len(df_existing), "path": str(cache_path)}
            continue

        if dry_run:
            log.info(f"  [DRY-RUN] {ticker}: fetch_bars() 호출 예정 → {cache_path.name}")
            results[ticker] = {"status": "dry_run", "path": str(cache_path)}
            continue

        log.info(f"  [FETCH] {ticker}: Alpaca 1분봉 수집 중...")
        try:
            df = fetch_bars(
                tickers=[ticker],
                timeframe_minutes=1,
                start_date=start_date,
                end_date=end_date,
                cache_path=cache_path,
                use_cache=True,
                market_hours_only=True,
                verbose=True,
            )
            if df.empty:
                log.warning(f"  [WARN] {ticker}: 빈 데이터 반환")
                results[ticker] = {"status": "empty", "path": str(cache_path)}
            else:
                log.info(f"  [OK] {ticker}: {len(df):,} rows → {cache_path.name}")
                results[ticker] = {"status": "fetched", "rows": len(df), "path": str(cache_path)}
        except Exception as e:
            log.error(f"  [ERROR] {ticker}: {e}")
            results[ticker] = {"status": "error", "error": str(e)}

    # verify_needed 종목 시도 (실패해도 OK)
    for ticker in GROUP_A_VERIFY_NEEDED:
        cache_path = config.OHLCV_DIR / f"{ticker.lower()}_1min.parquet"
        if cache_path.exists():
            results[ticker] = {"status": "exists", "path": str(cache_path)}
            continue
        if dry_run:
            results[ticker] = {"status": "dry_run_verify", "path": str(cache_path)}
            continue
        log.info(f"  [VERIFY] {ticker}: Alpaca 지원 여부 테스트...")
        try:
            df = fetch_bars(
                tickers=[ticker],
                timeframe_minutes=1,
                start_date=date(2026, 2, 10),
                end_date=date(2026, 2, 14),
                cache_path=None,
                use_cache=False,
                market_hours_only=True,
                verbose=False,
            )
            if not df.empty:
                log.info(f"  [OK] {ticker}: Alpaca 지원 확인됨 ({len(df)} rows 샘플)")
                # 전체 기간 수집
                df_full = fetch_bars(
                    tickers=[ticker],
                    timeframe_minutes=1,
                    start_date=start_date,
                    end_date=end_date,
                    cache_path=cache_path,
                    use_cache=True,
                    verbose=True,
                )
                results[ticker] = {"status": "fetched", "rows": len(df_full), "path": str(cache_path)}
            else:
                log.warning(f"  [NOT SUPPORTED] {ticker}: Alpaca에서 데이터 없음")
                results[ticker] = {"status": "not_supported_on_alpaca"}
        except Exception as e:
            log.warning(f"  [NOT SUPPORTED] {ticker}: {e}")
            results[ticker] = {"status": "not_supported_on_alpaca", "error": str(e)}

    return results


# ============================================================
# Step 3: 히스토리 메타데이터 생성 (yfinance 일봉 기반)
# ============================================================

def build_history_metadata(dry_run: bool = False) -> dict:
    """
    종목별 3년 최고가, 1년 저가, ATH, VNQ 120일 MA를 계산하여
    data/meta/history_metadata.json 에 저장한다.
    """
    log.info("=== 히스토리 메타데이터 생성 ===")

    meta_path = config.META_DIR / "history_metadata.json"
    today = datetime.now().strftime("%Y-%m-%d")

    # 기존 메타데이터 로드
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
        log.info(f"  기존 메타데이터 로드: {len(existing_meta)} 항목")
    else:
        existing_meta = {}

    metadata = dict(existing_meta)

    # 대상 종목: 모든 메타 계산 대상
    all_meta_tickers = set()
    for tlist in GROUP_B_META_TICKERS.values():
        all_meta_tickers.update(tlist)

    # VNQ는 별도 처리
    all_meta_tickers.add("VNQ")

    log.info(f"  메타데이터 계산 대상: {sorted(all_meta_tickers)}")

    if dry_run:
        log.info("  [DRY-RUN] yfinance 호출 예정만 표시")
        for ticker in sorted(all_meta_tickers):
            log.info(f"    {ticker}: yf.download(period='3y') 예정")
        return {"dry_run": True, "tickers": list(all_meta_tickers)}

    # yfinance로 3년 일봉 다운로드
    tickers_str = " ".join(sorted(all_meta_tickers))
    log.info(f"  yfinance 3년 일봉 다운로드: {tickers_str}")
    try:
        df_all = yf.download(
            tickers_str,
            period="3y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        log.error(f"  yfinance 다운로드 실패: {e}")
        return {"error": str(e)}

    if df_all.empty:
        log.error("  빈 데이터 반환")
        return {"error": "empty data"}

    log.info(f"  다운로드 완료: {df_all.shape}")

    # 컬럼 처리 (단일 vs 멀티 티커)
    is_multi = isinstance(df_all.columns, pd.MultiIndex)

    def get_close(ticker: str) -> pd.Series | None:
        try:
            if is_multi:
                return df_all[("Close", ticker)].dropna()
            else:
                return df_all["Close"].dropna()
        except Exception:
            return None

    for ticker in sorted(all_meta_tickers):
        close = get_close(ticker)
        if close is None or close.empty:
            log.warning(f"  [SKIP] {ticker}: 데이터 없음")
            continue

        entry = metadata.get(ticker, {})

        # 3년 최고가
        entry["high_3y"] = round(float(close.max()), 4)

        # 1년 저가
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        close_1y = close[close.index >= one_year_ago]
        if not close_1y.empty:
            entry["low_1y"] = round(float(close_1y.min()), 4)

        # ATH (= 3년 최고가로 근사 - 실제 ATH는 별도 계산 필요)
        entry["ath_approx"] = round(float(close.max()), 4)
        entry["latest_close"] = round(float(close.iloc[-1]), 4)
        entry["as_of"] = today

        metadata[ticker] = entry
        log.info(f"  [OK] {ticker}: high_3y={entry['high_3y']}, low_1y={entry.get('low_1y', 'N/A')}, close={entry['latest_close']}")

    # VNQ 120일 이동평균 추가 계산
    vnq_close = get_close("VNQ")
    if vnq_close is not None and not vnq_close.empty:
        ma120 = vnq_close.rolling(120).mean()
        if not ma120.empty:
            metadata.setdefault("VNQ", {})["ma120"] = round(float(ma120.iloc[-1]), 4)
            log.info(f"  [OK] VNQ MA120: {metadata['VNQ']['ma120']}")

    # 저장
    if not dry_run:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        log.info(f"  저장 완료: {meta_path} ({len(metadata)} 항목)")

    return metadata


# ============================================================
# Step 4: 크립토 스팟 일봉 수집 (jab_bitu 전략)
# ============================================================

def fetch_crypto_daily(dry_run: bool = False) -> dict:
    """
    BTC/ETH/SOL/XRP 스팟 일봉 데이터를 수집하여
    data/market/daily/crypto_daily.parquet 에 저장한다.
    """
    log.info("=== 크립토 스팟 일봉 수집 ===")

    crypto_path = config.DAILY_DIR / "crypto_daily.parquet"

    if crypto_path.exists():
        df_existing = pd.read_parquet(crypto_path)
        log.info(f"  기존 크립토 데이터 존재: {len(df_existing):,} rows")
        if "symbol" in df_existing.columns:
            log.info(f"  종목: {sorted(df_existing['symbol'].unique().tolist())}")
        return {
            "status": "exists",
            "rows": len(df_existing),
            "path": str(crypto_path),
        }

    if dry_run:
        log.info(f"  [DRY-RUN] yfinance 크립토 다운로드 예정 → {crypto_path}")
        return {"status": "dry_run", "path": str(crypto_path)}

    tickers_str = " ".join(GROUP_C_CRYPTO.keys())
    log.info(f"  yfinance 크립토 다운로드: {tickers_str}")
    try:
        df = yf.download(
            tickers_str,
            period="3y",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        log.error(f"  크립토 다운로드 실패: {e}")
        return {"status": "error", "error": str(e)}

    if df.empty:
        log.error("  빈 데이터 반환")
        return {"status": "empty"}

    # 정리: 멀티 인덱스 → 롱 포맷
    if isinstance(df.columns, pd.MultiIndex):
        frames = []
        for ticker in GROUP_C_CRYPTO.keys():
            try:
                sub = df.xs(ticker, level=1, axis=1).copy()
                sub["symbol"] = ticker.split("-")[0]  # BTC-USD → BTC
                sub = sub.reset_index()
                sub.rename(columns={"Date": "date", "index": "date"}, errors="ignore", inplace=True)
                if "Datetime" in sub.columns:
                    sub.rename(columns={"Datetime": "date"}, inplace=True)
                frames.append(sub)
            except Exception:
                continue
        if frames:
            df_out = pd.concat(frames, ignore_index=True)
        else:
            return {"status": "parse_error"}
    else:
        df_out = df.reset_index()
        df_out["symbol"] = "BTC"

    df_out.to_parquet(crypto_path, index=False)
    log.info(f"  [OK] 크립토 일봉 저장: {crypto_path} ({len(df_out):,} rows)")
    return {"status": "fetched", "rows": len(df_out), "path": str(crypto_path)}


# ============================================================
# Step 5: 데이터 갭 리포트 생성
# ============================================================

def generate_gap_report(
    coverage: dict,
    fetch_1min_results: dict,
    meta_results: dict,
    crypto_results: dict,
) -> str:
    """데이터 갭 리포트 문자열을 생성한다."""

    current_1min = set(coverage["current_1min_tickers"])
    soxx_available = coverage["soxx_1min_available"]
    if soxx_available:
        current_1min.add("SOXX")

    daily_tickers = set(coverage["daily_3y_tickers"])

    lines = [
        "# Line B 데이터 확대 리포트",
        f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 현재 데이터 현황",
        "",
        f"### 1분봉 OHLCV (backtest_1min_v2.parquet)",
        f"- 종목 수: {coverage['current_1min_count']}개",
        f"- 날짜 범위: {coverage['current_1min_date_range'][0]} ~ {coverage['current_1min_date_range'][1]}",
        f"- 포함 종목: {', '.join(coverage['current_1min_tickers'])}",
        f"- SOXX 별도 파일: {'있음' if soxx_available else '없음'}",
        "",
        f"### 일봉 3년 (strategy_daily_3y.parquet)",
        f"- 종목 수: {coverage['daily_3y_count']}개",
        f"- 날짜 범위: {coverage['daily_3y_date_range'][0]} ~ {coverage['daily_3y_date_range'][1]}",
        f"- 포함 종목: {', '.join(coverage['daily_3y_tickers'])}",
        "",
        "## Group A: 1분봉 OHLCV 추가 수집 결과",
        "",
        "| 종목 | 필요 전략 | 상태 | 비고 |",
        "|------|----------|------|------|",
    ]

    strategy_map = {
        "SOXX": "jab_soxl, sector_rotate",
        "TSLA": "jab_tsll",
        "TSLL": "jab_tsll",
        "UVXY": "vix_gold (VIX 대리)",
        "IAU": "vix_gold",
        "GDXU": "vix_gold, short_macro",
        "ETQ": "jab_etq",
        "BAC": "bank_conditional",
        "JPM": "bank_conditional",
        "HSBC": "bank_conditional",
        "WFC": "bank_conditional",
        "C": "bank_conditional",
        "RBC": "sector_rotate",
        "VNQ": "reit_risk",
        "SOXS": "bear_regime",
        "BITI": "bear_regime",
        "CONZ": "bear_regime (인버스)",
        "IREZ": "bear_regime (인버스)",
        "HOOZ": "bear_regime (인버스)",
        "MSTZ": "bear_regime (인버스)",
    }

    already_in_daily = set(coverage["daily_3y_tickers"])

    for ticker, strategy in strategy_map.items():
        # Check individual parquet file too
        individual_path = config.OHLCV_DIR / f"{ticker.lower()}_1min.parquet"
        if ticker in current_1min:
            status = "있음 (1분봉)"
        elif individual_path.exists():
            df_ind = pd.read_parquet(individual_path)
            status = f"수집완료 ({len(df_ind):,} rows)"
        elif ticker in already_in_daily:
            status = "있음 (일봉만)"
        elif ticker in fetch_1min_results:
            r = fetch_1min_results[ticker]
            s = r.get("status", "unknown")
            if s == "fetched":
                status = f"수집완료 ({r.get('rows', 0):,} rows)"
            elif s == "exists":
                status = f"기존 파일 존재 ({r.get('rows', 0):,} rows)"
            elif s == "dry_run":
                status = "dry-run (미실행)"
            elif s == "not_supported_on_alpaca":
                status = "Alpaca 미지원"
            elif s == "error":
                status = f"오류: {r.get('error', '')[:40]}"
            else:
                status = s
        else:
            status = "미수집"

        # 비고
        note = ""
        if ticker in ["CONZ", "IREZ", "HOOZ", "MSTZ"]:
            note = "Alpaca 미지원 (non-US ETP/역방향)"
        elif ticker == "ETQ":
            note = "ETP — Alpaca 미지원, 대안: yfinance 확인 필요"
        elif ticker == "HSBC":
            note = "NYSE ADR — Alpaca 지원 가능"
        elif ticker == "RBC":
            note = "Alpaca 지원 가능"

        lines.append(f"| {ticker} | {strategy} | {status} | {note} |")

    lines += [
        "",
        "## Group B: 히스토리 메타데이터",
        "",
        f"- 파일: `data/meta/history_metadata.json`",
        f"- 상태: {'수집완료' if isinstance(meta_results, dict) and meta_results and 'error' not in meta_results else str(meta_results)}",
        "",
        "| 용도 | 대상 종목 | 데이터 항목 |",
        "|------|----------|------------|",
        "| bargain_buy 진입 기준 | CONL, SOXL, AMDL, NVDL, ROBN, ETHU, BRKU | 3년 최고가 (high_3y) |",
        "| sector_rotate 활성화 | BITU, SOXX, ROBN, GLD | 1년 저가 (low_1y) |",
        "| short_macro 발동 | QQQ, SPY | ATH 근사값 (ath_approx) |",
        "| reit_risk 감지 | VNQ | 120일 이동평균 (ma120) |",
        "",
    ]

    # 메타데이터 상세 (수집된 경우)
    if isinstance(meta_results, dict) and "error" not in meta_results and "dry_run" not in meta_results:
        lines += ["### 계산된 메타데이터 요약", "", "| 종목 | high_3y | low_1y | ath_approx | 최근 종가 |",
                  "|------|---------|--------|------------|----------|"]
        for ticker in sorted(meta_results.keys()):
            if not isinstance(meta_results[ticker], dict):
                continue
            m = meta_results[ticker]
            lines.append(
                f"| {ticker} | {m.get('high_3y', 'N/A')} | {m.get('low_1y', 'N/A')} "
                f"| {m.get('ath_approx', 'N/A')} | {m.get('latest_close', 'N/A')} |"
            )
        lines.append("")

    lines += [
        "## Group C: 크립토 스팟 데이터",
        "",
        f"- 파일: `data/market/daily/crypto_daily.parquet`",
        f"- 상태: {crypto_results.get('status', 'unknown')}",
        f"- 필요 전략: jab_bitu (BTC, ETH, SOL, XRP 당일 변동률 조건)",
        "",
        "| 심볼 | 설명 | 소스 |",
        "|------|------|------|",
        "| BTC-USD | Bitcoin | yfinance |",
        "| ETH-USD | Ethereum | yfinance |",
        "| SOL-USD | Solana | yfinance |",
        "| XRP-USD | Ripple | yfinance |",
        "",
        "## Group D: Polymarket 확장",
        "",
        "| 지표 | 필요 전략 | 현재 상태 | 비고 |",
        "|------|----------|----------|------|",
        "| btc_monthly_dip | bear_regime_long | poly_config.py에 btc_monthly 존재 | Dip 컴포넌트 파싱 추가 필요 |",
        "",
        "## 남은 작업 (수동/설정 필요)",
        "",
        "### 즉시 가능",
        "1. `python experiments/expand_line_b_data.py --fetch-1min` — Alpaca로 Group A 수집",
        "2. `python experiments/expand_line_b_data.py --fetch-meta` — yfinance로 Group B 메타데이터",
        "3. `python experiments/expand_line_b_data.py --fetch-crypto` — yfinance로 Group C 크립토",
        "",
        "### Alpaca 미지원 (대안 필요)",
        "- **ETQ**: GraniteShares ETP — Alpha Vantage, 직접 API 또는 수동 수집 필요",
        "- **HSBC/RBC**: Alpaca 지원 여부 실행 시 자동 확인",
        "- **CONZ/IREZ/HOOZ/MSTZ**: 역방향 인버스 ETP — 대부분 비US 상장으로 미지원",
        "",
        "### Polygon API (미설정)",
        "- `.env`에 `POLYGON_API_KEY` 없음",
        "- VIX 지수 직접 조회 시 필요 (현재 UVXY로 대체 가능)",
        "",
        "### Polymarket btc_monthly_dip",
        "- `polymarket/poly_config.py`의 `btc_monthly` 지표 활용 가능",
        "- `poly_history.py`에서 Dip 마켓 컴포넌트만 추출하면 됨",
        "- 별도 API 키 불필요",
        "",
        "## 권장 실행 순서",
        "",
        "```bash",
        "# 1. 전체 dry-run으로 계획 확인",
        "pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --dry-run",
        "",
        "# 2. 메타데이터 먼저 수집 (API 부하 없음)",
        "pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-meta",
        "",
        "# 3. 크립토 일봉 수집",
        "pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-crypto",
        "",
        "# 4. Alpaca 1분봉 수집 (시간 소요: 약 15~30분)",
        "pyenv shell ptj_stock_lab && python experiments/expand_line_b_data.py --fetch-1min",
        "```",
    ]

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Line B 데이터 확대 스크립트")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 계획만 출력")
    parser.add_argument("--fetch-1min", action="store_true", help="Alpaca 1분봉 수집만 실행")
    parser.add_argument("--fetch-meta", action="store_true", help="히스토리 메타데이터 수집만 실행")
    parser.add_argument("--fetch-crypto", action="store_true", help="크립토 일봉 수집만 실행")
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="수집할 종목 (쉼표 구분). 예: SOXX,TSLA,VNQ"
    )
    parser.add_argument(
        "--start", type=str, default="2025-01-03",
        help="수집 시작일 (YYYY-MM-DD). 기본: 2025-01-03"
    )
    parser.add_argument(
        "--end", type=str, default="2026-02-20",
        help="수집 종료일 (YYYY-MM-DD). 기본: 2026-02-20"
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    tickers_override = None
    if args.tickers:
        tickers_override = [t.strip() for t in args.tickers.split(",")]

    run_all = not any([args.fetch_1min, args.fetch_meta, args.fetch_crypto, dry_run])

    # Step 1: 커버리지 분석 (항상)
    coverage = analyze_coverage()
    print()

    fetch_1min_results = {}
    meta_results = {}
    crypto_results = {}

    # Step 2: 1분봉 수집
    if args.fetch_1min or dry_run or run_all:
        fetch_1min_results = fetch_additional_1min(
            tickers=tickers_override,
            start_date=start_date,
            end_date=end_date,
            dry_run=dry_run,
        )
        print()

    # Step 3: 메타데이터
    if args.fetch_meta or dry_run or run_all:
        meta_results = build_history_metadata(dry_run=dry_run)
        print()

    # Step 4: 크립토 일봉
    if args.fetch_crypto or dry_run or run_all:
        crypto_results = fetch_crypto_daily(dry_run=dry_run)
        print()

    # Step 5: 갭 리포트
    log.info("=== 갭 리포트 생성 ===")
    report = generate_gap_report(coverage, fetch_1min_results, meta_results, crypto_results)

    report_path = (
        _ROOT / "docs" / "reports" / "backtest" / "line_b_data_expansion_plan.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"리포트 저장: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()

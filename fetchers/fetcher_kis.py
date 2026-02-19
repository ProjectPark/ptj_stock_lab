"""
한국투자증권 API 기반 데이터 수집기
===================================
fetcher.py (yfinance)와 동일한 인터페이스 제공.
app.py에서 import만 바꾸면 전환 완료.
"""
from __future__ import annotations

import sys
import logging
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FETCHERS = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_FETCHERS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from config import TICKERS
from kis_client import KISClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 모듈 레벨 클라이언트 (재사용, 토큰 캐시)
_client: KISClient | None = None


def _get_client() -> KISClient:
    global _client
    if _client is None:
        _client = KISClient()
    return _client


# ── fetcher.py 호환 인터페이스 ───────────────────────────────

def fetch_intraday() -> pd.DataFrame:
    """당일 1분봉 데이터 (모든 종목).

    Returns:
        MultiIndex columns DataFrame — fetcher.py와 동일 형태
        (Price, Ticker) 구조
    """
    client = _get_client()
    frames = {}

    for symbol, info in TICKERS.items():
        exchange = info["exchange"]
        try:
            rows = client.get_overseas_minutes(symbol, exchange, nmin=1)
            if not rows:
                continue

            df = pd.DataFrame(rows)
            # time 파싱: "20260217093000" → datetime
            df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S", errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime")
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            frames[symbol] = df

        except Exception as e:
            log.warning("[%s] 분봉 수집 실패: %s", symbol, e)

    if not frames:
        return pd.DataFrame()

    # MultiIndex columns: (Price, Ticker) — yfinance 포맷 호환
    combined = pd.concat(frames, axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [(col, sym) for sym, df in frames.items() for col in df.columns],
        names=["Price", "Ticker"],
    )
    return combined


def get_current_snapshot(intraday_df: pd.DataFrame = None) -> dict[str, dict]:
    """종목별 현재 스냅샷.

    KIS API는 현재가 API를 직접 호출하므로 intraday_df 없이도 동작.
    (인자는 fetcher.py 호환성을 위해 유지)

    Returns:
        {symbol: {symbol, name, category, price, open, high, low,
                  prev_close, change_pct, volume}}
    """
    client = _get_client()
    results = {}

    for symbol, info in TICKERS.items():
        exchange = info["exchange"]
        try:
            snap = client.get_overseas_price(symbol, exchange)
            snap["name"] = info["name"]
            snap["category"] = info["category"]
            results[symbol] = snap
        except Exception as e:
            log.warning("[%s] 현재가 조회 실패: %s", symbol, e)

    return results


def get_intraday_pct_series(
    intraday_df: pd.DataFrame, symbols: list[str]
) -> pd.DataFrame:
    """장중 등락률 시계열 (%) — fetcher.py와 동일 로직.

    시가 대비 등락률로 변환.
    """
    if intraday_df.empty:
        return pd.DataFrame()

    multi = isinstance(intraday_df.columns, pd.MultiIndex)
    series = {}

    for sym in symbols:
        try:
            if multi:
                close = intraday_df[("Close", sym)]
                opn = intraday_df[("Open", sym)]
            else:
                close = intraday_df["Close"]
                opn = intraday_df["Open"]

            first_open = opn.dropna().iloc[0]
            if first_open == 0:
                continue
            pct = (close - first_open) / first_open * 100
            series[sym] = pct
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series)
    df.index.name = "time"
    return df


# ── 디버그 ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== KIS API 현재가 스냅샷 ===")
    snap = get_current_snapshot()
    for sym, d in snap.items():
        pct = d["change_pct"]
        sign = "+" if pct >= 0 else ""
        print(f"  {sym:6s} ${d['price']:>10.2f}  {sign}{pct:.2f}%  ({d['name']})")

    print(f"\n총 {len(snap)}개 종목 조회 완료")

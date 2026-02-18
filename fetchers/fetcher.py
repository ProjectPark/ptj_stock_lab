"""
yfinance 기반 데이터 수집기
============================
- 장중 1분봉 (프리마켓 포함) 으로 실시간 시세 조회
- 3개월 일봉 히스토리 (parquet 캐시)
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import CACHE_EXPIRE_HOURS, DATA_DIR, LOOKBACK_PERIOD, TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# yfinance 는 ticker 목록을 공백으로 구분
_ALL_SYMBOLS = list(TICKERS.keys())
_SYMBOLS_STR = " ".join(_ALL_SYMBOLS)


# ------------------------------------------------------------------
# 장중 시세 (1분봉, 프리마켓 포함)
# ------------------------------------------------------------------
def fetch_intraday() -> pd.DataFrame:
    """
    당일 1분봉 데이터를 가져옴 (프리마켓 포함).

    Returns:
        MultiIndex columns (Price, Ticker) DataFrame.
        Price: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(
            _SYMBOLS_STR,
            period="1d",
            interval="1m",
            prepost=True,          # 프리마켓 + 애프터마켓
            progress=False,
            threads=True,
        )
        return df
    except Exception as e:
        log.error("장중 데이터 수집 실패: %s", e)
        return pd.DataFrame()


def get_current_snapshot(intraday_df: pd.DataFrame) -> dict[str, dict]:
    """
    1분봉 데이터에서 종목별 현재 스냅샷을 추출.

    Returns:
        {symbol: {price, open, high, low, prev_close, change_pct, volume, ...}}
    """
    if intraday_df.empty:
        return {}

    results: dict[str, dict] = {}

    for symbol in _ALL_SYMBOLS:
        try:
            # 단일 종목인 경우와 멀티 종목인 경우 컬럼 접근 방식이 다름
            if len(_ALL_SYMBOLS) == 1:
                close_series = intraday_df["Close"]
                open_series = intraday_df["Open"]
                high_series = intraday_df["High"]
                low_series = intraday_df["Low"]
                vol_series = intraday_df["Volume"]
            else:
                close_series = intraday_df[("Close", symbol)]
                open_series = intraday_df[("Open", symbol)]
                high_series = intraday_df[("High", symbol)]
                low_series = intraday_df[("Low", symbol)]
                vol_series = intraday_df[("Volume", symbol)]

            close_valid = close_series.dropna()
            if close_valid.empty:
                continue

            current_price = float(close_valid.iloc[-1])
            day_open = float(open_series.dropna().iloc[0])
            day_high = float(high_series.max())
            day_low = float(low_series.min())
            total_volume = int(vol_series.sum())

            # 등락률: 당일 시가 대비
            change_pct = ((current_price - day_open) / day_open * 100) if day_open else 0

            results[symbol] = {
                "symbol": symbol,
                "name": TICKERS[symbol]["name"],
                "category": TICKERS[symbol]["category"],
                "price": current_price,
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "change_pct": round(change_pct, 2),
                "volume": total_volume,
            }
        except Exception as e:
            log.warning("[%s] 스냅샷 추출 실패: %s", symbol, e)

    return results


# ------------------------------------------------------------------
# 차트용: 장중 등락률 시계열
# ------------------------------------------------------------------
def get_intraday_pct_series(intraday_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    1분봉 종가를 시가 대비 등락률(%)로 변환.

    Returns:
        DataFrame — index: datetime, columns: symbol별 등락률(%)
    """
    if intraday_df.empty:
        return pd.DataFrame()

    multi = len(_ALL_SYMBOLS) > 1
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


# ------------------------------------------------------------------
# 히스토리 (3개월 일봉, parquet 캐시)
# ------------------------------------------------------------------
def _cache_path() -> str:
    return str(DATA_DIR / "history.parquet")


def _cache_is_fresh() -> bool:
    """캐시 파일이 존재하고 만료되지 않았으면 True"""
    path = DATA_DIR / "history.parquet"
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=CACHE_EXPIRE_HOURS)


def fetch_history(force: bool = False) -> pd.DataFrame:
    """
    3개월 일봉 데이터를 가져옴 (parquet 캐시 사용).

    Args:
        force: True이면 캐시 무시하고 새로 다운로드

    Returns:
        MultiIndex columns DataFrame
    """
    if not force and _cache_is_fresh():
        log.info("캐시 사용: %s", _cache_path())
        return pd.read_parquet(_cache_path())

    try:
        df = yf.download(
            _SYMBOLS_STR,
            period=LOOKBACK_PERIOD,
            interval="1d",
            progress=False,
            threads=True,
        )
        if not df.empty:
            df.to_parquet(_cache_path())
            log.info("히스토리 저장: %s (%d행)", _cache_path(), len(df))
        return df
    except Exception as e:
        log.error("히스토리 수집 실패: %s", e)
        return pd.DataFrame()


# ------------------------------------------------------------------
# 디버그: 단독 실행
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 장중 1분봉 수집 ===")
    intra = fetch_intraday()
    print(f"shape: {intra.shape}")
    if not intra.empty:
        print(intra.tail(3))

    print("\n=== 현재 스냅샷 ===")
    snap = get_current_snapshot(intra)
    for sym, data in snap.items():
        pct = data["change_pct"]
        sign = "+" if pct >= 0 else ""
        print(f"  {sym:6s} ${data['price']:>10.2f}  {sign}{pct:.2f}%  ({data['name']})")

    print("\n=== 히스토리 (3개월) ===")
    hist = fetch_history()
    print(f"shape: {hist.shape}")

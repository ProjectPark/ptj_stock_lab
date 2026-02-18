"""
PTJ 매매법 - 백테스트 공통 유틸리티
===================================
모든 버전(v1/v2/v3)에서 공유하는 함수:
- 데이터 로드 (parquet, Polymarket JSON)
- 수수료 계산 (KIS 미국주식)
- 등락률 계산
- 리포트 지표 (MDD, Sharpe)
"""

import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

# ============================================================
# KIS 미국주식 수수료 상수
# ============================================================
KIS_COMMISSION_PCT = 0.25      # 매매 수수료 0.25% (매수/매도 각각)
KIS_SEC_FEE_PCT = 0.00278      # SEC Fee 0.00278% (매도 시에만)
KIS_FX_SPREAD_PCT = 0.10       # 환전 스프레드 약 0.1% (편도)


# ============================================================
# 데이터 로드
# ============================================================

def load_parquet(path: str | Path) -> pd.DataFrame:
    """Parquet 캐시 파일을 로드한다.

    Parameters
    ----------
    path : str | Path
        parquet 파일 경로

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않을 때
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    df = pd.read_parquet(path)
    print(f"  Loaded: {len(df):,} rows ({path.name})")
    return df


def load_fx_hourly(path: str | Path) -> pd.Series:
    """시간별 USD/KRW 환율 데이터를 로드한다.

    Parameters
    ----------
    path : str | Path
        parquet 파일 경로 (timestamp, close 컬럼 필수)

    Returns
    -------
    pd.Series
        timestamp(tz-aware) → close 환율. sort_index 완료.
        비어있으면 빈 Series 반환.
    """
    path = Path(path)
    if not path.exists():
        print(f"  [WARN] FX 데이터 없음: {path} — 고정 환율 사용")
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    series = df.set_index("timestamp")["close"].sort_index()
    print(f"  Loaded FX: {len(series):,} rows ({series.index.min().date()} ~ {series.index.max().date()})")
    return series


def load_polymarket_daily(history_dir: str | Path) -> dict[date, dict]:
    """Polymarket JSON 히스토리 파일 전체를 로드한다.

    Parameters
    ----------
    history_dir : str | Path
        polymarket/history/ 디렉토리 경로

    Returns
    -------
    dict[date, dict]
        {date_obj: {"btc_up": float, "ndx_up": float, "eth_up": float, "rate_hike": float}}
        각 값은 0.0~1.0 확률. 데이터 없으면 0.5 기본값.
    """
    history_dir = Path(history_dir)
    result: dict[date, dict] = {}

    json_files = sorted(history_dir.glob("*.json"))
    if not json_files:
        print(f"  Loaded Polymarket: 0 days (no files)")
        return result

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # 날짜 파싱: JSON 내부 "date" 필드 우선, 없으면 파일명에서 추출
        date_str = data.get("date")
        if not date_str:
            # 파일명: {YYYY-MM-DD}_{fidelity}m.json
            date_str = fp.stem.split("_")[0]

        try:
            dt = date.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        indicators = data.get("indicators", {})

        btc_up = _extract_up_down_prob(indicators.get("btc_up_down"), "Up")
        ndx_up = _extract_up_down_prob(indicators.get("ndx_up_down"), "Up")
        eth_up = _extract_yes_no_prob(indicators.get("eth_above_today"), "Yes")
        rate_hike = _extract_yes_no_prob(indicators.get("fed_decision"), "Yes")

        # 같은 날짜에 여러 파일(fidelity 다름)이 있을 수 있으므로,
        # 더 세밀한 데이터(time series 있는 쪽)를 우선 사용
        if dt in result:
            existing = result[dt]
            # 기존 값이 0.5(기본값)이고 새 값이 아니면 덮어쓰기
            if existing["btc_up"] == 0.5 and btc_up != 0.5:
                existing["btc_up"] = btc_up
            if existing["ndx_up"] == 0.5 and ndx_up != 0.5:
                existing["ndx_up"] = ndx_up
            if existing["eth_up"] == 0.5 and eth_up != 0.5:
                existing["eth_up"] = eth_up
            if existing.get("rate_hike", 0.5) == 0.5 and rate_hike != 0.5:
                existing["rate_hike"] = rate_hike
        else:
            result[dt] = {"btc_up": btc_up, "ndx_up": ndx_up, "eth_up": eth_up, "rate_hike": rate_hike}

    if result:
        min_date = min(result.keys())
        max_date = max(result.keys())
        print(f"  Loaded Polymarket: {len(result)} days ({min_date} ~ {max_date})")
    else:
        print(f"  Loaded Polymarket: 0 days (no valid data)")

    return result


def _extract_up_down_prob(indicator: dict | None, outcome_key: str) -> float:
    """btc_up_down / ndx_up_down 구조에서 확률을 추출한다.

    Parameters
    ----------
    indicator : dict | None
        indicators["btc_up_down"] 등
    outcome_key : str
        "Up" 또는 "Down"

    Returns
    -------
    float
        0.0~1.0 확률. 데이터 없으면 0.5
    """
    if indicator is None:
        return 0.5

    if "error" in indicator:
        return 0.5

    # 1) time series 데이터가 있으면 마지막 값 사용
    markets = indicator.get("markets", [])
    if markets:
        outcomes = markets[0].get("outcomes", {})
        series = outcomes.get(outcome_key, [])
        if series and len(series) > 0:
            last_entry = series[-1]
            # 형식: {"t": timestamp, "p": probability} 또는 [timestamp, price]
            if isinstance(last_entry, dict):
                return float(last_entry.get("p", 0.5))
            elif isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                return float(last_entry[1])

    # 2) final_prices 사용
    final_prices = indicator.get("final_prices", {})
    if outcome_key in final_prices:
        try:
            return float(final_prices[outcome_key])
        except (ValueError, TypeError):
            pass

    return 0.5


def _extract_yes_no_prob(indicator: dict | None, outcome_key: str) -> float:
    """eth_above_today 구조에서 확률을 추출한다.

    Parameters
    ----------
    indicator : dict | None
        indicators["eth_above_today"]
    outcome_key : str
        "Yes" 또는 "No"

    Returns
    -------
    float
        0.0~1.0 확률. 데이터 없으면 0.5
    """
    if indicator is None:
        return 0.5

    if "error" in indicator:
        return 0.5

    # 1) time series 데이터가 있으면 마지막 값 사용
    markets = indicator.get("markets", [])
    if markets:
        outcomes = markets[0].get("outcomes", {})
        series = outcomes.get(outcome_key, [])
        if series and len(series) > 0:
            last_entry = series[-1]
            if isinstance(last_entry, dict):
                return float(last_entry.get("p", 0.5))
            elif isinstance(last_entry, (list, tuple)) and len(last_entry) >= 2:
                return float(last_entry[1])

    # 2) final_prices 사용
    final_prices = indicator.get("final_prices", {})
    if outcome_key in final_prices:
        try:
            return float(final_prices[outcome_key])
        except (ValueError, TypeError):
            pass

    return 0.5


# ============================================================
# DataFrame 사전 인덱싱 (성능 최적화)
# ============================================================

def preindex_dataframe(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Pre-index DataFrame into nested dicts for O(1) bar-level lookup.

    Eliminates per-bar DataFrame filtering (the primary backtest bottleneck).
    One-time cost during data loading; all subsequent access is dict lookup.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, timestamp, symbol, open, high, low, close, volume

    Returns
    -------
    ts_prices : dict
        ``{date: {timestamp: {symbol: close_price}}}``
    sym_bars : dict
        ``{date: {symbol: [{"timestamp", "open", "high", "low", "close", "volume"}, ...]}}``
        Each symbol's bars are sorted by timestamp.
    day_timestamps : dict
        ``{date: [sorted unique timestamps]}``
    """
    ts_prices: dict = {}
    sym_bars: dict = {}

    for row in df.itertuples(index=False):
        d = row.date
        ts = row.timestamp
        sym = row.symbol

        # ts_prices: date -> timestamp -> {symbol: close}
        if d not in ts_prices:
            ts_prices[d] = {}
        if ts not in ts_prices[d]:
            ts_prices[d][ts] = {}
        ts_prices[d][ts][sym] = float(row.close)

        # sym_bars: date -> symbol -> [bar_dicts]
        if d not in sym_bars:
            sym_bars[d] = {}
        if sym not in sym_bars[d]:
            sym_bars[d][sym] = []
        sym_bars[d][sym].append({
            "timestamp": ts,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        })

    # Ensure bars are sorted by timestamp within each (date, symbol)
    for d in sym_bars:
        for sym in sym_bars[d]:
            sym_bars[d][sym].sort(key=lambda b: b["timestamp"])

    # Build sorted unique timestamps per day
    day_timestamps = {d: sorted(ts_prices[d].keys()) for d in ts_prices}

    n_days = len(day_timestamps)
    print(f"  Pre-indexed: {len(df):,} rows -> {n_days} days")

    return ts_prices, sym_bars, day_timestamps


# ============================================================
# 수수료 계산
# ============================================================

def calc_buy_fee(amount: float) -> float:
    """매수 수수료를 계산한다.

    Buy-side fee = amount * (수수료 + 환전 스프레드) / 100

    Parameters
    ----------
    amount : float
        매수 금액 (USD)

    Returns
    -------
    float
        수수료 금액
    """
    return amount * (KIS_COMMISSION_PCT + KIS_FX_SPREAD_PCT) / 100


def calc_sell_fee(proceeds: float) -> float:
    """매도 수수료를 계산한다.

    Sell-side fee = proceeds * (수수료 + SEC Fee + 환전 스프레드) / 100

    Parameters
    ----------
    proceeds : float
        매도 대금 (USD)

    Returns
    -------
    float
        수수료 금액
    """
    return proceeds * (KIS_COMMISSION_PCT + KIS_SEC_FEE_PCT + KIS_FX_SPREAD_PCT) / 100


def calc_net_buy(amount: float) -> tuple[float, float]:
    """매수 시 순 투입금액과 수수료를 계산한다.

    Parameters
    ----------
    amount : float
        총 매수 금액 (USD)

    Returns
    -------
    tuple[float, float]
        (net_amount_for_shares, fee)
        net_amount = amount - fee
    """
    fee = calc_buy_fee(amount)
    return (amount - fee, fee)


def calc_net_sell(proceeds: float) -> tuple[float, float]:
    """매도 시 순 수취금액과 수수료를 계산한다.

    Parameters
    ----------
    proceeds : float
        매도 대금 (USD)

    Returns
    -------
    tuple[float, float]
        (net_proceeds, fee)
        net_proceeds = proceeds - fee
    """
    fee = calc_sell_fee(proceeds)
    return (proceeds - fee, fee)


# ============================================================
# 등락률 계산
# ============================================================

def calc_changes(
    cur_prices: dict[str, float],
    prev_close: dict[str, float],
) -> dict[str, dict]:
    """현재가 대비 전일 종가 등락률을 계산한다.

    Parameters
    ----------
    cur_prices : dict[str, float]
        {ticker: current_price}
    prev_close : dict[str, float]
        {ticker: previous_close_price}

    Returns
    -------
    dict[str, dict]
        {ticker: {"close": price, "prev_close": prev, "change_pct": rounded_pct}}
    """
    result = {}
    for ticker, price in cur_prices.items():
        prev = prev_close.get(ticker)
        if prev and prev != 0:
            change_pct = round((price - prev) / prev * 100, 2)
        else:
            change_pct = 0.0
        result[ticker] = {
            "close": price,
            "prev_close": prev_close.get(ticker, 0.0),
            "change_pct": change_pct,
        }
    return result


# ============================================================
# 리포트 지표
# ============================================================

def calc_mdd(equity_curve: list[tuple]) -> float:
    """최대낙폭(MDD)을 계산한다.

    Parameters
    ----------
    equity_curve : list[tuple]
        [(date, value), ...] 형태의 자산곡선

    Returns
    -------
    float
        MDD를 양수 퍼센트로 반환 (예: 15.3 → -15.3% 낙폭)
    """
    if len(equity_curve) < 2:
        return 0.0

    values = [v for _, v in equity_curve]
    peak = values[0]
    max_drawdown = 0.0

    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            drawdown = (peak - v) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return round(max_drawdown, 2)


def calc_sharpe(equity_curve: list[tuple], risk_free: float = 0.0) -> float:
    """연환산 샤프비율을 계산한다.

    Parameters
    ----------
    equity_curve : list[tuple]
        [(date, value), ...] 형태의 자산곡선
    risk_free : float
        무위험수익률 (연율, 기본 0.0)

    Returns
    -------
    float
        연환산 Sharpe ratio. 계산 불가 시 0.0
    """
    if len(equity_curve) < 3:
        return 0.0

    values = np.array([v for _, v in equity_curve], dtype=np.float64)

    # 일별 수익률
    daily_returns = np.diff(values) / values[:-1]

    if len(daily_returns) == 0:
        return 0.0

    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns, ddof=1)

    if std_return == 0.0:
        return 0.0

    # 일별 무위험수익률
    daily_rf = risk_free / 252

    sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
    return round(float(sharpe), 4)

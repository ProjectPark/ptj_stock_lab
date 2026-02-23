"""
Phase C: Polymarket 포지션 스케일 효과 검증 — todd_fuck_v1
==========================================================
btc_rsi_e 고정 unit_mul=3.0 vs scale_unit_mul_by_poly 적용 버전 비교
기간: 2024-02-01 ~ 2025-12-31 (Polymarket 데이터 있는 구간)
종목: IREN, CONL

데이터 소스:
  - OHLCV : data/market/daily/profit_curve_ohlcv.parquet (yfinance, 2023-09-01~)
  - BTC   : data/market/daily/btc_mstr_daily.parquet (ticker == 'BTC-USD')
  - VIX   : data/market/daily/vix_daily.parquet (tz-aware timestamp)
  - Poly  : data/polymarket/{year}/{date}_1m.json (btc_up_down.final_prices.Up)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
OHLCV_PATH = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
BTC_PATH   = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
VIX_PATH   = ROOT / "data" / "market" / "daily" / "vix_daily.parquet"
POLY_DIR   = Path(config.POLY_DATA_DIR)

# ── 백테스트 기간 ─────────────────────────────────────────────────────────────
START_DATE = date(2024, 2, 1)   # Polymarket 데이터 최초일
END_DATE   = date(2025, 12, 31)

# ── 검증 종목 ──────────────────────────────────────────────────────────────────
TICKERS = ["IREN", "CONL"]

# ── 거래 파라미터 ─────────────────────────────────────────────────────────────
EXCHANGE_RATE = 1_350          # KRW/USD 고정 환율
CAPITAL_KRW   = 20_000_000     # 총 투자금
UNIT_KRW      = 3_000_000      # 1-unit 기준 금액 (KRW)

# ── btc_rsi_e 보수적 파라미터 ─────────────────────────────────────────────────
P = {
    "di_min_gap":       7.82,
    "surge_pct":        5.96,
    "btc_rsi_max":      64.62,
    "vix_min":          12.40,
    "vix_max":          34.74,
    "pct_ma20_min":     12.76,
    "vol_ratio_min":    1.42,
    "rsi14_min":        37.16,
    "target_pct":       54.4,
    "stop_pct":         -11.0,
    "hold_days":        50,
    "unit_mul":         3.0,
    "max_pyramid":      2,
    "pyramid_add_pct":  5.7,
    "trailing_pct":     -8.0,
}

# ── Polymarket 스케일 설정 ────────────────────────────────────────────────────
POLY_POSITION_SCALE = {
    "btc_up_thresholds":  [0.45, 0.55, 0.70],
    "unit_mul_factors":   [0.0,  0.7,  1.0,  1.5],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 지표 계산 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high - prev_high) > (prev_low - low), (high - prev_high).clip(lower=0), 0.0)
    dm_minus = np.where((prev_low - low) > (high - prev_high), (prev_low - low).clip(lower=0), 0.0)

    atr_s   = pd.Series(tr).rolling(period).mean()
    dmp_s   = pd.Series(dm_plus,  index=high.index).rolling(period).mean()
    dmm_s   = pd.Series(dm_minus, index=high.index).rolling(period).mean()

    di_plus  = 100 * dmp_s / atr_s.replace(0, np.nan)
    di_minus = 100 * dmm_s / atr_s.replace(0, np.nan)

    dx    = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx   = dx.rolling(period).mean()
    return adx, di_plus, di_minus


# ═══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_ohlcv(ticker: str) -> pd.DataFrame:
    """profit_curve_ohlcv.parquet 에서 해당 ticker OHLCV 로드"""
    df = pd.read_parquet(OHLCV_PATH)
    df = df[df["ticker"] == ticker].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


def load_btc_rsi() -> pd.Series:
    """BTC-USD Close → RSI14 시리즈 반환 (Date 인덱스)"""
    df = pd.read_parquet(BTC_PATH)
    df = df[df["ticker"] == "BTC-USD"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return calc_rsi(df["Close"], 14)


def load_vix() -> pd.Series:
    """VIX close 시리즈 반환 (date str 기준 → Timestamp 인덱스)"""
    df = pd.read_parquet(VIX_PATH)
    # timestamp는 tz-aware → tz 제거 후 date만 사용
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt").set_index("dt")
    return df["close"]


def load_btcup_series(start: date, end: date) -> pd.DataFrame:
    """
    기간별 btc_up_raw (binary 0/1) 로드 후 5d rolling 평균으로 연속화.
    반환: DataFrame with columns [btc_up_raw, btc_up_5d] (Timestamp index)
    """
    records = []
    current = start
    while current <= end:
        fp = POLY_DIR / str(current.year) / f"{current.isoformat()}_1m.json"
        if fp.exists():
            try:
                with open(fp) as f:
                    data = json.load(f)
                indicators = data.get("indicators", {})
                btc_ind    = indicators.get("btc_up_down", {})
                if "error" not in btc_ind:
                    fp_data = btc_ind.get("final_prices", {})
                    up_val  = fp_data.get("Up", None)
                    if up_val is not None:
                        records.append({
                            "date":       pd.Timestamp(current),
                            "btc_up_raw": float(up_val),
                        })
            except Exception:
                pass
        current += timedelta(days=1)

    if not records:
        return pd.DataFrame(columns=["btc_up_raw", "btc_up_5d"])

    df = pd.DataFrame(records).set_index("date").sort_index()
    df["btc_up_5d"] = df["btc_up_raw"].rolling(5).mean()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket 스케일 함수
# ═══════════════════════════════════════════════════════════════════════════════

def scale_unit_mul(base_mul: float, btc_up_5d: float) -> float:
    """btc_up 5d rolling 기반 unit_mul 스케일링"""
    if pd.isna(btc_up_5d):
        return base_mul

    thresholds = POLY_POSITION_SCALE["btc_up_thresholds"]  # [0.45, 0.55, 0.70]
    factors    = POLY_POSITION_SCALE["unit_mul_factors"]    # [0.0,  0.7,  1.0,  1.5]

    if btc_up_5d < thresholds[0]:
        return base_mul * factors[0]   # 0.0 → 진입 차단
    elif btc_up_5d < thresholds[1]:
        return base_mul * factors[1]   # 0.7x
    elif btc_up_5d < thresholds[2]:
        return base_mul * factors[2]   # 1.0x
    else:
        return base_mul * factors[3]   # 1.5x


# ═══════════════════════════════════════════════════════════════════════════════
# 기술 지표 준비
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """RSI14, MA20, vol_ratio, surge_pct, ADX, DI+, DI- 계산"""
    df = ohlcv.copy()

    df["rsi14"]     = calc_rsi(df["Close"], 14)
    df["ma20"]      = df["Close"].rolling(20).mean()
    df["vol_ma20"]  = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"].replace(0, np.nan)
    df["pct_ma20"]  = (df["Close"] / df["ma20"] - 1) * 100
    df["ret1"]      = df["Close"].pct_change() * 100  # 전일 대비 등락률(%)

    adx, di_plus, di_minus = calc_adx(df["High"], df["Low"], df["Close"], 14)
    df["adx"]      = adx
    df["di_plus"]  = di_plus
    df["di_minus"] = di_minus
    df["di_gap"]   = di_plus - di_minus

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 진입 조건 판단 (공통)
# ═══════════════════════════════════════════════════════════════════════════════

def entry_signal(row: pd.Series, btc_rsi: float, vix: float) -> bool:
    """
    btc_rsi_e 진입 조건 체크.
    True 이면 진입 가능, False 이면 SKIP.
    """
    # 1. DI+ - DI- 갭
    if row["di_gap"] < P["di_min_gap"]:
        return False
    # 2. 전일 급등 필터 (surge_pct 이상 오른 날은 추격 매수 금지)
    if row["ret1"] >= P["surge_pct"]:
        return False
    # 3. BTC RSI 상한
    if pd.isna(btc_rsi) or btc_rsi > P["btc_rsi_max"]:
        return False
    # 4. VIX 범위
    if pd.isna(vix) or not (P["vix_min"] <= vix <= P["vix_max"]):
        return False
    # 5. 가격 ≥ MA20 + pct_ma20_min%
    if pd.isna(row["pct_ma20"]) or row["pct_ma20"] < P["pct_ma20_min"]:
        return False
    # 6. 거래량 배수
    if pd.isna(row["vol_ratio"]) or row["vol_ratio"] < P["vol_ratio_min"]:
        return False
    # 7. RSI14 하한
    if pd.isna(row["rsi14"]) or row["rsi14"] < P["rsi14_min"]:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MDD 계산
# ═══════════════════════════════════════════════════════════════════════════════

def calc_mdd(trades: list[dict]) -> float:
    """거래 리스트에서 최대 낙폭 계산 (누적 KRW 기준)"""
    if not trades:
        return 0.0
    cumulative = np.cumsum([t["pnl_krw"] for t in trades])
    peak       = np.maximum.accumulate(cumulative)
    drawdown   = (cumulative - peak) / np.where(peak != 0, peak, 1) * 100
    return round(float(drawdown.min()), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 백테스트 엔진
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    ticker: str,
    indicators: pd.DataFrame,
    btc_rsi_s: pd.Series,
    vix_s: pd.Series,
    btcup_df: pd.DataFrame,
    use_poly_scale: bool,
) -> list[dict]:
    """
    단일 종목 백테스트.

    Parameters
    ----------
    use_poly_scale : bool
        True → Polymarket scale_unit_mul 적용 (버전 B)
        False → 고정 unit_mul (버전 A)

    Returns
    -------
    list of trade dicts
    """
    version_tag = "poly_scaled" if use_poly_scale else "standard"
    trades: list[dict] = []

    # 활성 포지션 상태
    in_position    = False
    entry_price    = 0.0
    entry_date_ts  = None
    shares         = 0.0
    invested_krw   = 0.0
    pyramid_count  = 0
    peak_price     = 0.0
    unit_mul_used  = P["unit_mul"]  # 진입 시 확정된 unit_mul
    blocked_count  = 0  # poly 차단 횟수 (버전 B 전용)

    dates = indicators.index[
        (indicators.index >= pd.Timestamp(START_DATE)) &
        (indicators.index <= pd.Timestamp(END_DATE))
    ]

    for dt in dates:
        row   = indicators.loc[dt]
        close = row["Close"]

        # ── 보조 지표 조회 ─────────────────────────────────────────────────
        btc_rsi = btc_rsi_s.get(dt, np.nan)
        vix_val = vix_s.get(dt, np.nan)
        btc_up_5d = np.nan
        if not btcup_df.empty and dt in btcup_df.index:
            btc_up_5d = btcup_df.loc[dt, "btc_up_5d"]

        if in_position:
            # ── 보유 중: 청산 조건 체크 ───────────────────────────────────
            hold_days = (dt - entry_date_ts).days

            # peak 갱신
            if close > peak_price:
                peak_price = close

            # 트레일링 손절 기준가 (peak 기준)
            trailing_stop = peak_price * (1 + P["trailing_pct"] / 100)
            # 고정 손절 기준가 (진입가 기준)
            fixed_stop    = entry_price * (1 + P["stop_pct"] / 100)
            # 목표가
            target_price  = entry_price * (1 + P["target_pct"] / 100)

            exit_reason = None
            exit_price  = close

            if close >= target_price:
                exit_reason = "target"
            elif close <= trailing_stop:
                exit_reason = "trailing"
            elif close <= fixed_stop:
                exit_reason = "stoploss"
            elif hold_days >= P["hold_days"]:
                exit_reason = "hold_max"

            if exit_reason:
                # 청산 처리
                pnl_usd = (exit_price - entry_price) * shares
                pnl_krw = round(pnl_usd * EXCHANGE_RATE)
                pnl_pct = round((exit_price / entry_price - 1) * 100, 2)

                trades.append({
                    "ticker":      ticker,
                    "version":     version_tag,
                    "entry_date":  entry_date_ts.date(),
                    "exit_date":   dt.date(),
                    "hold_days":   hold_days,
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price, 4),
                    "shares":      round(shares, 4),
                    "invested_krw": round(invested_krw),
                    "pnl_pct":    pnl_pct,
                    "pnl_krw":    pnl_krw,
                    "exit_reason": exit_reason,
                    "pyramid":    pyramid_count,
                    "unit_mul":   unit_mul_used,
                })

                # 상태 초기화
                in_position   = False
                entry_price   = 0.0
                entry_date_ts = None
                shares        = 0.0
                invested_krw  = 0.0
                pyramid_count = 0
                peak_price    = 0.0
                unit_mul_used = P["unit_mul"]
                continue

            # ── 피라미드 추가 조건 ────────────────────────────────────────
            if pyramid_count < P["max_pyramid"]:
                gain_pct = (close / entry_price - 1) * 100
                if gain_pct >= P["pyramid_add_pct"]:
                    # 피라미드 추가 매수
                    add_krw    = UNIT_KRW * unit_mul_used
                    add_usd    = add_krw / EXCHANGE_RATE
                    add_shares = add_usd / close
                    shares      += add_shares
                    invested_krw += add_krw
                    # 평균 진입가 갱신
                    entry_price = (entry_price * (shares - add_shares) + close * add_shares) / shares
                    pyramid_count += 1

        else:
            # ── 포지션 없음: 진입 조건 체크 ──────────────────────────────
            if pd.isna(close) or close <= 0:
                continue

            if not entry_signal(row, btc_rsi, vix_val):
                continue

            # unit_mul 결정
            if use_poly_scale:
                effective_mul = scale_unit_mul(P["unit_mul"], btc_up_5d)
                if effective_mul == 0.0:
                    # 진입 차단
                    blocked_count += 1
                    continue
            else:
                effective_mul = P["unit_mul"]

            # 진입
            invest_krw = UNIT_KRW * effective_mul
            invest_usd = invest_krw / EXCHANGE_RATE
            new_shares = invest_usd / close

            in_position    = True
            entry_price    = close
            entry_date_ts  = dt
            shares         = new_shares
            invested_krw   = invest_krw
            pyramid_count  = 0
            peak_price     = close
            unit_mul_used  = effective_mul

    # 기간 종료 시 미청산 포지션 강제 청산
    if in_position and entry_date_ts is not None:
        last_dt    = dates[-1]
        last_close = indicators.loc[last_dt, "Close"]
        hold_days  = (last_dt - entry_date_ts).days
        pnl_usd    = (last_close - entry_price) * shares
        pnl_krw    = round(pnl_usd * EXCHANGE_RATE)
        pnl_pct    = round((last_close / entry_price - 1) * 100, 2)

        trades.append({
            "ticker":       ticker,
            "version":      version_tag,
            "entry_date":   entry_date_ts.date(),
            "exit_date":    last_dt.date(),
            "hold_days":    hold_days,
            "entry_price":  round(entry_price, 4),
            "exit_price":   round(last_close, 4),
            "shares":       round(shares, 4),
            "invested_krw": round(invested_krw),
            "pnl_pct":     pnl_pct,
            "pnl_krw":     pnl_krw,
            "exit_reason":  "period_end",
            "pyramid":     pyramid_count,
            "unit_mul":    unit_mul_used,
        })

    # blocked_count를 trades 리스트에 메타로 넣지 않고 별도 반환 필요
    # → trades 자체에 "blocked_count" 키를 더미로 첫 항목에 추가하기보다
    #   글로벌 dict로 관리하는 방식 사용 (run_ticker_comparison에서 처리)
    trades.append({"_meta_blocked": blocked_count, "version": version_tag})
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# 요약 통계
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(trades_raw: list[dict]) -> dict:
    """메타 항목 제거 후 통계 계산"""
    # 메타 항목 분리
    meta    = [t for t in trades_raw if "_meta_blocked" in t]
    trades  = [t for t in trades_raw if "_meta_blocked" not in t]

    blocked = meta[0]["_meta_blocked"] if meta else 0

    if not trades:
        return {
            "n_trades":  0,
            "win_rate":  0.0,
            "avg_pnl":   0.0,
            "total_pnl": 0,
            "mdd":       0.0,
            "blocked":   blocked,
        }

    n       = len(trades)
    wins    = sum(1 for t in trades if t["pnl_krw"] > 0)
    avg_pnl = round(sum(t["pnl_pct"] for t in trades) / n, 2)
    total   = sum(t["pnl_krw"] for t in trades)
    mdd     = calc_mdd(trades)

    return {
        "n_trades":  n,
        "win_rate":  round(wins / n * 100, 1),
        "avg_pnl":   avg_pnl,
        "total_pnl": total,
        "mdd":       mdd,
        "blocked":   blocked,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 종목별 비교 실행
# ═══════════════════════════════════════════════════════════════════════════════

def run_ticker_comparison(
    ticker: str,
    btc_rsi_s: pd.Series,
    vix_s: pd.Series,
    btcup_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """버전 A(표준) + 버전 B(poly scaled) 백테스트 실행"""
    print(f"  [{ticker}] 지표 준비 중...", end="", flush=True)
    ohlcv = load_ohlcv(ticker)
    ind   = prepare_indicators(ohlcv)
    print(" 완료")

    print(f"  [{ticker}] 버전 A (고정 unit_mul) 실행...", end="", flush=True)
    trades_a = run_backtest(ticker, ind, btc_rsi_s, vix_s, btcup_df, use_poly_scale=False)
    stat_a   = summarize(trades_a)
    print(f" 완료 ({stat_a['n_trades']}건)")

    print(f"  [{ticker}] 버전 B (Poly Scale) 실행...", end="", flush=True)
    trades_b = run_backtest(ticker, ind, btc_rsi_s, vix_s, btcup_df, use_poly_scale=True)
    stat_b   = summarize(trades_b)
    print(f" 완료 ({stat_b['n_trades']}건)")

    return stat_a, stat_b


# ═══════════════════════════════════════════════════════════════════════════════
# 출력
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_krw(val: int) -> str:
    """KRW → '만원' 표기"""
    man = val // 10_000
    sign = "+" if man >= 0 else ""
    return f"{sign}{man:,} 만원"


def fmt_pct(val: float, sign: bool = True) -> str:
    prefix = "+" if (sign and val > 0) else ""
    return f"{prefix}{val:.1f}%"


def print_ticker_table(ticker: str, stat_a: dict, stat_b: dict) -> None:
    col_w  = 20
    hdr_a  = "표준 (고정 3.0x)"
    hdr_b  = "Poly Scale 적용"

    rows = [
        ("거래 수",          f"{stat_a['n_trades']}건",                       f"{stat_b['n_trades']}건"),
        ("승률",             fmt_pct(stat_a['win_rate'],  sign=False),         fmt_pct(stat_b['win_rate'],  sign=False)),
        ("평균 P&L",         fmt_pct(stat_a['avg_pnl']),                       fmt_pct(stat_b['avg_pnl'])),
        ("총 P&L",           fmt_krw(stat_a['total_pnl']),                     fmt_krw(stat_b['total_pnl'])),
        ("MDD",              fmt_pct(stat_a['mdd'],       sign=False),          fmt_pct(stat_b['mdd'],       sign=False)),
        ("차단된 진입 (scaled=0)", "-",                                         f"{stat_b['blocked']}건"),
    ]

    print(f"\n종목: {ticker}")
    print(f"  {'':20s}  {hdr_a:<{col_w}}  {hdr_b:<{col_w}}")
    print(f"  {'-'*60}")
    for label, va, vb in rows:
        print(f"  {label:<20s}  {va:<{col_w}}  {vb:<{col_w}}")


def print_summary(results: dict[str, tuple[dict, dict]]) -> None:
    total_a = sum(r[0]["total_pnl"] for r in results.values())
    total_b = sum(r[1]["total_pnl"] for r in results.values())
    mdd_a   = min(r[0]["mdd"]       for r in results.values())
    mdd_b   = min(r[1]["mdd"]       for r in results.values())
    blocked = sum(r[1]["blocked"]    for r in results.values())
    improve = round((total_b - total_a) / abs(total_a) * 100, 1) if total_a != 0 else 0.0

    print("\n" + "=" * 65)
    print("[비교 결론]")
    pnl_diff = total_b - total_a
    diff_sign = "+" if pnl_diff >= 0 else ""
    print(f"  총 P&L 변화 : {fmt_krw(total_a)} → {fmt_krw(total_b)}  ({diff_sign}{improve}%)")
    print(f"  MDD 변화    : {fmt_pct(mdd_a, sign=False)} → {fmt_pct(mdd_b, sign=False)}")
    print(f"  진입 차단으로 손실 회피 후보: {blocked}건")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("Phase C: Polymarket 포지션 스케일 효과 검증")
    print(f"기간: {START_DATE} ~ {END_DATE}  (Polymarket 데이터 구간)")
    print("=" * 65)

    # ── 공통 데이터 로드 ───────────────────────────────────────────────────
    print("\n[데이터 로드]")
    print("  BTC RSI 로드...", end="", flush=True)
    btc_rsi_s = load_btc_rsi()
    print(f" 완료 ({len(btc_rsi_s)}일)")

    print("  VIX 로드...", end="", flush=True)
    vix_s = load_vix()
    print(f" 완료 ({len(vix_s)}일)")

    print("  Polymarket btc_up 로드...", end="", flush=True)
    btcup_df = load_btcup_series(START_DATE, END_DATE)
    if btcup_df.empty:
        print(" [WARNING] Polymarket 데이터 없음 — btc_up_5d = NaN 처리")
    else:
        valid = btcup_df["btc_up_5d"].notna().sum()
        print(f" 완료 ({len(btcup_df)}일, 유효 5d={valid}일)")
        # 분포 요약
        up_mean = btcup_df["btc_up_5d"].mean()
        up_min  = btcup_df["btc_up_5d"].min()
        up_max  = btcup_df["btc_up_5d"].max()
        print(f"  btc_up_5d: min={up_min:.2f}, mean={up_mean:.2f}, max={up_max:.2f}")

    # ── 종목별 백테스트 ────────────────────────────────────────────────────
    print("\n[백테스트 실행]")
    results: dict[str, tuple[dict, dict]] = {}
    for ticker in TICKERS:
        stat_a, stat_b = run_ticker_comparison(ticker, btc_rsi_s, vix_s, btcup_df)
        results[ticker] = (stat_a, stat_b)

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Phase C: Polymarket 포지션 스케일 효과 검증")
    print(f"기간: {START_DATE} ~ {END_DATE}  (Polymarket 데이터 구간)")
    print("=" * 65)

    for ticker in TICKERS:
        stat_a, stat_b = results[ticker]
        print_ticker_table(ticker, stat_a, stat_b)

    print_summary(results)


if __name__ == "__main__":
    main()

"""
Phase B: 종목 Universe 확장 백테스트 — todd_fuck_v1
====================================================
btc_rsi_e 보수적 파라미터 → IREN/CONL/PTIR (기존) + MSTU/BITU/ROBN (신규)
기간: 2023-09-01 ~ 2025-12-31 (Train)

데이터 소스:
- IREN / CONL / PTIR / MSTU / ROBN : data/market/daily/profit_curve_ohlcv.parquet
- BITU                               : data/market/cache/BITU.parquet  (2025-11-14~)
- BTC RSI                            : data/market/daily/btc_mstr_daily.parquet
- VIX                                : data/market/daily/vix_daily.parquet
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_START = "2023-09-01"
TRAIN_END   = "2025-12-31"
FX          = 1350.0  # USD → KRW

TICKERS = ["IREN", "CONL", "PTIR", "MSTU", "BITU", "ROBN"]

# Optuna 최적화 결과 — btc_rsi_e 보수적 파라미터
CONSERVATIVE_PARAMS = {
    "di_min_gap":     7.82,
    "surge_pct":      5.96,
    "btc_rsi_max":   64.62,
    "vix_min":       12.40,
    "vix_max":       34.74,
    "pct_ma20_min":  12.76,
    "vol_ratio_min":  1.42,
    "rsi14_min":     37.16,
    "target_pct":    54.4,
    "stop_pct":     -11.0,
    "hold_days":      50,
    "unit_mul":        3.0,
    "max_pyramid":     2,
    "pyramid_add_pct": 5.7,
    "trailing_pct":   -8.0,
}


# ─────────────────────────────────────────────────────────────
# 기술 지표 계산
# ─────────────────────────────────────────────────────────────
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l_ = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l_.replace(0, np.nan))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Close, High, Low, Volume 컬럼 기반 기술 지표 추가."""
    df = df.copy().sort_values("Date").reset_index(drop=True)
    c = df["Close"]

    df["ma20"]      = c.rolling(20).mean()
    df["pct_ma20"]  = (c - df["ma20"]) / df["ma20"] * 100
    df["rsi14"]     = rsi(c, 14)
    df["ret1"]      = c.pct_change(1) * 100
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # DI+, DI-
    high, low = df["High"], df["Low"]
    tr  = pd.concat(
        [high - low,
         (high - c.shift(1)).abs(),
         (low  - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    dmp = (high - high.shift(1)).clip(lower=0)
    dmp = dmp.where((high - high.shift(1)) > (low.shift(1) - low), 0)
    dmm = (low.shift(1) - low).clip(lower=0)
    dmm = dmm.where((low.shift(1) - low) > (high - high.shift(1)), 0)
    atr = tr.ewm(span=14, adjust=False).mean()
    df["di_plus"]  = dmp.ewm(span=14, adjust=False).mean() / atr * 100
    df["di_minus"] = dmm.ewm(span=14, adjust=False).mean() / atr * 100

    return df


# ─────────────────────────────────────────────────────────────
# 매크로 데이터 구성 (BTC RSI, VIX)
# ─────────────────────────────────────────────────────────────
def build_macro(btc_path: Path, vix_path: Path) -> pd.DataFrame:
    # BTC RSI14
    btc_df = pd.read_parquet(btc_path)
    btc_df["Date"] = pd.to_datetime(btc_df["Date"])
    btc = btc_df[btc_df["ticker"] == "BTC-USD"].copy().sort_values("Date")
    btc["btc_rsi14"] = rsi(btc["Close"], 14)
    btc_sig = btc.set_index("Date")[["btc_rsi14"]]

    # VIX close
    vix_df = pd.read_parquet(vix_path)
    # timestamp는 tz-aware → tz 제거 후 date 파싱
    vix_df["Date"] = pd.to_datetime(vix_df["timestamp"]).dt.tz_localize(None).dt.normalize()
    vix_sig = vix_df.sort_values("Date").set_index("Date")[["close"]].rename(
        columns={"close": "vix"}
    )

    macro = btc_sig.join(vix_sig, how="outer").sort_index().ffill()
    return macro


# ─────────────────────────────────────────────────────────────
# 개별 종목 DataFrame 준비
# ─────────────────────────────────────────────────────────────
def prep_ticker(
    raw: pd.DataFrame,
    macro: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """raw: Date, Open, High, Low, Close, Volume 포함 단일 종목 DF."""
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # tz-aware 인덱스 처리
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 30:
        return None

    df = add_features(df)

    # 매크로 병합
    macro_reset = macro.reset_index().rename(columns={"index": "Date", "Date": "Date"})
    # macro index name이 'Date'인 경우 대응
    if macro.index.name == "Date":
        macro_reset = macro.reset_index()
    else:
        macro_reset = macro.reset_index().rename(columns={macro.index.name: "Date"})

    df = df.merge(macro_reset, on="Date", how="left")
    df["btc_rsi14"] = df["btc_rsi14"].ffill()
    df["vix"]       = df["vix"].ffill()

    # 기간 필터
    df = df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))].copy()
    return df if len(df) > 0 else None


# ─────────────────────────────────────────────────────────────
# 백테스트 엔진
# ─────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, p: dict, fx: float = 1350.0) -> dict:
    """
    df 컬럼: Date, Close, High, Low, Volume, pct_ma20, di_plus, di_minus,
             rsi14, vol_ratio, btc_rsi14, vix, ret1, ma20
    p: CONSERVATIVE_PARAMS
    반환: dict(n, pnl_krw, wr, avg_pct, trades)
    """
    unit_usd = 740 * p["unit_mul"]
    tgt      = p["target_pct"]
    stp      = p["stop_pct"]
    hld      = p["hold_days"]
    pyr      = p["max_pyramid"]
    pyr_add  = p["pyramid_add_pct"]
    trail    = p.get("trailing_pct")

    def entry_ok(row) -> bool:
        return (
            row.get("btc_rsi14", 99) <= p["btc_rsi_max"]
            and p["vix_min"] <= row.get("vix", 20) <= p["vix_max"]
            and (row["di_plus"] - row["di_minus"]) >= p["di_min_gap"]
            and row["rsi14"] >= p["rsi14_min"]
            and row["vol_ratio"] >= p["vol_ratio_min"]
            and row["pct_ma20"] >= p["pct_ma20_min"]
            and row.get("ret1", 0) >= p["surge_pct"]
        )

    trades: list[dict] = []
    pos:    list[tuple] = []  # (entry_date, qty, entry_price)
    peak:   float | None = None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")):
            continue
        price = row["Close"]
        d     = row["Date"]

        if pos:
            tq  = sum(x[1] for x in pos)
            tc  = sum(x[1] * x[2] for x in pos)
            avg = tc / tq
            pp  = (price - avg) / avg * 100
            held = (d - pos[0][0]).days

            if peak is None or price > peak:
                peak = price
            trail_hit = (
                trail is not None
                and peak is not None
                and (price - peak) / peak * 100 <= trail
            )

            if pp >= tgt or pp <= stp or held >= hld or trail_hit or row.get("pct_ma20", 0) < -35:
                trades.append({
                    "date":    str(d.date()),
                    "entry":   round(avg, 2),
                    "exit":    round(price, 2),
                    "pnl_pct": round(pp, 2),
                    "held":    held,
                    "layers":  len(pos),
                    "pnl_krw": round((tq * price - tc) * fx),
                })
                pos, peak = [], None
                continue

            # 피라미딩
            if len(pos) < pyr and price > pos[-1][2] * (1 + pyr_add / 100) and entry_ok(row):
                pos.append((d, unit_usd * 0.7 / price, price))

        elif entry_ok(row):
            pos.append((d, unit_usd / price, price))
            peak = price

    if not trades:
        return {"n": 0, "pnl_krw": 0, "wr": 0.0, "avg_pct": 0.0, "trades": []}

    tdf = pd.DataFrame(trades)
    return {
        "n":       len(tdf),
        "pnl_krw": int(tdf["pnl_krw"].sum()),
        "wr":      round((tdf["pnl_krw"] > 0).mean() * 100, 1),
        "avg_pct": round(tdf["pnl_pct"].mean(), 2),
        "trades":  trades,
    }


# ─────────────────────────────────────────────────────────────
# 데이터 로드 헬퍼
# ─────────────────────────────────────────────────────────────
def load_ohlcv_all() -> dict[str, pd.DataFrame]:
    """각 ticker별 raw OHLCV DataFrame 반환."""
    OHLCV_PATH  = ROOT / "data" / "market" / "daily" / "profit_curve_ohlcv.parquet"
    BITU_CACHE  = ROOT / "data" / "market" / "cache" / "BITU.parquet"

    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])

    result: dict[str, pd.DataFrame] = {}
    for t in TICKERS:
        if t == "BITU":
            # BITU는 cache에서 로드 (2025-11-14 ~)
            bitu = pd.read_parquet(BITU_CACHE)
            bitu["Date"] = pd.to_datetime(bitu["Date"])
            result[t] = bitu[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        else:
            sub = ohlcv[ohlcv["ticker"] == t].copy()
            result[t] = sub[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    return result


# ─────────────────────────────────────────────────────────────
# 출력 포매터
# ─────────────────────────────────────────────────────────────
LEGACY_TICKERS = {"IREN", "CONL", "PTIR"}


def print_results(results: dict[str, dict], data_ranges: dict[str, tuple]) -> None:
    sep  = "=" * 65
    line = "-" * 65

    print()
    print(sep)
    print("Phase B: 종목 Universe 확장 백테스트 (btc_rsi_e 보수적 파라미터)")
    print(f"기간: {TRAIN_START} ~ {TRAIN_END}")
    print(sep)
    print()
    print(f"{'종목':<8}{'거래수':>6}  {'WR%':>6}  {'평균P&L':>8}  {'총P&L(만원)':>12}  {'실제기간':>22}  비고")
    print(line)

    for t in TICKERS:
        r = results[t]
        n   = r["n"]
        wr  = f"{r['wr']:.1f}%" if n > 0 else "  -  "
        avg = f"{r['avg_pct']:+.1f}%" if n > 0 else "  -  "
        pnl = r["pnl_krw"] // 10000  # 만원 단위
        pnl_str = f"{pnl:+,.0f}" if n > 0 else "  -  "

        rng = data_ranges.get(t, ("?", "?"))
        rng_str = f"{rng[0]} ~ {rng[1]}"

        note = "기존" if t in LEGACY_TICKERS else "신규"
        print(
            f"{t:<8}{n:>6}  {wr:>6}  {avg:>8}  {pnl_str:>12}  {rng_str:>22}  {note}"
        )

    print(line)

    # 합계
    total_pnl  = sum(results[t]["pnl_krw"] for t in TICKERS)
    total_n    = sum(results[t]["n"]        for t in TICKERS)
    total_wins = sum(
        round(results[t]["n"] * results[t]["wr"] / 100)
        for t in TICKERS
        if results[t]["n"] > 0
    )
    total_wr = (total_wins / total_n * 100) if total_n > 0 else 0.0
    total_pnl_man = total_pnl // 10000

    print(
        f"{'합계':<8}{total_n:>6}  {total_wr:>5.1f}%"
        f"  {'':>8}  {total_pnl_man:>+12,.0f}  {'':>22}"
    )
    print()

    # 거래 상세
    for t in TICKERS:
        r = results[t]
        if r["n"] == 0:
            print(f"[{t}] 거래 없음")
            continue
        print(f"[{t}] 거래 상세 ({r['n']}건):")
        for tr in r["trades"]:
            sign = "+" if tr["pnl_pct"] >= 0 else ""
            pnl_man = tr["pnl_krw"] // 10000
            print(
                f"  {tr['date']}  진입={tr['entry']:.2f}  청산={tr['exit']:.2f}"
                f"  P&L={sign}{tr['pnl_pct']:.1f}%  보유={tr['held']}일"
                f"  {pnl_man:+,.0f}만원"
            )
        print()


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    BTC_PATH = ROOT / "data" / "market" / "daily" / "btc_mstr_daily.parquet"
    VIX_PATH = ROOT / "data" / "market" / "daily" / "vix_daily.parquet"

    print("데이터 로드 중...")
    macro     = build_macro(BTC_PATH, VIX_PATH)
    raw_data  = load_ohlcv_all()

    print("백테스트 실행 중...")
    results:     dict[str, dict]  = {}
    data_ranges: dict[str, tuple] = {}

    for t in TICKERS:
        raw = raw_data[t]
        df  = prep_ticker(raw, macro, TRAIN_START, TRAIN_END)

        if df is None or len(df) == 0:
            print(f"  {t}: 데이터 없음 (TRAIN 기간 내 해당 없음)")
            results[t]     = {"n": 0, "pnl_krw": 0, "wr": 0.0, "avg_pct": 0.0, "trades": []}
            data_ranges[t] = ("데이터없음", "")
            continue

        d_start = str(df["Date"].min().date())
        d_end   = str(df["Date"].max().date())
        data_ranges[t] = (d_start, d_end)
        print(f"  {t}: {d_start} ~ {d_end}  ({len(df)}일)")

        results[t] = run_backtest(df, CONSERVATIVE_PARAMS, fx=FX)

    print_results(results, data_ranges)


if __name__ == "__main__":
    main()

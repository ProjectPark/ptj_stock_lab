"""
MSTU Mania 백테스트 — BTC RSI > 68 모멘텀 추종 전략

발견:
  MSTU 급등일(surge >= 5.96% + pct_ma20 >= 12.76%)의 BTC RSI14 평균: 77.0
  기존 btc_rsi_e(max=64.62)로는 24건 중 2건만 통과 → MSTU 실질 0거래
  → BTC RSI > 68 구간에서 MSTU 전용 진입 설계
"""

import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── 파라미터 정의 ──────────────────────────────────────────────

class ManiaParams(NamedTuple):
    label: str
    btc_rsi_min: float    # BTC RSI 하한 (핵심 조건, 기존과 반대)
    btc_rsi_max: float    # BTC RSI 상한 (극단 과열 방지)
    surge_pct: float      # 당일 상승률 최소
    pct_ma20_min: float   # MA20 대비 최소
    di_gap_min: float     # DI 갭 최소
    vol_ratio_min: float  # 거래량 비율 최소
    target_pct: float
    stop_pct: float
    hold_days: int


PARAM_GRID = [
    ManiaParams("mania_v1_기준",     68, 90,  5.96, 12.76, 7.82, 1.42, 40, -15, 30),
    ManiaParams("mania_v2_완화",     68, 90,  5.00, 10.00, 5.00, 1.20, 40, -15, 30),
    ManiaParams("mania_v3_넓은범위", 65, 90,  5.96, 12.76, 7.82, 1.00, 40, -15, 30),
    ManiaParams("mania_v4_고목표",   70, 90,  5.96, 12.76, 7.82, 1.42, 60, -15, 20),
    ManiaParams("mania_v5_균형",     70, 88,  5.00, 10.00, 5.00, 1.20, 50, -12, 25),
    ManiaParams("mania_v6_최완화",   65, 95,  3.00,  8.00, 3.00, 1.00, 35, -15, 35),
]


# ── 기술 지표 ──────────────────────────────────────────────────

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    dmp = (high - high.shift(1)).clip(lower=0).where(
        (high - high.shift(1)) > (low.shift(1) - low), 0
    )
    dmm = (low.shift(1) - low).clip(lower=0).where(
        (low.shift(1) - low) > (high - high.shift(1)), 0
    )
    atr = tr.ewm(span=n, adjust=False).mean()
    di_plus = dmp.ewm(span=n, adjust=False).mean() / atr * 100
    di_minus = dmm.ewm(span=n, adjust=False).mean() / atr * 100
    return di_plus, di_minus


# ── 데이터 로드 & 전처리 ───────────────────────────────────────

def load_data() -> pd.DataFrame:
    # 1. MSTU OHLCV
    raw = pd.read_parquet(ROOT / "data/market/daily/profit_curve_ohlcv.parquet")
    mstu = raw[raw["ticker"] == "MSTU"].copy().reset_index(drop=True)
    mstu = mstu.sort_values("Date").reset_index(drop=True)

    # 2. BTC RSI14
    btc_raw = pd.read_parquet(ROOT / "data/market/daily/btc_mstr_daily.parquet")
    btc = btc_raw[btc_raw["ticker"] == "BTC-USD"].copy()
    btc = btc.sort_values("Date").reset_index(drop=True)
    btc["btc_rsi14"] = rsi(btc["Close"], 14)
    btc_rsi = btc[["Date", "btc_rsi14"]].copy()

    # 3. VIX
    vix_raw = pd.read_parquet(ROOT / "data/market/daily/vix_daily.parquet")
    vix_raw["date_norm"] = (
        pd.to_datetime(vix_raw["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    vix = vix_raw[["date_norm", "close"]].rename(
        columns={"date_norm": "Date", "close": "vix"}
    )

    # 4. MSTU 지표 계산
    mstu["ma20"] = mstu["Close"].rolling(20).mean()
    mstu["pct_ma20"] = (mstu["Close"] - mstu["ma20"]) / mstu["ma20"] * 100
    mstu["vol_ratio"] = mstu["Volume"] / mstu["Volume"].rolling(20).mean()
    mstu["ret1"] = mstu["Close"].pct_change() * 100  # 당일 수익률 (%)

    di_plus, di_minus = compute_di(mstu["High"], mstu["Low"], mstu["Close"])
    mstu["di_gap"] = di_plus - di_minus

    # 5. BTC RSI, VIX 병합 (ffill로 주말 등 누락 처리)
    mstu["Date"] = pd.to_datetime(mstu["Date"])
    btc_rsi["Date"] = pd.to_datetime(btc_rsi["Date"])
    vix["Date"] = pd.to_datetime(vix["Date"])

    df = mstu.merge(btc_rsi, on="Date", how="left")
    df = df.merge(vix, on="Date", how="left")
    df["btc_rsi14"] = df["btc_rsi14"].ffill()
    df["vix"] = df["vix"].ffill()

    return df


# ── 백테스트 엔진 ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, p: ManiaParams, fx: float = 1350.0) -> dict:
    """
    df 컬럼: Date, Close, High, Low, Volume, ma20, pct_ma20, di_gap,
             rsi14, vol_ratio, btc_rsi14, vix, ret1
    """
    unit_usd = 740 * 3.0  # 기본 unit_mul=3.0

    def entry_ok(row) -> bool:
        btc_rsi = row.get("btc_rsi14", np.nan)
        vix_val = row.get("vix", np.nan)
        if pd.isna(btc_rsi) or pd.isna(vix_val):
            return False
        return (
            btc_rsi >= p.btc_rsi_min           # BTC RSI 최소 (핵심 반전)
            and btc_rsi <= p.btc_rsi_max        # BTC RSI 최대 (극단 방지)
            and vix_val <= 35                   # 극단 공포 회피
            and row["di_gap"] >= p.di_gap_min
            and row["vol_ratio"] >= p.vol_ratio_min
            and row["pct_ma20"] >= p.pct_ma20_min
            and row.get("ret1", 0) >= p.surge_pct
        )

    trades, pos, peak = [], [], None

    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")):
            continue
        price = row["Close"]
        d = row["Date"]

        if pos:
            tq = sum(x[1] for x in pos)
            tc = sum(x[1] * x[2] for x in pos)
            avg = tc / tq
            pp = (price - avg) / avg * 100
            held = (d - pos[0][0]).days

            if peak is None or price > peak:
                peak = price

            trail_hit = peak is not None and (price - peak) / peak * 100 <= -10.0

            if (
                pp >= p.target_pct
                or pp <= p.stop_pct
                or held >= p.hold_days
                or trail_hit
            ):
                if pp >= p.target_pct:
                    reason = "TARGET"
                elif pp <= p.stop_pct:
                    reason = "STOP"
                elif trail_hit:
                    reason = "TRAIL"
                else:
                    reason = "HOLD"

                trades.append(
                    {
                        "date": str(d.date()),
                        "entry": round(avg, 2),
                        "exit": round(price, 2),
                        "pnl_pct": round(pp, 2),
                        "held": held,
                        "exit_reason": reason,
                        "pnl_krw": round((tq * price - tc) * fx),
                    }
                )
                pos, peak = [], None
                continue

        elif not pos and entry_ok(row):
            pos.append((d, unit_usd / price, price))
            peak = price

    if not trades:
        return {"n": 0, "pnl_krw": 0, "wr": 0.0, "avg_pct": 0.0, "trades": []}

    tdf = pd.DataFrame(trades)
    return {
        "n": len(tdf),
        "pnl_krw": int(tdf["pnl_krw"].sum()),
        "wr": round((tdf["pnl_krw"] > 0).mean() * 100, 1),
        "avg_pct": round(tdf["pnl_pct"].mean(), 2),
        "trades": trades,
    }


# ── IREN 기존 btc_rsi_e (비교용) ─────────────────────────────

def run_iren_reference(start_date: str = "2024-09-18", fx: float = 1350.0) -> dict:
    """
    IREN btc_rsi_e 기존 전략 재현 (동일 기간으로 재계산)
    btc_rsi_max=64.62, surge_pct=5.96, pct_ma20=12.76, di_gap=7.82, vol_ratio=1.42
    target=40, stop=-15, hold=30
    """
    raw = pd.read_parquet(ROOT / "data/market/daily/profit_curve_ohlcv.parquet")
    iren = raw[raw["ticker"] == "IREN"].copy().reset_index(drop=True)
    iren = iren.sort_values("Date").reset_index(drop=True)

    btc_raw = pd.read_parquet(ROOT / "data/market/daily/btc_mstr_daily.parquet")
    btc = btc_raw[btc_raw["ticker"] == "BTC-USD"].copy().sort_values("Date").reset_index(drop=True)
    btc["btc_rsi14"] = rsi(btc["Close"], 14)
    btc_rsi = btc[["Date", "btc_rsi14"]].copy()

    vix_raw = pd.read_parquet(ROOT / "data/market/daily/vix_daily.parquet")
    vix_raw["date_norm"] = (
        pd.to_datetime(vix_raw["timestamp"]).dt.tz_localize(None).dt.normalize()
    )
    vix_df = vix_raw[["date_norm", "close"]].rename(
        columns={"date_norm": "Date", "close": "vix"}
    )

    iren["ma20"] = iren["Close"].rolling(20).mean()
    iren["pct_ma20"] = (iren["Close"] - iren["ma20"]) / iren["ma20"] * 100
    iren["vol_ratio"] = iren["Volume"] / iren["Volume"].rolling(20).mean()
    iren["ret1"] = iren["Close"].pct_change() * 100

    di_plus, di_minus = compute_di(iren["High"], iren["Low"], iren["Close"])
    iren["di_gap"] = di_plus - di_minus

    iren["Date"] = pd.to_datetime(iren["Date"])
    btc_rsi["Date"] = pd.to_datetime(btc_rsi["Date"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    df = iren.merge(btc_rsi, on="Date", how="left")
    df = df.merge(vix_df, on="Date", how="left")
    df["btc_rsi14"] = df["btc_rsi14"].ffill()
    df["vix"] = df["vix"].ffill()

    # 동일 기간 필터
    df = df[df["Date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

    # btc_rsi_e 조건: BTC RSI <= 64.62 (기존 전략)
    unit_usd = 740 * 3.0
    unit_usd_full = 740  # btc_rsi_e는 unit_mul=1.0 기준으로 추정 (원 보고서: 9건 1114만원)
    # 원 보고서와 근사하기 위해 unit_mul=3.0 유지

    def entry_ok_iren(row) -> bool:
        btc_rsi_val = row.get("btc_rsi14", np.nan)
        vix_val = row.get("vix", np.nan)
        if pd.isna(btc_rsi_val) or pd.isna(vix_val):
            return False
        return (
            btc_rsi_val <= 64.62
            and vix_val <= 35
            and row["di_gap"] >= 7.82
            and row["vol_ratio"] >= 1.42
            and row["pct_ma20"] >= 12.76
            and row.get("ret1", 0) >= 5.96
        )

    trades, pos, peak = [], [], None
    for _, row in df.iterrows():
        if pd.isna(row.get("ma20")):
            continue
        price = row["Close"]
        d = row["Date"]

        if pos:
            tq = sum(x[1] for x in pos)
            tc = sum(x[1] * x[2] for x in pos)
            avg = tc / tq
            pp = (price - avg) / avg * 100
            held = (d - pos[0][0]).days
            if peak is None or price > peak:
                peak = price
            trail_hit = peak is not None and (price - peak) / peak * 100 <= -10.0
            if pp >= 40 or pp <= -15 or held >= 30 or trail_hit:
                reason = (
                    "TARGET" if pp >= 40
                    else "STOP" if pp <= -15
                    else "TRAIL" if trail_hit
                    else "HOLD"
                )
                trades.append(
                    {
                        "date": str(d.date()),
                        "entry": round(avg, 2),
                        "exit": round(price, 2),
                        "pnl_pct": round(pp, 2),
                        "held": held,
                        "exit_reason": reason,
                        "pnl_krw": round((tq * price - tc) * fx),
                    }
                )
                pos, peak = [], None
                continue
        elif not pos and entry_ok_iren(row):
            pos.append((d, unit_usd / price, price))
            peak = price

    if not trades:
        return {"n": 0, "pnl_krw": 0, "wr": 0.0, "avg_pct": 0.0, "trades": []}

    tdf = pd.DataFrame(trades)
    return {
        "n": len(tdf),
        "pnl_krw": int(tdf["pnl_krw"].sum()),
        "wr": round((tdf["pnl_krw"] > 0).mean() * 100, 1),
        "avg_pct": round(tdf["pnl_pct"].mean(), 2),
        "trades": trades,
    }


# ── 메인 ─────────────────────────────────────────────────────

def main():
    sep = "=" * 65

    print(sep)
    print("MSTU Mania 백테스트 — BTC RSI > 68 모멘텀 추종 전략")
    print(sep)
    print()
    print("분석 배경:")
    print("  MSTU 급등일(24건)의 BTC RSI14 평균: 77.0")
    print("  기존 btc_rsi_e(max=64.62)로는 24건 중 2건만 통과 → 0거래")
    print("  → BTC RSI > 68 구간에서 MSTU 전용 진입 설계")
    print()

    # 데이터 로드
    df = load_data()

    # 기간 필터: 2024-09-18 ~ 2025-12-31
    START = "2024-09-18"
    END = "2025-12-31"
    df_bt = df[
        (df["Date"] >= pd.to_datetime(START)) & (df["Date"] <= pd.to_datetime(END))
    ].reset_index(drop=True)

    print(f"데이터 기간: {START} ~ {END}")
    print(f"MSTU 총 거래일: {len(df_bt)}일")
    print(
        f"BTC RSI14 유효 행: {df_bt['btc_rsi14'].notna().sum()}일 "
        f"(NaN: {df_bt['btc_rsi14'].isna().sum()}일)"
    )
    print()

    # IREN 비교 기준 (동일 기간)
    print("IREN btc_rsi_e 기준 재계산 중 (동일 기간)...")
    iren_ref = run_iren_reference(start_date=START)

    print(sep)
    print(f"파라미터별 비교 (MSTU, {START}~{END})")
    print(sep)
    header = f"{'설정':<20} {'거래수':>5} {'WR%':>6} {'평균P&L':>8} {'총P&L(만원)':>12}"
    print(header)
    print("-" * 65)

    results = []
    for p in PARAM_GRID:
        r = run_backtest(df_bt, p)
        results.append((p, r))
        pnl_man = r["pnl_krw"] / 10000
        sign = "+" if pnl_man >= 0 else ""
        avg_sign = "+" if r["avg_pct"] >= 0 else ""
        print(
            f"{p.label:<20} {r['n']:>5}건 {r['wr']:>5.1f}%"
            f"  {avg_sign}{r['avg_pct']:>6.1f}%  {sign}{pnl_man:>10,.1f}"
        )

    # IREN 비교 출력
    print("-" * 65)
    iren_pnl = iren_ref["pnl_krw"] / 10000
    iren_sign = "+" if iren_pnl >= 0 else ""
    iren_avg_sign = "+" if iren_ref["avg_pct"] >= 0 else ""
    print(
        f"{'[IREN btc_rsi_e 기준]':<20} {iren_ref['n']:>5}건 {iren_ref['wr']:>5.1f}%"
        f"  {iren_avg_sign}{iren_ref['avg_pct']:>6.1f}%  {iren_sign}{iren_pnl:>10,.1f}"
    )
    print(sep)
    print()

    # 최고 파라미터 찾기 (총 P&L 기준)
    best_p, best_r = max(results, key=lambda x: x[1]["pnl_krw"])

    if best_r["n"] > 0:
        print(sep)
        print(f"[최고 파라미터 상세 거래] — {best_p.label}")
        print(
            f"  BTC RSI: {best_p.btc_rsi_min}~{best_p.btc_rsi_max}  |  "
            f"surge: {best_p.surge_pct}%  |  pct_ma20: {best_p.pct_ma20_min}%"
        )
        print(
            f"  di_gap: {best_p.di_gap_min}  |  vol_ratio: {best_p.vol_ratio_min}  |  "
            f"target/stop/hold: {best_p.target_pct}/{best_p.stop_pct}/{best_p.hold_days}"
        )
        print(sep)
        print(
            f"{'날짜':<12} {'진입가':>8} {'청산가':>8} {'P&L%':>7} "
            f"{'보유일':>5} {'사유':<8} {'P&L(원)':>10}"
        )
        print("-" * 65)
        for t in best_r["trades"]:
            sign = "+" if t["pnl_pct"] >= 0 else ""
            print(
                f"{t['date']:<12} ${t['entry']:>7.2f}  ${t['exit']:>7.2f}"
                f"  {sign}{t['pnl_pct']:>5.1f}%  {t['held']:>4}일  {t['exit_reason']:<8}"
                f"  {t['pnl_krw']:>+10,.0f}"
            )
        print("-" * 65)
        best_pnl = best_r["pnl_krw"] / 10000
        print(
            f"  합계: {best_r['n']}건  WR={best_r['wr']}%  "
            f"avg={'+' if best_r['avg_pct']>=0 else ''}{best_r['avg_pct']}%  "
            f"총P&L={'+' if best_pnl>=0 else ''}{best_pnl:,.1f}만원"
        )
        print(sep)
    else:
        print(f"\n[주의] 최고 파라미터({best_p.label})에서 거래 0건 — 진입 조건 점검 필요")

    # 요약 비교
    print()
    print("== 요약 비교 ==")
    best_pnl_man = best_r["pnl_krw"] / 10000
    print(
        f"  MSTU Mania 최고: {best_p.label}  "
        f"{best_r['n']}건 / WR {best_r['wr']}% / "
        f"{'+' if best_pnl_man>=0 else ''}{best_pnl_man:,.1f}만원"
    )
    print(
        f"  IREN btc_rsi_e:  {iren_ref['n']}건 / WR {iren_ref['wr']}% / "
        f"{'+' if iren_pnl>=0 else ''}{iren_pnl:,.1f}만원  (동일 기간 {START}~{END})"
    )

    # IREN 상세 거래 (참고용)
    if iren_ref["n"] > 0:
        print()
        print(sep)
        print(f"[IREN btc_rsi_e 상세 거래] ({START}~{END})")
        print(sep)
        print(
            f"{'날짜':<12} {'진입가':>8} {'청산가':>8} {'P&L%':>7} "
            f"{'보유일':>5} {'사유':<8} {'P&L(원)':>10}"
        )
        print("-" * 65)
        for t in iren_ref["trades"]:
            sign = "+" if t["pnl_pct"] >= 0 else ""
            print(
                f"{t['date']:<12} ${t['entry']:>7.2f}  ${t['exit']:>7.2f}"
                f"  {sign}{t['pnl_pct']:>5.1f}%  {t['held']:>4}일  {t['exit_reason']:<8}"
                f"  {t['pnl_krw']:>+10,.0f}"
            )
        print("-" * 65)
        print(
            f"  합계: {iren_ref['n']}건  WR={iren_ref['wr']}%  "
            f"avg={'+' if iren_ref['avg_pct']>=0 else ''}{iren_ref['avg_pct']}%  "
            f"총P&L={'+' if iren_pnl>=0 else ''}{iren_pnl:,.1f}만원"
        )
        print(sep)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Study F — BB %B < 0 구간 분석 (V-바운스 패턴 근거 마련)
=========================================================
핵심 질문:
  Q1. 진입 시점 BB %B < 0 포지션은 실제로 어떤 회복 경로를 보이는가?
  Q2. crash 하락폭 × BB %B 구간 조합별로 성과가 얼마나 다른가?
  Q3. v1 DCA 예외 조건(%B < 0.15 + crash -10% + R14)은 통계적으로 유효한가?

Phase 1 EDA : 거래내역에서 진입 시점 %B 구간별 포지션 추출 → 일별 회복 경로
Phase 2 Grid: crash 하락폭 × %B 구간 × 회복 기준 3D 분석
Phase 3 V-바운스: %B < 0.15 + crash < -10% 조합 집중 분석

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_f_bb_vbounce.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 데이터 로드
# ============================================================

def load_data():
    trades_path = _PROJECT_ROOT / "data" / "results" / "backtests" / "d2s_trades.csv"
    market_path = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily.parquet"

    if not trades_path.exists():
        raise FileNotFoundError(f"trades 파일 없음: {trades_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"market 파일 없음: {market_path}")

    trades = pd.read_csv(trades_path, parse_dates=["date"])
    market = pd.read_parquet(market_path)
    market.index = pd.to_datetime(market.index)

    closes = market.xs("Close", axis=1, level=1)
    trading_dates = sorted(closes.index)

    print(f"  거래 데이터: {len(trades)}건")
    print(f"  시장 데이터: {len(market)} days, {len(closes.columns)} tickers")

    return trades, closes, trading_dates


# ============================================================
# BB %B 계산 (진입 시점)
# ============================================================

def calc_bb_pct_b(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """볼린저밴드 %B 계산."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = (series - lower) / (upper - lower)
    return pct_b


def build_entry_bb_table(trades: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """각 매수 거래의 진입 시점 BB %B를 계산해서 붙인다."""
    buys = trades[trades["side"] == "BUY"].copy()
    rows = []

    for ticker in buys["ticker"].unique():
        if ticker not in closes.columns:
            continue
        close_series = closes[ticker].dropna()
        if len(close_series) < 25:
            continue

        bb = calc_bb_pct_b(close_series)

        ticker_buys = buys[buys["ticker"] == ticker].copy()
        for _, row in ticker_buys.iterrows():
            entry_date = row["date"]
            if entry_date not in bb.index:
                # 가장 가까운 날짜 찾기
                prior = bb.index[bb.index <= entry_date]
                if prior.empty:
                    continue
                entry_date_key = prior[-1]
            else:
                entry_date_key = entry_date

            bb_val = bb.loc[entry_date_key]
            if pd.isna(bb_val):
                continue

            rows.append({
                "ticker":      ticker,
                "entry_date":  row["date"].date() if hasattr(row["date"], "date") else row["date"],
                "entry_price": row["price"],
                "bb_pct_b":    float(bb_val),
                "reason":      row.get("reason", ""),
                "score":       row.get("score", 0.0),
            })

    return pd.DataFrame(rows)


# ============================================================
# Phase 1 EDA — %B 구간별 진입 후 회복 경로
# ============================================================

def phase1_eda(trades: pd.DataFrame, closes: pd.DataFrame, trading_dates: list) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  Phase 1 EDA — BB %B 구간별 진입 후 회복 경로")
    print("=" * 70)

    entry_bb = build_entry_bb_table(trades, closes)
    print(f"\n  진입 시점 BB %B 계산 완료: {len(entry_bb)}건")

    # %B 구간 정의
    bins   = [-np.inf, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50, np.inf]
    labels = ["<-0.20", "-0.20~-0.10", "-0.10~0", "0~0.10",
              "0.10~0.20", "0.20~0.30", "0.30~0.50", ">0.50"]
    entry_bb["bb_zone"] = pd.cut(entry_bb["bb_pct_b"], bins=bins, labels=labels)

    # 진입 후 1/2/3/5/7일 수익률 추출
    HOLD_DAYS = [1, 2, 3, 5, 7]
    results = []

    for _, row in entry_bb.iterrows():
        ticker     = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])
        entry_price = row["entry_price"]

        if ticker not in closes.columns:
            continue

        # 진입일 이후 trading_dates 인덱스 찾기
        future_dates = [d for d in trading_dates if d > entry_date]
        if not future_dates:
            continue

        pnl_by_day = {}
        for hd in HOLD_DAYS:
            if len(future_dates) >= hd:
                exit_date = future_dates[hd - 1]
                exit_price = closes.loc[exit_date, ticker] if exit_date in closes.index else np.nan
                if pd.notna(exit_price) and entry_price > 0:
                    pnl_by_day[f"pnl_d{hd}"] = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_by_day[f"pnl_d{hd}"] = np.nan
            else:
                pnl_by_day[f"pnl_d{hd}"] = np.nan

        results.append({**row.to_dict(), **pnl_by_day})

    df = pd.DataFrame(results)

    # 구간별 통계
    print(f"\n  {'BB구간':<15} {'n':>5} {'d1avg':>8} {'d3avg':>8} {'d5avg':>8} "
          f"{'d5win%':>8} {'d7avg':>8}")
    print("  " + "-" * 65)

    zone_stats = []
    for zone in labels:
        sub = df[df["bb_zone"] == zone]
        n = len(sub)
        if n == 0:
            continue
        d1  = sub["pnl_d1"].mean()
        d3  = sub["pnl_d3"].mean()
        d5  = sub["pnl_d5"].mean()
        d7  = sub["pnl_d7"].mean()
        d5w = (sub["pnl_d5"] > 0).mean() * 100 if sub["pnl_d5"].notna().sum() > 0 else 0

        mark = " ★" if zone in ["<-0.20", "-0.20~-0.10", "-0.10~0"] else ""
        print(f"  {zone:<15} {n:>5} {d1:>+8.2f}% {d3:>+8.2f}% {d5:>+8.2f}% "
              f"{d5w:>7.1f}% {d7:>+8.2f}%{mark}")

        zone_stats.append({
            "bb_zone": zone, "n": n,
            "pnl_d1": d1, "pnl_d3": d3, "pnl_d5": d5, "pnl_d7": d7,
            "win_d5": d5w,
        })

    # %B < 0 전체 요약
    below_zero = df[df["bb_pct_b"] < 0]
    above_zero = df[df["bb_pct_b"] >= 0]
    print(f"\n  [요약] %B < 0 구간 (n={len(below_zero)}):")
    for hd in HOLD_DAYS:
        col = f"pnl_d{hd}"
        avg = below_zero[col].mean()
        win = (below_zero[col] > 0).mean() * 100
        print(f"    d{hd:>2}: 평균 {avg:>+6.2f}%  승률 {win:>5.1f}%")

    print(f"\n  [요약] %B >= 0 구간 (n={len(above_zero)}):")
    for hd in HOLD_DAYS:
        col = f"pnl_d{hd}"
        avg = above_zero[col].mean()
        win = (above_zero[col] > 0).mean() * 100
        print(f"    d{hd:>2}: 평균 {avg:>+6.2f}%  승률 {win:>5.1f}%")

    df.to_csv(RESULTS_DIR / "study_f_phase1_entry_bb.csv", index=False)
    return df


# ============================================================
# Phase 2 Grid — crash 하락폭 × %B 구간 교차 분석
# ============================================================

def phase2_grid(entry_df: pd.DataFrame, closes: pd.DataFrame, trading_dates: list):
    print("\n" + "=" * 70)
    print("  Phase 2 Grid — crash 하락폭 × %B 구간 교차 분석")
    print("=" * 70)

    # crash: 진입일 기준 직전 5일 최대 하락폭 계산
    crash_vals = []
    for _, row in entry_df.iterrows():
        ticker     = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])
        if ticker not in closes.columns:
            crash_vals.append(np.nan)
            continue
        prior = closes[ticker].loc[:entry_date].tail(6)  # 진입일 포함 6일
        if len(prior) < 2:
            crash_vals.append(np.nan)
            continue
        peak = prior[:-1].max()
        trough = prior.min()
        crash = (trough - peak) / peak * 100 if peak > 0 else np.nan
        crash_vals.append(crash)

    entry_df = entry_df.copy()
    entry_df["crash_5d"] = crash_vals

    # crash 구간 × BB %B 구간
    crash_thresholds = [0, -3, -5, -8, -10, -12, -15]
    bb_thresholds    = [0.0, 0.10, 0.20, 0.30]

    print(f"\n  {'crash':>8}  {'bb_max':>8}  {'n':>5}  {'d3avg':>8}  {'d5avg':>8}  "
          f"{'d5win%':>8}  {'d7avg':>8}")
    print("  " + "-" * 68)

    grid_rows = []
    for crash_th in crash_thresholds:
        for bb_th in bb_thresholds:
            sub = entry_df[
                (entry_df["crash_5d"] <= crash_th) &
                (entry_df["bb_pct_b"]  < bb_th)
            ]
            n = len(sub)
            if n < 3:
                continue
            d3  = sub["pnl_d3"].mean()
            d5  = sub["pnl_d5"].mean()
            d7  = sub["pnl_d7"].mean()
            d5w = (sub["pnl_d5"] > 0).mean() * 100

            vbounce_mark = " ★ V-BOUNCE" if crash_th <= -10 and bb_th <= 0.20 else ""
            print(f"  {crash_th:>7}%  {bb_th:>8.2f}  {n:>5}  {d3:>+8.2f}%  "
                  f"{d5:>+8.2f}%  {d5w:>7.1f}%  {d7:>+8.2f}%{vbounce_mark}")

            grid_rows.append({
                "crash_th": crash_th, "bb_th": bb_th, "n": n,
                "pnl_d3": d3, "pnl_d5": d5, "pnl_d7": d7, "win_d5": d5w,
            })

    pd.DataFrame(grid_rows).to_csv(RESULTS_DIR / "study_f_phase2_grid.csv", index=False)


# ============================================================
# Phase 3 — V-바운스 조건 집중 분석
# ============================================================

def phase3_vbounce(entry_df: pd.DataFrame, closes: pd.DataFrame, trading_dates: list):
    print("\n" + "=" * 70)
    print("  Phase 3 — V-바운스 조건 집중 분석 (%B < 0.15 + crash < -10%)")
    print("=" * 70)

    # crash_5d가 있어야 함 (phase2에서 생성)
    if "crash_5d" not in entry_df.columns:
        print("  [!] crash_5d 컬럼 없음 — Phase 2 먼저 실행 필요")
        return

    # V-바운스 후보
    vb = entry_df[
        (entry_df["bb_pct_b"] < 0.15) &
        (entry_df["crash_5d"] <= -10)
    ].copy()

    non_vb = entry_df[
        ~(
            (entry_df["bb_pct_b"] < 0.15) &
            (entry_df["crash_5d"] <= -10)
        )
    ].copy()

    print(f"\n  V-바운스 후보: {len(vb)}건 / 전체 {len(entry_df)}건")
    print(f"  비-V-바운스:   {len(non_vb)}건")

    # V-바운스 일별 성과
    print(f"\n  [V-바운스 후보 성과]")
    for hd in [1, 2, 3, 5, 7]:
        col = f"pnl_d{hd}"
        if col not in vb.columns or vb[col].isna().all():
            continue
        avg = vb[col].mean()
        win = (vb[col] > 0).mean() * 100
        rec2 = (vb[col] > 2).mean() * 100   # +2% 회복
        rec5 = (vb[col] > 5).mean() * 100   # +5% 회복
        print(f"    d{hd:>2}: 평균 {avg:>+6.2f}%  승률 {win:>5.1f}%  "
              f"+2%회복 {rec2:>5.1f}%  +5%회복 {rec5:>5.1f}%")

    # 비교: 비-V-바운스 성과
    print(f"\n  [비-V-바운스 성과 (비교)]")
    for hd in [1, 2, 3, 5, 7]:
        col = f"pnl_d{hd}"
        if col not in non_vb.columns or non_vb[col].isna().all():
            continue
        avg = non_vb[col].mean()
        win = (non_vb[col] > 0).mean() * 100
        print(f"    d{hd:>2}: 평균 {avg:>+6.2f}%  승률 {win:>5.1f}%")

    # 2일 내 +2% 회복 성공/실패 분포
    print(f"\n  [V-바운스 2일 내 회복 여부]")
    if "pnl_d2" in vb.columns:
        rec_ok  = vb[vb["pnl_d2"] >= 2]
        rec_fail = vb[vb["pnl_d2"] < 2]
        print(f"    2일 내 +2% 회복 성공: {len(rec_ok)}건 ({len(rec_ok)/max(1,len(vb))*100:.1f}%)")
        if len(rec_ok) > 0:
            print(f"      → 최종 d5 평균: {rec_ok['pnl_d5'].mean():>+.2f}%  승률: {(rec_ok['pnl_d5']>0).mean()*100:.1f}%")
        print(f"    2일 내 +2% 미회복:   {len(rec_fail)}건 ({len(rec_fail)/max(1,len(vb))*100:.1f}%)")
        if len(rec_fail) > 0:
            print(f"      → 최종 d5 평균: {rec_fail['pnl_d5'].mean():>+.2f}%  승률: {(rec_fail['pnl_d5']>0).mean()*100:.1f}%")

    # 결론 요약
    print(f"\n  [결론]")
    if len(vb) >= 5:
        d5_avg = vb["pnl_d5"].mean()
        d5_win = (vb["pnl_d5"] > 0).mean() * 100
        base_d5_avg = non_vb["pnl_d5"].mean()
        base_d5_win = (non_vb["pnl_d5"] > 0).mean() * 100
        diff_avg = d5_avg - base_d5_avg
        diff_win = d5_win - base_d5_win
        print(f"    V-바운스 d5 승률 {d5_win:.1f}% vs 기준 {base_d5_win:.1f}% "
              f"(차이 {diff_win:>+.1f}%p)")
        print(f"    V-바운스 d5 평균 {d5_avg:>+.2f}% vs 기준 {base_d5_avg:>+.2f}% "
              f"(차이 {diff_avg:>+.2f}%p)")
        if diff_win > 5:
            print("    → V-바운스 조건 유효: 일반 진입 대비 통계적 우위 확인")
        elif diff_win > 0:
            print("    → V-바운스 조건 약한 우위: 추가 검증 필요")
        else:
            print("    → V-바운스 조건 효과 없음: 일반 진입과 유사하거나 열등")
    else:
        print(f"    V-바운스 후보 {len(vb)}건 — 샘플 부족 (n < 5), 결론 보류")

    if len(vb) > 0:
        vb.to_csv(RESULTS_DIR / "study_f_phase3_vbounce.csv", index=False)


# ============================================================
# main
# ============================================================

def main():
    print("=" * 70)
    print("  Study F — BB %B < 0 구간 분석 (V-바운스 패턴 근거 마련)")
    print("=" * 70)

    print("\n[1/4] 데이터 로드")
    trades, closes, trading_dates = load_data()

    print("\n[2/4] 진입 시점 BB %B 계산")
    entry_df = build_entry_bb_table(trades, closes)
    print(f"  계산 완료: {len(entry_df)}건")

    print("\n[3/4] Phase 1 — %B 구간별 회복 경로")
    entry_df = phase1_eda(trades, closes, trading_dates)

    print("\n[4/4] Phase 2 + Phase 3")

    # crash_5d 계산 (phase2에서 entry_df에 추가)
    crash_vals = []
    for _, row in entry_df.iterrows():
        ticker     = row["ticker"]
        entry_date = pd.Timestamp(row["entry_date"])
        if ticker not in closes.columns:
            crash_vals.append(np.nan)
            continue
        prior = closes[ticker].loc[:entry_date].tail(6)
        if len(prior) < 2:
            crash_vals.append(np.nan)
            continue
        peak   = prior[:-1].max()
        trough = prior.min()
        crash  = (trough - peak) / peak * 100 if peak > 0 else np.nan
        crash_vals.append(crash)
    entry_df["crash_5d"] = crash_vals

    phase2_grid(entry_df, closes, trading_dates)
    phase3_vbounce(entry_df, closes, trading_dates)

    print("\n" + "=" * 70)
    print("  Study F 완료")
    print(f"  결과 저장: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

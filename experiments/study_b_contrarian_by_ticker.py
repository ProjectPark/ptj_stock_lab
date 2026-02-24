"""
Study B — 역발상 진입 종목별 정밀화
====================================
v1 규칙서 분석: MSTU 85% 승률(역발상 59%) vs CONL 48% 승률(순발상 62%)
목표: 종목별로 최적 진입 조건(역발상 임계값, 하락폭 구간)을 차별화한다.

분석 항목
---------
B1. 진입 시 ticker_pct 구간별 승률 — 종목별 비교
B2. 역발상 임계값 탐색 — 최적 하락폭 컷오프 (MSTU vs CONL vs ROBN vs AMDL)
B3. 역발상 × 시황 교차 분석 — R14(리스크오프) 발동 여부에 따른 종목별 차이
B4. 역발상 × 요일 교차 분석 — 금요일 역발상 효과
B5. 진입 후 forward return 프로파일 — 1d/3d/5d/7d by ticker
B6. 종목별 최적 진입 조건 요약

입력
----
- data/market/ohlcv/backtest_1min_3y.parquet (3년 1분봉)
- Study A split 보정 로직 재사용

출력
----
- data/results/analysis/study_b_by_ticker.csv     — 종목별 구간 승률
- data/results/analysis/study_b_threshold.csv     — 종목별 최적 임계값
- data/results/analysis/study_b_crossanalysis.csv — 교차 분석 결과
- data/results/analysis/study_b_events.csv        — 이벤트 로그
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_1MIN = ROOT / "data/market/ohlcv/backtest_1min_3y.parquet"
OUT_DIR = ROOT / "data/results/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 분석 대상 종목
TARGET_TICKERS = ["MSTU", "CONL", "ROBN", "AMDL"]
MARKET_REFS = ["SPY", "GLD"]

# forward return 보유 기간
HOLD_DAYS = [1, 3, 5, 7]


# ─── Step 0: Split 보정 (Study A 재사용) ──────────────────────────────────────

STANDARD_SPLIT_RATIOS = [2.0, 4.0, 5.0, 10.0, 0.5, 0.25, 0.1, 0.2, 0.05]


def detect_splits(daily: pd.DataFrame) -> pd.DataFrame:
    """overnight gap > 80% 이상 → split 이벤트 탐지."""
    rows = []
    for sym, df in daily.groupby("symbol"):
        df = df.sort_values("date").reset_index(drop=True)
        df["prev_close"] = df["close"].shift(1)
        df["gap_ratio"] = df["open"] / df["prev_close"]

        for _, row in df.iterrows():
            if pd.isna(row["gap_ratio"]):
                continue
            g = row["gap_ratio"]
            if g < 0.1 or g > 10:
                # 극단 비율 — 표준 배수와 ±15% 이내인지 확인
                for std in STANDARD_SPLIT_RATIOS:
                    if abs(g - std) / std <= 0.15:
                        rows.append(
                            {
                                "date": row["date"],
                                "symbol": sym,
                                "gap_ratio": round(g, 4),
                                "adj_factor": round(1.0 / g, 6),
                                "matched_std": std,
                            }
                        )
                        break
    split_df = pd.DataFrame(rows)
    if not split_df.empty:
        split_df = split_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return split_df


def apply_split_adjustment(daily: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """split 이벤트 기준 과거 데이터에 adj_factor 소급 적용."""
    daily = daily.copy()
    daily["adj_close"] = daily["close"].astype(float)

    for sym, splits in split_df.groupby("symbol"):
        mask = daily["symbol"] == sym
        sym_dates = daily.loc[mask, "date"].values
        splits_sorted = splits.sort_values("date", ascending=False)  # 최신 → 과거

        cum_factor = 1.0
        for _, sp in splits_sorted.iterrows():
            cum_factor *= sp["adj_factor"]
            pre_split = sym_dates < sp["date"]
            daily.loc[mask & (daily["date"].isin(sym_dates[pre_split])), "adj_close"] *= cum_factor
            cum_factor = 1.0  # 각 구간별 독립 계산 (누적 아님)

        # 다중 split 시 정방향 재처리
        if len(splits_sorted) > 1:
            adj = daily.loc[mask].sort_values("date").copy()
            adj["adj_close"] = adj["close"].astype(float)
            split_dates = sorted(splits_sorted["date"].tolist())
            factors = {row["date"]: row["adj_factor"] for _, row in splits_sorted.iterrows()}

            cum = 1.0
            for sd in reversed(split_dates):
                cum *= factors[sd]
                adj.loc[adj["date"] < sd, "adj_close"] = (
                    adj.loc[adj["date"] < sd, "close"] * cum
                )
            daily.loc[mask, "adj_close"] = adj["adj_close"].values

    return daily


def load_daily(path: Path) -> pd.DataFrame:
    """1분봉 → 일봉 집계 + split 보정."""
    print("[Step 0] 1분봉 데이터 로드...")
    df = pd.read_parquet(path)
    all_syms = TARGET_TICKERS + MARKET_REFS
    df = df[df["symbol"].isin(all_syms)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    # 일봉 집계
    daily = (
        df.groupby(["symbol", "date"])
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )

    # split 보정
    split_df = detect_splits(daily)
    if not split_df.empty:
        print(f"  → Split 감지 {len(split_df)}건: {split_df[['symbol','date','gap_ratio']].to_string(index=False)}")
        split_df.to_csv(OUT_DIR / "study_b_split_log.csv", index=False)
    daily = apply_split_adjustment(daily, split_df)

    daily["pct"] = daily.groupby("symbol")["adj_close"].pct_change() * 100
    print(f"  → 일봉 집계 완료: {len(daily)}행, {daily.date.min().date()}~{daily.date.max().date()}")
    return daily


# ─── Step 1: 이벤트 구성 ──────────────────────────────────────────────────────

def build_events(daily: pd.DataFrame) -> pd.DataFrame:
    """종목별 일봉 등락률 + 시황 변수(SPY/GLD) 결합."""
    pivot = daily.pivot(index="date", columns="symbol", values="pct")

    # 시황 변수
    spy_pct = pivot.get("SPY", pd.Series(dtype=float))
    gld_pct = pivot.get("GLD", pd.Series(dtype=float))

    # R14 신호
    riskoff = (gld_pct > 0) & (spy_pct >= -1.5) & (spy_pct < 0)
    riskoff_level2 = (gld_pct >= 0.5) & (gld_pct < 1.0) & (spy_pct >= -0.5) & (spy_pct < -0.5)
    # Study A 연속 신호
    riskoff_streak = riskoff.astype(int)
    streak_col = []
    cur = 0
    for v in riskoff_streak:
        cur = cur + 1 if v == 1 else 0
        streak_col.append(cur)
    riskoff_streak_series = pd.Series(streak_col, index=riskoff.index)

    # 종목별 adj_close
    adj_pivot = daily.pivot(index="date", columns="symbol", values="adj_close")

    rows = []
    for ticker in TARGET_TICKERS:
        if ticker not in pivot.columns:
            continue
        ticker_pct = pivot[ticker]
        adj_price = adj_pivot.get(ticker)

        for date in pivot.index:
            tp = ticker_pct.get(date)
            sp = spy_pct.get(date)
            gp = gld_pct.get(date)
            ap = adj_price.get(date) if adj_price is not None else np.nan

            if pd.isna(tp) or pd.isna(sp) or pd.isna(gp):
                continue

            # 요일
            weekday = date.weekday()  # 0=월, 4=금

            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "ticker_pct": tp,
                    "spy_pct": sp,
                    "gld_pct": gp,
                    "adj_close": ap,
                    "is_riskoff": bool(riskoff.get(date, False)),
                    "riskoff_streak": int(riskoff_streak_series.get(date, 0)),
                    "weekday": weekday,
                    "is_friday": weekday == 4,
                    # 역발상 구분
                    "is_contrarian": tp < 0,
                    "is_momentum": tp >= 0,
                }
            )

    events = pd.DataFrame(rows)
    return events


# ─── Step 2: forward return 계산 ──────────────────────────────────────────────

def add_forward_returns(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """이벤트 당일 adj_close 기준 N일 후 수익률 계산."""
    adj_pivot = daily.pivot(index="date", columns="symbol", values="adj_close")
    dates = sorted(adj_pivot.index.tolist())
    date_idx = {d: i for i, d in enumerate(dates)}

    for hd in HOLD_DAYS:
        ret_col = f"ret_{hd}d"
        events[ret_col] = np.nan

    for row_i, row in events.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        entry = row["adj_close"]
        if pd.isna(entry) or entry <= 0 or ticker not in adj_pivot.columns:
            continue
        idx = date_idx.get(date)
        if idx is None:
            continue
        for hd in HOLD_DAYS:
            target_idx = idx + hd
            if target_idx < len(dates):
                exit_date = dates[target_idx]
                exit_price = adj_pivot.at[exit_date, ticker] if exit_date in adj_pivot.index else np.nan
                if not pd.isna(exit_price) and exit_price > 0:
                    events.at[row_i, f"ret_{hd}d"] = (exit_price / entry - 1) * 100

    return events


# ─── Step 3: B1 종목별 구간 승률 분석 ────────────────────────────────────────

def analyze_b1_by_ticker(events: pd.DataFrame) -> pd.DataFrame:
    """종목별 ticker_pct 구간(역발상 vs 순발상)별 성과."""
    bins = [-50, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 50]
    labels = ["-5%↓", "-3~-5%", "-2~-3%", "-1~-2%", "-0.5~-1%", "-0.5~0%",
              "0~0.5%", "0.5~1%", "1~2%", "2~3%", "3~5%", "5%↑"]
    events = events.copy()
    events["pct_bin"] = pd.cut(events["ticker_pct"], bins=bins, labels=labels)

    rows = []
    for ticker in TARGET_TICKERS:
        sub = events[events["ticker"] == ticker]
        for pct_bin, grp in sub.groupby("pct_bin", observed=True):
            n = len(grp)
            if n < 3:
                continue
            ret5 = grp["ret_5d"].dropna()
            rows.append({
                "ticker": ticker,
                "pct_bin": str(pct_bin),
                "n": n,
                "win_rate": round((ret5 > 0).mean() * 100, 1),
                "avg_ret": round(ret5.mean(), 2),
                "median_ret": round(ret5.median(), 2),
            })
    return pd.DataFrame(rows)


# ─── Step 4: B2 역발상 최적 임계값 탐색 ──────────────────────────────────────

def analyze_b2_threshold(events: pd.DataFrame) -> pd.DataFrame:
    """종목별로 -N% 이하 진입 시 5일 승률이 최대화되는 임계값 탐색."""
    thresholds = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
    rows = []
    for ticker in TARGET_TICKERS:
        sub = events[events["ticker"] == ticker]
        for thr in thresholds:
            grp = sub[sub["ticker_pct"] <= thr]
            n = len(grp)
            if n < 5:
                continue
            ret5 = grp["ret_5d"].dropna()
            rows.append({
                "ticker": ticker,
                "threshold": thr,
                "n": n,
                "win_rate": round((ret5 > 0).mean() * 100, 1),
                "avg_ret": round(ret5.mean(), 2),
                "median_ret": round(ret5.median(), 2),
            })
    return pd.DataFrame(rows)


# ─── Step 5: B3 역발상 × 시황 교차 분석 ─────────────────────────────────────

def analyze_b3_riskoff_cross(events: pd.DataFrame) -> pd.DataFrame:
    """역발상 진입 × R14 리스크오프 여부 교차 성과."""
    rows = []
    for ticker in TARGET_TICKERS:
        sub = events[events["ticker"] == ticker]
        for riskoff_flag in [False, True]:
            for contrarian_flag in [False, True]:
                grp = sub[
                    (sub["is_riskoff"] == riskoff_flag) &
                    (sub["is_contrarian"] == contrarian_flag)
                ]
                n = len(grp)
                if n < 3:
                    continue
                ret5 = grp["ret_5d"].dropna()
                rows.append({
                    "ticker": ticker,
                    "is_riskoff": riskoff_flag,
                    "is_contrarian": contrarian_flag,
                    "n": n,
                    "win_rate": round((ret5 > 0).mean() * 100, 1),
                    "avg_ret": round(ret5.mean(), 2),
                })
    return pd.DataFrame(rows)


# ─── Step 6: B4 요일 × 역발상 교차 ──────────────────────────────────────────

def analyze_b4_weekday_cross(events: pd.DataFrame) -> pd.DataFrame:
    """요일 × 역발상 여부 교차 성과."""
    day_names = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금"}
    rows = []
    for ticker in TARGET_TICKERS:
        sub = events[events["ticker"] == ticker]
        for day, day_name in day_names.items():
            for contrarian_flag in [False, True]:
                grp = sub[
                    (sub["weekday"] == day) &
                    (sub["is_contrarian"] == contrarian_flag)
                ]
                n = len(grp)
                if n < 3:
                    continue
                ret5 = grp["ret_5d"].dropna()
                rows.append({
                    "ticker": ticker,
                    "weekday": day_name,
                    "is_contrarian": contrarian_flag,
                    "n": n,
                    "win_rate": round((ret5 > 0).mean() * 100, 1),
                    "avg_ret": round(ret5.mean(), 2),
                })
    return pd.DataFrame(rows)


# ─── Step 7: 종목별 최적 임계값 요약 ─────────────────────────────────────────

def summarize_optimal(b1: pd.DataFrame, b2: pd.DataFrame) -> pd.DataFrame:
    """종목별 최적 역발상 진입 조건 도출."""
    rows = []
    for ticker in TARGET_TICKERS:
        # 전체 contrarian 성과
        b2_sub = b2[b2["ticker"] == ticker]
        if b2_sub.empty:
            rows.append({"ticker": ticker, "optimal_threshold": "N/A"})
            continue
        # win_rate 최대화 기준 최적 임계값
        best = b2_sub.loc[b2_sub["win_rate"].idxmax()]
        rows.append({
            "ticker": ticker,
            "optimal_threshold": best["threshold"],
            "optimal_n": int(best["n"]),
            "optimal_win_rate": best["win_rate"],
            "optimal_avg_ret": best["avg_ret"],
        })
    return pd.DataFrame(rows)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    # 데이터 로드
    daily = load_daily(DATA_1MIN)

    # 이벤트 구성
    print("[Step 1] 이벤트 구성...")
    events = build_events(daily)
    print(f"  → 이벤트: {len(events)}건 ({events.ticker.nunique()}개 종목)")

    # forward return 추가
    print("[Step 2] Forward return 계산...")
    events = add_forward_returns(events, daily)

    # B1: 구간별 승률
    print("[Step 3] B1: 종목별 pct 구간 승률...")
    b1 = analyze_b1_by_ticker(events)
    print(b1.to_string(index=False))

    # B2: 임계값 탐색
    print("\n[Step 4] B2: 역발상 임계값 탐색...")
    b2 = analyze_b2_threshold(events)

    # 핵심 요약 출력
    print("\n=== 종목별 역발상 임계값 성과 ===")
    for ticker in TARGET_TICKERS:
        sub = b2[b2["ticker"] == ticker]
        if not sub.empty:
            print(f"\n--- {ticker} ---")
            print(sub[["threshold", "n", "win_rate", "avg_ret"]].to_string(index=False))

    # B3: 리스크오프 교차
    print("\n[Step 5] B3: 역발상 × 리스크오프 교차...")
    b3 = analyze_b3_riskoff_cross(events)
    print(b3.to_string(index=False))

    # B4: 요일 교차
    print("\n[Step 6] B4: 요일 × 역발상 교차...")
    b4 = analyze_b4_weekday_cross(events)
    # 금요일만 요약
    fri = b4[b4["weekday"] == "금"]
    print("금요일 역발상 효과:")
    print(fri.to_string(index=False))

    # 최적 임계값 요약
    print("\n[Step 7] 종목별 최적 역발상 조건 요약:")
    summary = summarize_optimal(b1, b2)
    print(summary.to_string(index=False))

    # 저장
    events.to_csv(OUT_DIR / "study_b_events.csv", index=False)
    b1.to_csv(OUT_DIR / "study_b_by_ticker.csv", index=False)
    b2.to_csv(OUT_DIR / "study_b_threshold.csv", index=False)

    cross_df = pd.concat([
        b3.assign(analysis="riskoff_cross"),
        b4.assign(analysis="weekday_cross"),
    ], ignore_index=True)
    cross_df.to_csv(OUT_DIR / "study_b_crossanalysis.csv", index=False)

    print(f"\n✓ 결과 저장: {OUT_DIR}")
    print("  - study_b_events.csv")
    print("  - study_b_by_ticker.csv")
    print("  - study_b_threshold.csv")
    print("  - study_b_crossanalysis.csv")


if __name__ == "__main__":
    main()

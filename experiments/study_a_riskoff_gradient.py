#!/usr/bin/env python3
"""
Study A — GLD↑+SPY↓ 리스크오프 강도 그라데이션 분석
=====================================================
목표:
  - v1 1-5절의 단순 binary(GLD↑+SPY↓ → 86.4%) 신호를 3단계 레벨로 정밀화
  - GLD 상승폭 × SPY 하락폭의 3×3 매트릭스로 승률/수익률 차등화
  - 연속 리스크오프 발생일수 효과 분석

데이터:
  - data/market/ohlcv/backtest_1min_3y.parquet (2023~2026, 1분봉)
  - 시황 지표: GLD, SPY
  - 성과 측정 종목: MSTU, CONL, ROBN, AMDL

가격 보정 (Step 0):
  - overnight gap |gap| > 80% → split 이벤트 자동 감지
  - split 비율이 표준 배수(2/5/10/20)에 ±15% 이내인 경우에만 보정 적용
  - 보정 방식: 최신 가격 기준, 과거로 누적 조정계수 소급 적용
  - 감지된 MSTU split 2건만 자동 보정 (CONL/AMDL 이상값은 실제 시장 이벤트)

출력:
  - data/results/analysis/study_a_riskoff_matrix.csv      — 3×3 매트릭스
  - data/results/analysis/study_a_riskoff_consecutive.csv — 연속 발생 효과
  - data/results/analysis/study_a_riskoff_events.csv      — 이벤트 원본 로그
  - data/results/analysis/study_a_split_log.csv           — split 감지/보정 로그

사용법:
  pyenv shell ptj_stock_lab && python experiments/study_a_riskoff_gradient.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OHLCV_PATH = DATA / "market" / "ohlcv" / "backtest_1min_3y.parquet"
OUT_DIR = DATA / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# split 감지 설정
# ─────────────────────────────────────────────────────────────────────────────
# overnight gap이 이 임계값을 초과하면 split 후보로 간주
SPLIT_GAP_THRESHOLD = 0.80          # |overnight_gap| > 80%

# 표준 split 배수 (forward: 0.05~0.5, reverse: 2~20)
STANDARD_SPLIT_RATIOS = [0.05, 0.1, 0.125, 0.2, 0.25, 0.5, 2.0, 4.0, 5.0, 8.0, 10.0, 20.0]
RATIO_TOLERANCE = 0.15              # 표준 배수와 ±15% 이내면 split으로 확정

# 분석 대상
SIGNAL_TICKERS = ["GLD", "SPY"]       # 시황 판단용
TARGET_TICKERS = ["MSTU", "CONL", "ROBN", "AMDL"]  # 성과 측정용
ALL_TICKERS = SIGNAL_TICKERS + TARGET_TICKERS

# 이후 수익률 측정 기간 (거래일 기준)
FORWARD_DAYS = [1, 3, 5, 7]
# v1 최적 보유 기간 5일을 기준 성과로 사용
PRIMARY_FORWARD = 5

# 3×3 매트릭스 구간 정의
GLD_BINS = [0.0, 0.5, 1.0, np.inf]           # GLD 상승폭 하한 (%)
GLD_LABELS = ["0~0.5%", "0.5~1.0%", "1.0%+"]

SPY_BINS = [-np.inf, -1.5, -0.5, 0.0]        # SPY 하락폭 상한 (%)
SPY_LABELS = ["-1.5%+", "-0.5~-1.5%", "0~-0.5%"]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def sep(title: str, width: int = 72) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def sub(title: str, width: int = 50) -> None:
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 0-A. Split 이벤트 감지
# ─────────────────────────────────────────────────────────────────────────────
def detect_splits(daily: pd.DataFrame) -> pd.DataFrame:
    """
    overnight gap(전일 종가 → 당일 시가)이 SPLIT_GAP_THRESHOLD를 초과하는
    날짜를 감지하고, STANDARD_SPLIT_RATIOS와 대조해 split 여부를 확정한다.

    반환 컬럼:
        symbol, date, prev_close, curr_open, raw_ratio,
        matched_ratio, is_confirmed, adj_factor, note
    """
    daily = daily.sort_values(["symbol", "date"]).copy()
    daily["prev_close"] = daily.groupby("symbol")["close"].shift(1)
    daily["overnight_gap"] = daily["open"] / daily["prev_close"] - 1

    candidates = daily[daily["overnight_gap"].abs() > SPLIT_GAP_THRESHOLD].copy()

    rows = []
    for _, row in candidates.iterrows():
        raw_ratio = row["open"] / row["prev_close"]  # e.g. 0.1093 or 10.256

        # 가장 가까운 표준 배수 탐색
        closest = min(STANDARD_SPLIT_RATIOS, key=lambda r: abs(r - raw_ratio) / r)
        rel_err = abs(closest - raw_ratio) / closest
        is_confirmed = rel_err <= RATIO_TOLERANCE

        rows.append({
            "symbol":        row["symbol"],
            "date":          row["date"],
            "prev_close":    round(row["prev_close"], 4),
            "curr_open":     round(row["open"], 4),
            "raw_ratio":     round(raw_ratio, 4),
            "matched_ratio": closest,
            "rel_err":       round(rel_err, 4),
            "is_confirmed":  is_confirmed,
            "adj_factor":    round(raw_ratio, 6) if is_confirmed else None,
            "note": (
                f"split ×{closest:.2g} (err {rel_err*100:.1f}%)" if is_confirmed
                else f"비표준 비율 {raw_ratio:.3f} — 보정 제외"
            ),
        })

    split_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol","date","prev_close","curr_open",
                 "raw_ratio","matched_ratio","rel_err",
                 "is_confirmed","adj_factor","note"]
    )
    return split_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 0-B. Split 조정 가격 적용
# ─────────────────────────────────────────────────────────────────────────────
def apply_split_adjustment(daily: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """
    감지된 split 이벤트를 최신 가격 기준으로 소급 보정한다.

    보정 방식:
      - 각 split 날짜 이전의 close 가격에 adj_factor를 곱한다.
      - split이 여러 개 있을 경우 과거로 갈수록 누적 조정계수가 적용된다.
      - 가장 최근 가격(마지막 split 이후)이 기준(무보정)이 된다.

    adj_close 컬럼이 추가된 daily DataFrame 반환.
    """
    daily = daily.copy()
    daily["adj_close"] = daily["close"].astype(float)

    confirmed = split_df[split_df["is_confirmed"]].copy()
    if confirmed.empty:
        return daily

    for symbol, grp in confirmed.groupby("symbol"):
        # 날짜 내림차순 정렬 (최신 → 과거 순서로 누적 적용)
        splits_desc = grp.sort_values("date", ascending=False)

        sym_mask = daily["symbol"] == symbol
        cumulative_factor = 1.0

        for _, sp in splits_desc.iterrows():
            # 이 split 이전 데이터에 현재까지의 누적 조정계수 × adj_factor 적용
            cumulative_factor *= sp["adj_factor"]
            before_mask = sym_mask & (daily["date"] < sp["date"])
            daily.loc[before_mask, "adj_close"] *= cumulative_factor

    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Step 0-C. 보정 결과 검증 출력
# ─────────────────────────────────────────────────────────────────────────────
def validate_adjustment(daily: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """
    split 날짜 전후 원본 close vs adj_close를 출력해
    가격 연속성 확인.
    """
    confirmed = split_df[split_df["is_confirmed"]]
    if confirmed.empty:
        print("  보정 대상 split 없음")
        return

    for _, sp in confirmed.iterrows():
        sym, sp_date = sp["symbol"], sp["date"]
        window = daily[daily["symbol"] == sym].sort_values("date")
        near = window[
            (window["date"] >= sp_date - pd.Timedelta(days=5)) &
            (window["date"] <= sp_date + pd.Timedelta(days=3))
        ][["date", "close", "adj_close"]]

        print(f"\n  [{sym}] split 날짜={sp_date}  adj_factor={sp['adj_factor']:.4f}  ({sp['note']})")
        print(near.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1. 1분봉 → 일봉 집계
# ─────────────────────────────────────────────────────────────────────────────
def load_daily(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1분봉 데이터를 읽어 종목별 일봉으로 집계한 뒤 split 보정을 적용한다.

    반환:
        daily     — open / close / adj_close / pct 포함 DataFrame
        split_log — 감지된 split 이벤트 로그 DataFrame
    """
    sep("Step 1 — 1분봉 로드 및 일봉 집계")
    print(f"  파일: {path}")
    raw = pd.read_parquet(path, columns=["symbol", "timestamp", "open", "close"])

    # timestamp → date (ET 기준)
    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.date
    raw = raw[raw["symbol"].isin(ALL_TICKERS)]

    # 일봉 집계: 첫 open, 마지막 close
    grp = raw.groupby(["symbol", "date"])
    daily = pd.DataFrame({
        "open":  grp["open"].first(),
        "close": grp["close"].last(),
    }).reset_index()

    # date를 datetime으로 통일 (이후 비교에서 타입 오류 방지)
    daily["date"] = pd.to_datetime(daily["date"])

    print(f"  종목: {sorted(daily['symbol'].unique())}")
    print(f"  기간: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"  총 행: {len(daily):,}")

    # ── Step 0: 가격 보정 ──────────────────────────────────────────────────
    sep("Step 0 — 가격 불일치 감지 및 Split 보정")

    split_log = detect_splits(daily)
    print(f"  split 후보: {len(split_log)}건")
    if not split_log.empty:
        print(split_log[["symbol","date","raw_ratio","matched_ratio",
                          "is_confirmed","note"]].to_string(index=False))

    confirmed_n = split_log["is_confirmed"].sum() if not split_log.empty else 0
    print(f"\n  확정 보정 대상: {confirmed_n}건")

    daily = apply_split_adjustment(daily, split_log)

    sub("보정 전후 가격 연속성 검증")
    validate_adjustment(daily, split_log)

    # pct: 장중 수익률 (split 영향 없음 — 당일 open 기준)
    daily["pct"] = (daily["close"] - daily["open"]) / daily["open"] * 100

    return daily, split_log


# ─────────────────────────────────────────────────────────────────────────────
# Step 2. 리스크오프 이벤트 식별
# ─────────────────────────────────────────────────────────────────────────────
def build_signal_df(daily: pd.DataFrame) -> pd.DataFrame:
    """
    날짜별 GLD_pct, SPY_pct를 Wide format으로 정리.
    리스크오프 조건: GLD_pct > 0 AND SPY_pct < 0
    """
    pivot = daily[daily["symbol"].isin(SIGNAL_TICKERS)].pivot(
        index="date", columns="symbol", values="pct"
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={"GLD": "GLD_pct", "SPY": "SPY_pct"})
    pivot = pivot.dropna(subset=["GLD_pct", "SPY_pct"])

    # 리스크오프 binary
    pivot["is_riskoff"] = (pivot["GLD_pct"] > 0) & (pivot["SPY_pct"] < 0)

    # 연속 리스크오프 일수 (현재 날짜 포함)
    streak = []
    count = 0
    for flag in pivot["is_riskoff"]:
        if flag:
            count += 1
        else:
            count = 0
        streak.append(count)
    pivot["riskoff_streak"] = streak

    # GLD 강도 레벨 (리스크오프 이벤트에서만 의미 있음)
    pivot["gld_bin"] = pd.cut(
        pivot["GLD_pct"],
        bins=GLD_BINS,
        labels=GLD_LABELS,
        right=False,
    )
    # SPY 강도 레벨 (음수 구간, bins 오름차순 유지)
    pivot["spy_bin"] = pd.cut(
        pivot["SPY_pct"],
        bins=SPY_BINS,
        labels=SPY_LABELS,
    )

    print(f"\n[Step 2] 리스크오프 이벤트 식별")
    total_days = len(pivot)
    riskoff_days = pivot["is_riskoff"].sum()
    print(f"  전체 거래일: {total_days}일")
    print(f"  리스크오프 이벤트: {riskoff_days}일 ({riskoff_days/total_days*100:.1f}%)")
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. 이벤트 후 N일 수익률 계산
# ─────────────────────────────────────────────────────────────────────────────
def build_forward_returns(daily: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    리스크오프 이벤트 날짜 D에서 종목별 D+N일 수익률 계산.
    수익률 = adj_close(D+N) / adj_close(D) - 1  ← split 보정 가격 사용
    """
    # 종목별 날짜→adj_close 딕셔너리
    target_daily = daily[daily["symbol"].isin(TARGET_TICKERS)].copy()
    target_daily["date"] = pd.to_datetime(target_daily["date"])

    # 날짜 인덱스 정렬
    all_dates = sorted(target_daily["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 이벤트 날짜
    signal_df = signal_df.copy()
    signal_df.index = pd.to_datetime(signal_df.index)
    events = signal_df[signal_df["is_riskoff"]].copy()

    rows = []
    for event_date, ev_row in events.iterrows():
        if event_date not in date_to_idx:
            continue
        idx = date_to_idx[event_date]

        for ticker in TARGET_TICKERS:
            # adj_close 사용 — split 이벤트가 수익률을 왜곡하지 않도록
            t_data = target_daily[target_daily["symbol"] == ticker].set_index("date")["adj_close"]

            # 진입가: 이벤트 당일 adj_close
            if event_date not in t_data.index:
                continue
            entry_price = t_data[event_date]

            row = {
                "date": event_date,
                "ticker": ticker,
                "GLD_pct": ev_row["GLD_pct"],
                "SPY_pct": ev_row["SPY_pct"],
                "gld_bin": ev_row["gld_bin"],
                "spy_bin": ev_row["spy_bin"],
                "riskoff_streak": ev_row["riskoff_streak"],
                "entry_price": entry_price,
            }

            # N일 이후 수익률
            for n in FORWARD_DAYS:
                if idx + n < len(all_dates):
                    fwd_date = all_dates[idx + n]
                    if fwd_date in t_data.index:
                        fwd_price = t_data[fwd_date]
                        row[f"ret_{n}d"] = (fwd_price / entry_price - 1) * 100
                    else:
                        row[f"ret_{n}d"] = np.nan
                else:
                    row[f"ret_{n}d"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n[Step 3] 이벤트-종목 페어: {len(df):,}건")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. 3×3 매트릭스 분석
# ─────────────────────────────────────────────────────────────────────────────
def analyze_matrix(events_df: pd.DataFrame) -> pd.DataFrame:
    """GLD 강도 × SPY 강도 3×3 셀별 승률/수익률 집계."""
    sep("Part 1 — GLD × SPY 3×3 강도 매트릭스")

    ret_col = f"ret_{PRIMARY_FORWARD}d"
    results = []

    for gld_label in GLD_LABELS:
        for spy_label in SPY_LABELS:
            mask = (events_df["gld_bin"] == gld_label) & (events_df["spy_bin"] == spy_label)
            sub_df = events_df[mask][ret_col].dropna()

            if len(sub_df) == 0:
                continue

            win_rate = (sub_df > 0).mean() * 100
            avg_ret = sub_df.mean()
            median_ret = sub_df.median()
            n = len(sub_df)

            results.append({
                "GLD_bin": gld_label,
                "SPY_bin": spy_label,
                "n": n,
                "win_rate": round(win_rate, 1),
                "avg_ret": round(avg_ret, 2),
                "median_ret": round(median_ret, 2),
            })

    result_df = pd.DataFrame(results)

    # 콘솔 출력: 테이블 형태로
    print(f"  기준 수익률: {PRIMARY_FORWARD}일 후 수익률\n")
    pivot_win = result_df.pivot(index="GLD_bin", columns="SPY_bin", values="win_rate")
    pivot_avg = result_df.pivot(index="GLD_bin", columns="SPY_bin", values="avg_ret")
    pivot_n   = result_df.pivot(index="GLD_bin", columns="SPY_bin", values="n")

    print("[승률 % — GLD ↓ / SPY →]")
    print(pivot_win.to_string())
    print("\n[평균 수익률 % — GLD ↓ / SPY →]")
    print(pivot_avg.to_string())
    print("\n[샘플 수 — GLD ↓ / SPY →]")
    print(pivot_n.to_string())

    # 전체 리스크오프 기준선 (binary)
    sub("전체 리스크오프 기준선 (GLD↑+SPY↓ 전체)")
    base = events_df[ret_col].dropna()
    print(f"  n={len(base)}, 승률={( base > 0).mean()*100:.1f}%, "
          f"평균={base.mean():.2f}%, 중앙값={base.median():.2f}%")

    # 전체 비리스크오프 기준선 (비교용)
    sub(f"다기간 수익률 비교 (전체 리스크오프 이벤트)")
    for n in FORWARD_DAYS:
        col = f"ret_{n}d"
        s = events_df[col].dropna()
        if len(s):
            print(f"  {n}d후: n={len(s)}, 승률={( s>0).mean()*100:.1f}%, "
                  f"평균={s.mean():.2f}%, 중앙값={s.median():.2f}%")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5. 레벨 정의 검증
# ─────────────────────────────────────────────────────────────────────────────
def analyze_levels(events_df: pd.DataFrame) -> None:
    """
    v1에서 제안한 3단계 레벨 기준을 검증:
      Level 1: GLD 0~0.5% + SPY 0~-0.5%
      Level 2: GLD 0.5~1.0% + SPY -0.5~-1.5%
      Level 3: GLD 1.0%+   + SPY -1.5%+
    """
    sep("Part 2 — 3단계 레벨 성과 비교")

    ret_col = f"ret_{PRIMARY_FORWARD}d"

    level_defs = {
        "Level 1 (약한)":  ("0~0.5%",  "0~-0.5%"),
        "Level 2 (중간)":  ("0.5~1.0%", "-0.5~-1.5%"),
        "Level 3 (강한)":  ("1.0%+",   "-1.5%+"),
    }

    for level_name, (g_label, s_label) in level_defs.items():
        mask = (events_df["gld_bin"] == g_label) & (events_df["spy_bin"] == s_label)
        sub_df = events_df[mask][ret_col].dropna()

        if len(sub_df) == 0:
            print(f"  {level_name}: 샘플 없음")
            continue

        win = (sub_df > 0).mean() * 100
        avg = sub_df.mean()
        med = sub_df.median()
        n = len(sub_df)

        # t-test vs 0
        if n >= 5:
            t_stat, p_val = stats.ttest_1samp(sub_df.dropna(), 0)
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        else:
            p_val, sig = np.nan, ""

        print(f"  {level_name}: n={n:3d}, 승률={win:5.1f}%, "
              f"평균={avg:+6.2f}%, 중앙값={med:+6.2f}%  p={p_val:.3f}{sig}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6. 연속 리스크오프 효과
# ─────────────────────────────────────────────────────────────────────────────
def analyze_consecutive(events_df: pd.DataFrame) -> pd.DataFrame:
    """연속 리스크오프 발생일수에 따른 성과 차이."""
    sep("Part 3 — 연속 리스크오프 일수 효과")

    ret_col = f"ret_{PRIMARY_FORWARD}d"
    results = []

    for streak in [1, 2, 3]:
        label = f"{streak}일" if streak < 3 else "3일+"
        mask = (events_df["riskoff_streak"] == streak) if streak < 3 else (events_df["riskoff_streak"] >= 3)
        sub_df = events_df[mask][ret_col].dropna()

        if len(sub_df) == 0:
            print(f"  연속 {label}: 샘플 없음")
            continue

        win = (sub_df > 0).mean() * 100
        avg = sub_df.mean()
        n = len(sub_df)
        print(f"  연속 {label}: n={n:3d}, 승률={win:5.1f}%, 평균={avg:+6.2f}%")
        results.append({"streak": label, "n": n, "win_rate": round(win, 1), "avg_ret": round(avg, 2)})

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7. 종목별 성과 분해
# ─────────────────────────────────────────────────────────────────────────────
def analyze_by_ticker(events_df: pd.DataFrame) -> None:
    """종목별 리스크오프 이벤트 성과."""
    sep("Part 4 — 종목별 리스크오프 성과")

    ret_col = f"ret_{PRIMARY_FORWARD}d"

    for ticker in TARGET_TICKERS:
        sub_df = events_df[events_df["ticker"] == ticker][ret_col].dropna()
        if len(sub_df) == 0:
            continue
        win = (sub_df > 0).mean() * 100
        avg = sub_df.mean()
        med = sub_df.median()
        print(f"  {ticker:6s}: n={len(sub_df):3d}, 승률={win:5.1f}%, "
              f"평균={avg:+6.2f}%, 중앙값={med:+6.2f}%")

    # Level 3 × 종목별 (핵심 신호)
    sub("Level 3(강한 리스크오프) × 종목별")
    lvl3 = events_df[
        (events_df["gld_bin"] == "1.0%+") & (events_df["spy_bin"] == "-1.5%+")
    ]
    for ticker in TARGET_TICKERS:
        sub_df = lvl3[lvl3["ticker"] == ticker][ret_col].dropna()
        if len(sub_df) == 0:
            continue
        win = (sub_df > 0).mean() * 100
        avg = sub_df.mean()
        print(f"  {ticker:6s}: n={len(sub_df):3d}, 승률={win:5.1f}%, 평균={avg:+6.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    sep("Study A — GLD↑+SPY↓ 리스크오프 강도 그라데이션 분석")
    print(f"  데이터: {OHLCV_PATH}")
    print(f"  출력:   {OUT_DIR}")

    # Step 0+1. 데이터 로드 + 가격 보정
    daily, split_log = load_daily(OHLCV_PATH)

    # Step 2. 시황 신호 생성
    signal_df = build_signal_df(daily)

    # Step 3. 이벤트 후 수익률
    events_df = build_forward_returns(daily, signal_df)

    # Step 4~7. 분석
    matrix_df = analyze_matrix(events_df)
    analyze_levels(events_df)
    consec_df = analyze_consecutive(events_df)
    analyze_by_ticker(events_df)

    # 저장
    matrix_path = OUT_DIR / "study_a_riskoff_matrix.csv"
    consec_path = OUT_DIR / "study_a_riskoff_consecutive.csv"
    events_path = OUT_DIR / "study_a_riskoff_events.csv"
    split_path  = OUT_DIR / "study_a_split_log.csv"

    matrix_df.to_csv(matrix_path, index=False)
    consec_df.to_csv(consec_path, index=False)
    events_df.to_csv(events_path, index=False)
    if not split_log.empty:
        split_log.to_csv(split_path, index=False)

    sep("완료")
    print(f"  → {matrix_path.name}")
    print(f"  → {consec_path.name}")
    print(f"  → {events_path.name}")
    print(f"  → {split_path.name}")


if __name__ == "__main__":
    main()

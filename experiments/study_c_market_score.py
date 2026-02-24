"""
Study C — 시황 필터 통합 점수화 (market_score 0~1)
====================================================
v1 규칙서의 개별 시황 신호(R1·R3·R6·R13·R14·R16)를 하나의 연속 점수로 통합한다.
점수화된 market_score 구간별 forward return을 검증하여 최적 진입 임계값을 도출한다.

분석 항목
---------
C1. 개별 신호 점수화 — 각 변수를 0~1로 정규화하고 기여도 측정
C2. 가중 통합 점수 — 단순 합산 vs 로지스틱 회귀 가중치
C3. market_score 분위수별 성과 — 구간별 승률/수익률
C4. 최적 진입 임계값 — market_score ≥ X 조건의 정보비(IR) 최대화
C5. 시간 안정성 — 전반기/후반기 분할 검증

입력 변수 (Polymarket 없이 1분봉 기반)
--------------------------------------
1. gld_score    — GLD 하락 시 역발상 기회 (R1 기반)
2. spy_score    — SPY 하락 시 역발상 기회 (R6 기반)
3. riskoff_score— GLD↑+SPY↓ 강도 (R14 기반, Study A 결과 반영)
4. streak_score — SPY 스트릭 패널티 (R13 기반)
5. vol_score    — 변동성 기회 (R16 ATR Q4 기반)
6. btc_score    — BITU(비트코인 2x) 등락률 역발상 기회 (R3 대용)
7. momentum_score — 전일 매수 모멘텀 (연속 매수 성향)

출력
----
- data/results/analysis/study_c_market_score.csv  — 일별 market_score
- data/results/analysis/study_c_quantile_perf.csv — 분위수별 성과
- data/results/analysis/study_c_threshold.csv     — 임계값별 성과
- data/results/analysis/study_c_weights.csv       — 신호별 기여도/가중치
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # sigmoid

warnings.filterwarnings("ignore")

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_1MIN = ROOT / "data/market/ohlcv/backtest_1min_3y.parquet"
OUT_DIR = ROOT / "data/results/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TICKERS = ["MSTU", "CONL", "ROBN", "AMDL"]
MARKET_REFS = ["SPY", "GLD", "BITU", "QQQ"]
HOLD_DAYS = [1, 3, 5, 7]

# 신호별 초기 가중치 (v1 규칙서 근거 기반)
WEIGHTS_INIT = {
    "gld_score":      0.20,  # R1: p=0.036 ★★
    "spy_score":      0.15,  # R6: 매수율 +5.6%p
    "riskoff_score":  0.25,  # R14: 최강 신호
    "streak_score":   0.15,  # R13: SPY 3일+ 스트릭
    "vol_score":      0.15,  # R16: ATR Q4
    "btc_score":      0.10,  # BTC 방향 (R3 대용)
}


# ─── Step 0: 데이터 로드 ──────────────────────────────────────────────────────

STANDARD_SPLIT_RATIOS = [2.0, 4.0, 5.0, 10.0, 0.5, 0.25, 0.1, 0.2, 0.05]


def detect_splits(daily: pd.DataFrame) -> pd.DataFrame:
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
                for std in STANDARD_SPLIT_RATIOS:
                    if abs(g - std) / std <= 0.15:
                        rows.append({
                            "date": row["date"], "symbol": sym,
                            "gap_ratio": round(g, 4), "adj_factor": round(1.0 / g, 6),
                        })
                        break
    split_df = pd.DataFrame(rows)
    if not split_df.empty:
        split_df = split_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return split_df


def apply_split_adjustment(daily: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["adj_close"] = daily["close"].astype(float)
    for sym, splits in split_df.groupby("symbol"):
        mask = daily["symbol"] == sym
        sym_dates = daily.loc[mask, "date"].values
        splits_sorted = splits.sort_values("date", ascending=False)
        if len(splits_sorted) == 1:
            sp = splits_sorted.iloc[0]
            pre_split = sym_dates < sp["date"]
            daily.loc[mask & daily["date"].isin(sym_dates[pre_split]), "adj_close"] *= sp["adj_factor"]
        else:
            adj = daily.loc[mask].sort_values("date").copy()
            adj["adj_close"] = adj["close"].astype(float)
            split_dates = sorted(splits_sorted["date"].tolist())
            factors = {row["date"]: row["adj_factor"] for _, row in splits_sorted.iterrows()}
            cum = 1.0
            for sd in reversed(split_dates):
                cum *= factors[sd]
                adj.loc[adj["date"] < sd, "adj_close"] = adj.loc[adj["date"] < sd, "close"] * cum
            daily.loc[mask, "adj_close"] = adj["adj_close"].values
    return daily


def load_daily(path: Path) -> pd.DataFrame:
    print("[Step 0] 데이터 로드...")
    df = pd.read_parquet(path)
    all_syms = TARGET_TICKERS + MARKET_REFS
    df = df[df["symbol"].isin(all_syms)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    daily = (
        df.groupby(["symbol", "date"])
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )
    split_df = detect_splits(daily)
    if not split_df.empty:
        print(f"  → Split 보정: {len(split_df)}건")
    daily = apply_split_adjustment(daily, split_df)
    daily["pct"] = daily.groupby("symbol")["adj_close"].pct_change() * 100
    print(f"  → {len(daily)}행, {daily.date.min().date()}~{daily.date.max().date()}")
    return daily


# ─── Step 1: 신호 점수화 ──────────────────────────────────────────────────────

def minmax_norm(series: pd.Series, low: float, high: float) -> pd.Series:
    """[low, high] 범위를 0~1로 정규화. low → 1.0, high → 0.0 (역발상)."""
    clipped = series.clip(low, high)
    return 1.0 - (clipped - low) / (high - low)


def build_signal_scores(daily: pd.DataFrame) -> pd.DataFrame:
    """시황 신호 점수 계산 (모두 0~1, 높을수록 매수 적합)."""
    pivot_pct = daily.pivot(index="date", columns="symbol", values="pct")
    pivot_adj = daily.pivot(index="date", columns="symbol", values="adj_close")

    spy = pivot_pct.get("SPY", pd.Series(dtype=float))
    gld = pivot_pct.get("GLD", pd.Series(dtype=float))
    bitu = pivot_pct.get("BITU", pd.Series(dtype=float))

    scores = pd.DataFrame(index=pivot_pct.index)

    # ── gld_score: GLD 하락 = 높은 점수 (R1: GLD↑는 리스크오프) ──────────────
    # GLD < -0.5% → 1.0, GLD > +1.0% → 0.0
    scores["gld_score"] = minmax_norm(gld, low=-0.5, high=1.0).fillna(0.5)

    # ── spy_score: SPY 하락 = 높은 점수 (R6: 역발상) ────────────────────────
    # SPY < -1.0% → 1.0, SPY > +1.0% → 0.0
    scores["spy_score"] = minmax_norm(spy, low=-1.0, high=1.0).fillna(0.5)

    # ── riskoff_score: GLD↑+SPY↓ 강도 (Study A Level 2 기준) ────────────────
    # GLD↑ + SPY -0.5~-1.5% = 최고점, SPY < -1.5% = 위험 → 0점
    def calc_riskoff(row):
        g = gld.get(row.name, np.nan)
        s = spy.get(row.name, np.nan)
        if pd.isna(g) or pd.isna(s):
            return 0.5
        # SPY 극단 하락 = 위험 구간 → 0
        if s < -1.5:
            return 0.0
        # R14 발동 조건: GLD > 0, -1.5 ≤ SPY < 0
        if g > 0 and -1.5 <= s < 0:
            # 강도: GLD 0~1% 정규화 × SPY -0.5~-1.5% 정규화
            gld_strength = min(g / 1.0, 1.0)  # GLD 0~1% → 0~1
            spy_strength = max(0, min((-s - 0.5) / 1.0, 1.0))  # SPY -0.5~-1.5% → 0~1
            return 0.5 + 0.5 * (gld_strength * 0.5 + spy_strength * 0.5)
        return 0.5
    scores["riskoff_score"] = pd.Series({d: calc_riskoff(pd.Series(name=d)) for d in scores.index})

    # ── streak_score: SPY 스트릭 패널티 (R13) ───────────────────────────────
    # SPY 3일+ 연속 상승 → 0점, 3일+ 연속 하락 → 보통 (0.5), 기본 → 1점
    spy_up = (spy > 0).astype(int)
    streak_col = []
    cur_up, cur_dn = 0, 0
    for v in spy_up:
        if v == 1:
            cur_up += 1
            cur_dn = 0
        else:
            cur_dn += 1
            cur_up = 0
        streak_col.append((cur_up, cur_dn))
    spy_streak_up = pd.Series([s[0] for s in streak_col], index=spy.index)
    spy_streak_dn = pd.Series([s[1] for s in streak_col], index=spy.index)

    def calc_streak(date):
        up = spy_streak_up.get(date, 0)
        dn = spy_streak_dn.get(date, 0)
        if up >= 3:
            return 0.0  # 연속 상승 = 매수 금지
        if dn >= 3:
            return 0.4  # 연속 하락 = 주의
        return 1.0
    scores["streak_score"] = pd.Series({d: calc_streak(d) for d in scores.index})

    # ── vol_score: 5일 변동성 (R16 ATR 대용) ────────────────────────────────
    spy_vol5 = spy.rolling(5).std()
    vol_q75 = spy_vol5.quantile(0.75)
    vol_q25 = spy_vol5.quantile(0.25)
    scores["vol_score"] = minmax_norm(spy_vol5, low=vol_q25, high=vol_q75 * 2).fillna(0.5)
    # 고변동 = 역발상 기회 → 점수 높음 (min-max 그대로: 저변동=0, 고변동=1)
    scores["vol_score"] = 1.0 - scores["vol_score"]  # 반전: 고변동=1

    # ── btc_score: BITU 등락률 역발상 (R3 BTC 방향 대용) ────────────────────
    if bitu is not None and not bitu.empty:
        # BITU 하락 = BTC 약세 = 매수 기회 (역발상)
        scores["btc_score"] = minmax_norm(bitu, low=-5.0, high=5.0).fillna(0.5)
    else:
        scores["btc_score"] = 0.5

    # riskoff_streak 연속 신호 (Study A: 3일+ = 특급)
    riskoff_flag = (gld > 0) & (spy >= -1.5) & (spy < 0)
    streak_riskoff = []
    cur = 0
    for v in riskoff_flag.astype(int):
        cur = cur + 1 if v == 1 else 0
        streak_riskoff.append(cur)
    scores["riskoff_streak"] = streak_riskoff

    scores["spy_pct"] = spy
    scores["gld_pct"] = gld
    scores["bitu_pct"] = bitu if bitu is not None else np.nan

    return scores


# ─── Step 2: market_score 통합 ────────────────────────────────────────────────

def compute_market_score(scores: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """가중 합산으로 market_score(0~1) 계산."""
    signal_cols = list(weights.keys())
    score_mat = scores[signal_cols].copy()

    # 가중 합산
    w = np.array([weights[c] for c in signal_cols])
    w = w / w.sum()  # 정규화
    market_score = score_mat.multiply(w, axis=1).sum(axis=1)

    # streak 보정: 연속 3일+ 리스크오프 → 점수 상한 부스트
    streak_boost = np.where(scores["riskoff_streak"] >= 3, 0.1, 0.0)
    market_score = (market_score + streak_boost).clip(0, 1)

    scores["market_score"] = market_score
    return scores


# ─── Step 3: forward return 계산 ──────────────────────────────────────────────

def add_forward_returns(scores: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """market_score 기준일 이후 종목 평균 forward return."""
    adj_pivot = daily.pivot(index="date", columns="symbol", values="adj_close")
    dates = sorted(adj_pivot.index.tolist())
    date_idx = {d: i for i, d in enumerate(dates)}

    for hd in HOLD_DAYS:
        scores[f"avg_ret_{hd}d"] = np.nan

    for date in scores.index:
        idx = date_idx.get(date)
        if idx is None:
            continue
        for hd in HOLD_DAYS:
            target_idx = idx + hd
            if target_idx >= len(dates):
                continue
            rets = []
            for t in TARGET_TICKERS:
                if t not in adj_pivot.columns:
                    continue
                entry = adj_pivot.at[date, t] if date in adj_pivot.index else np.nan
                exit_d = dates[target_idx]
                exit_p = adj_pivot.at[exit_d, t] if exit_d in adj_pivot.index else np.nan
                if not pd.isna(entry) and not pd.isna(exit_p) and entry > 0:
                    rets.append((exit_p / entry - 1) * 100)
            if rets:
                scores.at[date, f"avg_ret_{hd}d"] = np.mean(rets)

    return scores


# ─── Step 4: C3 분위수별 성과 ─────────────────────────────────────────────────

def analyze_quantile_perf(scores: pd.DataFrame) -> pd.DataFrame:
    """market_score 5분위수별 성과."""
    scores = scores.copy()
    scores["quantile"] = pd.qcut(scores["market_score"], q=5, labels=["Q1(최저)", "Q2", "Q3", "Q4", "Q5(최고)"])

    rows = []
    for q, grp in scores.groupby("quantile", observed=True):
        n = len(grp)
        score_range = f"{grp.market_score.min():.3f}~{grp.market_score.max():.3f}"
        ret5 = grp["avg_ret_5d"].dropna()
        if len(ret5) < 5:
            continue
        t_stat, p_val = stats.ttest_1samp(ret5, 0)
        rows.append({
            "quantile": q,
            "score_range": score_range,
            "n": n,
            "win_rate": round((ret5 > 0).mean() * 100, 1),
            "avg_ret_5d": round(ret5.mean(), 2),
            "median_ret_5d": round(ret5.median(), 2),
            "p_value": round(p_val, 4),
        })
    return pd.DataFrame(rows)


# ─── Step 5: C4 임계값별 성과 ─────────────────────────────────────────────────

def analyze_threshold(scores: pd.DataFrame) -> pd.DataFrame:
    """market_score ≥ X 조건의 진입 성과. 정보비(IR) = avg_ret / std_ret."""
    thresholds = np.arange(0.3, 0.85, 0.05)
    rows = []
    for thr in thresholds:
        grp = scores[scores["market_score"] >= thr]
        n = len(grp)
        if n < 10:
            break
        ret5 = grp["avg_ret_5d"].dropna()
        std5 = ret5.std()
        ir = ret5.mean() / std5 if std5 > 0 else 0
        t_stat, p_val = stats.ttest_1samp(ret5, 0) if len(ret5) > 5 else (0, 1)
        rows.append({
            "threshold": round(thr, 2),
            "n": n,
            "coverage_pct": round(n / len(scores) * 100, 1),
            "win_rate": round((ret5 > 0).mean() * 100, 1),
            "avg_ret_5d": round(ret5.mean(), 2),
            "std_ret_5d": round(std5, 2),
            "IR": round(ir, 3),
            "p_value": round(p_val, 4),
        })
    return pd.DataFrame(rows)


# ─── Step 6: C5 시간 안정성 (전반/후반 OOS) ──────────────────────────────────

def analyze_stability(scores: pd.DataFrame) -> pd.DataFrame:
    """전반기 / 후반기 분할 market_score 성과 비교."""
    dates_sorted = sorted(scores.index)
    mid = dates_sorted[len(dates_sorted) // 2]

    rows = []
    for label, subset in [("전반기", scores[scores.index < mid]), ("후반기", scores[scores.index >= mid])]:
        # market_score 상위 30%
        thr = subset["market_score"].quantile(0.7)
        high_score = subset[subset["market_score"] >= thr]
        ret5 = high_score["avg_ret_5d"].dropna()
        if len(ret5) < 5:
            continue
        rows.append({
            "기간": label,
            "기간_범위": f"{subset.index.min().date()}~{subset.index.max().date()}",
            "상위30% 기준점수": round(thr, 3),
            "n": len(high_score),
            "win_rate": round((ret5 > 0).mean() * 100, 1),
            "avg_ret_5d": round(ret5.mean(), 2),
        })
    return pd.DataFrame(rows)


# ─── Step 7: 신호별 기여도 분석 ──────────────────────────────────────────────

def analyze_signal_contribution(scores: pd.DataFrame) -> pd.DataFrame:
    """각 신호 점수와 forward return의 스피어만 상관계수."""
    signal_cols = list(WEIGHTS_INIT.keys())
    rows = []
    for col in signal_cols:
        valid = scores[[col, "avg_ret_5d"]].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid[col], valid["avg_ret_5d"])
        rows.append({
            "signal": col,
            "weight_init": WEIGHTS_INIT[col],
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 4),
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    daily = load_daily(DATA_1MIN)

    # 신호 점수화
    print("\n[Step 1] 시황 신호 점수화...")
    scores = build_signal_scores(daily)
    print(f"  → {len(scores)}일, 신호 컬럼: {[c for c in scores.columns if 'score' in c and c != 'riskoff_streak']}")

    # market_score 통합
    print("[Step 2] market_score 통합...")
    scores = compute_market_score(scores, WEIGHTS_INIT)
    print(f"  → market_score 범위: {scores.market_score.min():.3f} ~ {scores.market_score.max():.3f}")
    print(f"  → 평균: {scores.market_score.mean():.3f}, 중앙값: {scores.market_score.median():.3f}")

    # forward return 계산
    print("[Step 3] Forward return 계산...")
    scores = add_forward_returns(scores, daily)
    valid_ret = scores["avg_ret_5d"].dropna()
    print(f"  → 유효 데이터: {len(valid_ret)}일, 전체 평균: {valid_ret.mean():.2f}%")

    # C3: 분위수별 성과
    print("\n[Step 4] C3: market_score 분위수별 성과:")
    quantile_df = analyze_quantile_perf(scores)
    print(quantile_df.to_string(index=False))

    # C4: 임계값별 성과
    print("\n[Step 5] C4: 진입 임계값별 성과:")
    threshold_df = analyze_threshold(scores)
    print(threshold_df.to_string(index=False))

    # C5: 전반/후반 안정성
    print("\n[Step 6] C5: 전반기/후반기 시간 안정성:")
    stability_df = analyze_stability(scores)
    print(stability_df.to_string(index=False))

    # 신호별 기여도
    print("\n[Step 7] 신호별 기여도 (스피어만 상관):")
    contrib_df = analyze_signal_contribution(scores)
    print(contrib_df.to_string(index=False))

    # 저장
    scores.to_csv(OUT_DIR / "study_c_market_score.csv")
    quantile_df.to_csv(OUT_DIR / "study_c_quantile_perf.csv", index=False)
    threshold_df.to_csv(OUT_DIR / "study_c_threshold.csv", index=False)
    contrib_df.to_csv(OUT_DIR / "study_c_weights.csv", index=False)

    print(f"\n✓ 결과 저장: {OUT_DIR}")
    print("  - study_c_market_score.csv")
    print("  - study_c_quantile_perf.csv")
    print("  - study_c_threshold.csv")
    print("  - study_c_weights.csv")

    # 최적 임계값 요약
    if not threshold_df.empty:
        best = threshold_df.loc[threshold_df["IR"].idxmax()]
        print(f"\n★ 최적 임계값: market_score ≥ {best['threshold']}")
        print(f"  n={int(best['n'])}, 커버리지={best['coverage_pct']}%, 승률={best['win_rate']}%, 5d평균={best['avg_ret_5d']}%, IR={best['IR']}")


if __name__ == "__main__":
    main()

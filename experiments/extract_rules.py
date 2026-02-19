#!/usr/bin/env python3
"""
규칙 추출 — 규칙 복원 실험 Step 2
==================================
Decision Log에서 매수/관망을 구분하는 조건을 통계적으로 찾고,
Decision Tree로 자동 규칙을 추출한다.

입력:
  data/results/analysis/decision_log.csv
  data/results/analysis/decision_log_trades.csv

출력:
  콘솔 리포트 + data/results/analysis/rule_extraction_report.csv

사용법:
  pyenv shell ptj_stock_lab && python experiments/extract_rules.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE / "data" / "results" / "analysis"

# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(ANALYSIS_DIR / "decision_log.csv")
    trades = pd.read_csv(ANALYSIS_DIR / "decision_log_trades.csv")
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    trades["date"] = pd.to_datetime(trades["date"]).dt.date
    return daily, trades


# ════════════════════════════════════════════════════════════════
# 2. 통계 기반 구분력 분석
# ════════════════════════════════════════════════════════════════
def statistical_discrimination(daily: pd.DataFrame):
    """매수일 vs 관망일 — 각 feature의 유의성 검정."""
    print()
    print("=" * 70)
    print("  [Step 2a] 통계 기반 구분력 분석 (매수일 vs 관망일)")
    print("=" * 70)

    buy_mask = daily["action"].isin(["BUY", "BUY+SELL"])
    hold_mask = daily["action"] == "HOLD"
    buy_days = daily[buy_mask]
    hold_days = daily[hold_mask]

    features = [
        "SPY_pct", "QQQ_pct", "GLD_pct",
        "poly_btc_up", "poly_ndx_up", "poly_eth_up",
        "poly_btc_up_delta",
        "SPY_vol_5d", "buy_count_3d",
        "fear_signal", "confidence_signal",
        "SPY_3d_streak", "QQQ_3d_streak",
    ]

    results = []
    print(f"\n  {'Feature':<22s} {'매수일':>10s} {'관망일':>10s} {'차이':>10s} {'p-value':>10s} {'유의':>5s}")
    print("  " + "-" * 70)

    for feat in features:
        if feat not in daily.columns:
            continue

        buy_vals = buy_days[feat].dropna()
        hold_vals = hold_days[feat].dropna()

        if len(buy_vals) < 5 or len(hold_vals) < 5:
            continue

        buy_mean = buy_vals.mean()
        hold_mean = hold_vals.mean()
        diff = buy_mean - hold_mean

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(buy_vals, hold_vals, equal_var=False)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

        results.append({
            "feature": feat,
            "buy_mean": round(buy_mean, 4),
            "hold_mean": round(hold_mean, 4),
            "diff": round(diff, 4),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "significant": sig,
        })

        fmt = ".4f" if "poly" in feat else ".3f"
        print(f"  {feat:<22s} {buy_mean:>10{fmt}} {hold_mean:>10{fmt}} {diff:>+10{fmt}} {p_val:>10.4f} {sig:>5s}")

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
# 3. 조건부 분석 — 구간별 매수 확률
# ════════════════════════════════════════════════════════════════
def conditional_analysis(daily: pd.DataFrame):
    """주요 feature를 구간으로 나누어 매수 확률 분석."""
    print()
    print("=" * 70)
    print("  [Step 2b] 조건부 분석 — 구간별 매수 확률")
    print("=" * 70)

    daily["is_buy"] = daily["action"].isin(["BUY", "BUY+SELL"]).astype(int)
    base_rate = daily["is_buy"].mean()
    print(f"\n  기저 매수 확률: {base_rate*100:.1f}% ({daily['is_buy'].sum()}/{len(daily)}일)")

    analyses = [
        ("GLD_pct", [(-999, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 999)]),
        ("SPY_pct", [(-999, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 999)]),
        ("QQQ_pct", [(-999, -1.0), (-1.0, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 999)]),
        ("poly_btc_up", [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]),
        ("poly_ndx_up", [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]),
    ]

    for feat, bins in analyses:
        if feat not in daily.columns:
            continue

        print(f"\n  ── {feat} ──")
        print(f"  {'구간':<20s} {'일수':>5s} {'매수일':>6s} {'매수확률':>8s} {'vs 기저':>8s}")
        print("  " + "-" * 50)

        for lo, hi in bins:
            mask = (daily[feat] >= lo) & (daily[feat] < hi)
            subset = daily[mask]
            if len(subset) < 3:
                continue
            buy_rate = subset["is_buy"].mean()
            label = f"[{lo:+.1f}, {hi:+.1f})" if abs(lo) < 100 else f"< {hi:+.1f}" if lo < -100 else f">= {lo:+.1f}"
            diff = buy_rate - base_rate
            marker = " <<<" if abs(diff) > 0.15 else ""
            print(f"  {label:<20s} {len(subset):>5d} {subset['is_buy'].sum():>6d} "
                  f"{buy_rate*100:>7.1f}% {diff*100:>+7.1f}%{marker}")


# ════════════════════════════════════════════════════════════════
# 4. Decision Tree 규칙 추출
# ════════════════════════════════════════════════════════════════
def decision_tree_rules(daily: pd.DataFrame):
    """Decision Tree로 자동 규칙 추출."""
    print()
    print("=" * 70)
    print("  [Step 2c] Decision Tree 자동 규칙 추출")
    print("=" * 70)

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("\n  [SKIP] scikit-learn 미설치. pip install scikit-learn 후 재실행.")
        return None

    daily["is_buy"] = daily["action"].isin(["BUY", "BUY+SELL"]).astype(int)

    features = [
        "SPY_pct", "QQQ_pct", "GLD_pct",
        "poly_btc_up", "poly_ndx_up",
        "SPY_vol_5d", "buy_count_3d",
        "fear_signal", "confidence_signal",
        "SPY_3d_streak", "QQQ_3d_streak",
        "poly_btc_up_delta",
    ]

    available = [f for f in features if f in daily.columns]
    X = daily[available].copy()
    y = daily["is_buy"].copy()

    # NaN 행 제거
    valid = X.dropna().index
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 30:
        print("  데이터 부족")
        return None

    print(f"\n  학습 데이터: {len(X)}일 (매수 {y.sum()}일, 관망 {(1-y).sum():.0f}일)")
    print(f"  Features: {available}")

    # 깊이별 성능 비교
    print(f"\n  [깊이별 교차검증 정확도]")
    print(f"  {'depth':>5s} {'accuracy':>10s} {'recall(BUY)':>12s}")
    print("  " + "-" * 30)

    best_depth = 3
    best_score = 0
    for depth in [2, 3, 4, 5]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight="balanced")
        scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
        tree.fit(X, y)
        preds = tree.predict(X)
        recall = (preds[y == 1] == 1).mean()
        print(f"  {depth:>5d} {scores.mean():>10.3f} {recall:>12.3f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_depth = depth

    # 최적 깊이로 학습
    tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")
    tree.fit(X, y)

    print(f"\n  최적 깊이: {best_depth} (정확도: {best_score:.3f})")
    print(f"\n  [추출된 규칙 트리]")
    print("-" * 70)
    print(export_text(tree, feature_names=available, max_depth=best_depth))

    # Feature importance
    print(f"\n  [Feature 중요도]")
    print(f"  {'Feature':<22s} {'Importance':>12s}")
    print("  " + "-" * 36)
    importances = sorted(
        zip(available, tree.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for feat, imp in importances:
        if imp > 0.01:
            bar = "#" * int(imp * 40)
            print(f"  {feat:<22s} {imp:>12.4f}  {bar}")

    return tree


# ════════════════════════════════════════════════════════════════
# 5. 매도 패턴 분석
# ════════════════════════════════════════════════════════════════
def sell_pattern_analysis(daily: pd.DataFrame, trades: pd.DataFrame):
    """매도 시점의 시장 상태 분석."""
    print()
    print("=" * 70)
    print("  [Step 2d] 매도 패턴 분석")
    print("=" * 70)

    sell_trades = trades[trades["action"] == "판매"]
    if len(sell_trades) == 0:
        print("  매도 거래 없음")
        return

    print(f"\n  총 매도: {len(sell_trades)}건, {sell_trades['date'].nunique()}일")

    # 매도 시점 시장 상태
    print(f"\n  [매도 시점 시장 상태]")
    for col in ["SPY_pct", "QQQ_pct", "GLD_pct", "poly_btc_up", "ticker_pct"]:
        if col not in sell_trades.columns:
            continue
        vals = sell_trades[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {col:<16s}: 평균 {vals.mean():+.3f}  중앙값 {vals.median():+.3f}")

    # 종목별 매도 이유 추정 (매도 당일 종목 등락)
    print(f"\n  [매도 시점 종목 등락 분포]")
    ticker_pcts = sell_trades["ticker_pct"].dropna()
    if len(ticker_pcts) > 0:
        print(f"  상승 중 매도(이익실현): {(ticker_pcts > 0).sum()}건 ({(ticker_pcts > 0).mean()*100:.1f}%)")
        print(f"  하락 중 매도(손절):     {(ticker_pcts < 0).sum()}건 ({(ticker_pcts < 0).mean()*100:.1f}%)")
        print(f"  평균 등락: {ticker_pcts.mean():+.3f}%")


# ════════════════════════════════════════════════════════════════
# 6. DCA 패턴 분석
# ════════════════════════════════════════════════════════════════
def dca_pattern_analysis(trades: pd.DataFrame):
    """DCA(분할매수) 패턴 분석."""
    print()
    print("=" * 70)
    print("  [Step 2e] DCA 패턴 분석")
    print("=" * 70)

    buy_trades = trades[trades["action"] == "구매"].copy()
    buy_trades = buy_trades.sort_values(["yf_ticker", "date"])

    # 같은 날 같은 종목 반복 매수 횟수
    daily_dca = buy_trades.groupby(["date", "yf_ticker"]).size().reset_index(name="count")

    print(f"\n  [같은 날 같은 종목 매수 횟수 분포]")
    count_dist = daily_dca["count"].value_counts().sort_index()
    for cnt, freq in count_dist.items():
        print(f"  {cnt}회: {freq}건")

    # 연속일 매수 패턴 (같은 종목)
    print(f"\n  [종목별 연속 매수일 패턴 (상위 5)]")
    print(f"  {'종목':<8s} {'최장연속':>8s} {'평균연속':>8s} {'총매수일':>8s}")
    print("  " + "-" * 36)

    for ticker in buy_trades["yf_ticker"].dropna().value_counts().head(8).index:
        t_dates = sorted(buy_trades[buy_trades["yf_ticker"] == ticker]["date"].unique())
        if len(t_dates) < 2:
            continue

        streaks = []
        current_streak = 1
        for i in range(1, len(t_dates)):
            diff = (pd.Timestamp(t_dates[i]) - pd.Timestamp(t_dates[i - 1])).days
            if diff <= 3:  # 주말 고려 (1~3일 이내면 연속)
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)

        print(f"  {ticker:<8s} {max(streaks):>8d} {np.mean(streaks):>8.1f} {len(t_dates):>8d}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  규칙 추출 — 규칙 복원 실험 Step 2          ║")
    print("  ╚══════════════════════════════════════════════╝")

    daily, trades = load_data()
    print(f"  일별 로그: {len(daily)}일, 거래 로그: {len(trades)}건")

    # 2a. 통계 분석
    stat_df = statistical_discrimination(daily)

    # 2b. 조건부 분석
    conditional_analysis(daily)

    # 2c. Decision Tree
    tree = decision_tree_rules(daily)

    # 2d. 매도 패턴
    sell_pattern_analysis(daily, trades)

    # 2e. DCA 패턴
    dca_pattern_analysis(trades)

    # 저장
    if stat_df is not None:
        stat_df.to_csv(
            ANALYSIS_DIR / "rule_extraction_report.csv",
            index=False, encoding="utf-8-sig",
        )
        print(f"\n  저장: {ANALYSIS_DIR / 'rule_extraction_report.csv'}")

    print("\n  완료!")
    print()


if __name__ == "__main__":
    main()

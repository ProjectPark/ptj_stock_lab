#!/usr/bin/env python3
"""
규칙 추출 v2 — 심화 분석
========================
Step 1 (build_decision_log)의 결과에 다음을 추가:
  - 쌍둥이 갭(lead-follow 스프레드) 변수
  - 금액 가중 분석 (소액 DCA vs 대형 배팅)
  - Polymarket은 btc_up 중심 (커버리지 92%)
  - Walk-forward 검증 (전반기 학습 → 후반기 테스트)
  - 종합 규칙 후보 리스트 출력

사용법:
  pyenv shell ptj_stock_lab && python experiments/extract_rules_v2.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE / "data" / "results" / "analysis"
MARKET_DAILY = BASE / "data" / "market" / "daily" / "market_daily.parquet"
META_DIR = BASE / "data" / "meta"

# 쌍둥이 페어 정의 (lead → follow)
TWIN_PAIRS = {
    "coin": {"lead": "BITU", "follow": ["MSTU", "CONL"]},
    "bank": {"lead": "ROBN", "follow": ["CONL"]},
    "semi": {"lead": "NVDL", "follow": ["AMDL"]},
}


# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드 + 쌍둥이 갭 추가
# ════════════════════════════════════════════════════════════════
def load_and_enrich() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision Log + 쌍둥이 갭 + 금액 가중 변수 추가."""
    daily = pd.read_csv(ANALYSIS_DIR / "decision_log.csv")
    trades = pd.read_csv(ANALYSIS_DIR / "decision_log_trades.csv")
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    trades["date"] = pd.to_datetime(trades["date"]).dt.date

    # 시장 데이터에서 쌍둥이 갭 계산
    market = _load_market_pcts()

    # 일별 로그에 갭 추가
    gap_rows = []
    for dt in daily["date"]:
        row = {"date": dt}
        day_pcts = market.get(dt, {})

        # 쌍둥이 갭: lead - follow
        for pair_name, pair in TWIN_PAIRS.items():
            lead_pct = day_pcts.get(pair["lead"], np.nan)
            for follow in pair["follow"]:
                follow_pct = day_pcts.get(follow, np.nan)
                if not np.isnan(lead_pct) and not np.isnan(follow_pct):
                    gap = lead_pct - follow_pct
                else:
                    gap = np.nan
                row[f"gap_{pair_name}_{follow}"] = round(gap, 4) if not np.isnan(gap) else np.nan

        # 주요 종목 등락
        for sym in ["BITU", "MSTU", "CONL", "ROBN", "NVDL", "AMDL", "SOXL"]:
            row[f"{sym}_pct"] = day_pcts.get(sym, np.nan)

        gap_rows.append(row)

    gap_df = pd.DataFrame(gap_rows)
    daily = daily.merge(gap_df, on="date", how="left")

    # 일별 금액 가중 변수
    daily["is_buy"] = daily["action"].isin(["BUY", "BUY+SELL"]).astype(int)
    daily["is_heavy_buy"] = (daily["buy_amount_usd"] > 500).astype(int)

    # 갭 절대값 (방향 무관 변동성)
    for col in [c for c in daily.columns if c.startswith("gap_")]:
        daily[f"{col}_abs"] = daily[col].abs()

    # 거래 로그에도 갭 추가
    trades = trades.merge(
        gap_df, on="date", how="left", suffixes=("", "_gap")
    )

    return daily, trades


def _load_market_pcts() -> dict[object, dict[str, float]]:
    """market_daily.parquet → {date: {symbol: pct_change}}."""
    df = pd.read_parquet(MARKET_DAILY)
    df_long = df.stack(level=0, future_stack=True).reset_index()

    col_map = {}
    for c in df_long.columns:
        cl = c.lower()
        if cl == "date":
            col_map[c] = "date"
        elif cl in ("ticker", "symbol", "level_1"):
            col_map[c] = "symbol"
        elif cl == "close":
            col_map[c] = "close"
        else:
            col_map[c] = c
    df_long = df_long.rename(columns=col_map)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    df_long = df_long.sort_values(["symbol", "date"])
    df_long["prev_close"] = df_long.groupby("symbol")["close"].shift(1)
    df_long["pct"] = ((df_long["close"] / df_long["prev_close"] - 1) * 100).round(4)

    result: dict[object, dict[str, float]] = {}
    for _, row in df_long.iterrows():
        dt = row["date"]
        sym = row["symbol"]
        pct = row["pct"]
        if dt not in result:
            result[dt] = {}
        if pd.notna(pct):
            result[dt][sym] = pct

    return result


# ════════════════════════════════════════════════════════════════
# 2. 쌍둥이 갭 분석
# ════════════════════════════════════════════════════════════════
def twin_gap_analysis(daily: pd.DataFrame, trades: pd.DataFrame):
    """쌍둥이 갭과 매수 행동의 관계."""
    print()
    print("=" * 70)
    print("  [분석 1] 쌍둥이 갭과 매수 행동")
    print("=" * 70)

    gap_cols = [c for c in daily.columns if c.startswith("gap_") and not c.endswith("_abs")]

    for col in gap_cols:
        valid = daily[col].dropna()
        if len(valid) < 20:
            continue

        print(f"\n  ── {col} ──")
        # 구간별 매수 확률
        bins = [(-999, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 999)]
        base_rate = daily["is_buy"].mean()

        print(f"  {'구간':<16s} {'일수':>5s} {'매수일':>6s} {'매수율':>7s} {'vs기저':>7s} {'평균금액':>10s}")
        print("  " + "-" * 55)

        for lo, hi in bins:
            mask = (daily[col] >= lo) & (daily[col] < hi) & daily[col].notna()
            sub = daily[mask]
            if len(sub) < 3:
                continue
            rate = sub["is_buy"].mean()
            avg_amt = sub["buy_amount_usd"].mean()
            diff = rate - base_rate
            label = f"[{lo:+.0f}, {hi:+.0f}%)" if abs(lo) < 100 else (f"< {hi:+.0f}%" if lo < -100 else f">= {lo:+.0f}%")
            marker = " <<<" if abs(diff) > 0.12 else ""
            print(f"  {label:<16s} {len(sub):>5d} {sub['is_buy'].sum():>6d} "
                  f"{rate*100:>6.1f}% {diff*100:>+6.1f}% ${avg_amt:>9,.0f}{marker}")

    # 매수 거래의 갭 분포
    buy_trades = trades[trades["action"] == "구매"]
    print(f"\n  ── 매수 거래 시점의 갭 분포 ──")
    for col in gap_cols:
        vals = buy_trades[col].dropna()
        if len(vals) < 10:
            continue
        print(f"  {col}: 평균 {vals.mean():+.3f}%  중앙값 {vals.median():+.3f}%  "
              f"양갭매수 {(vals > 0).mean()*100:.0f}%  음갭매수 {(vals < 0).mean()*100:.0f}%")


# ════════════════════════════════════════════════════════════════
# 3. 금액 가중 분석
# ════════════════════════════════════════════════════════════════
def amount_weighted_analysis(daily: pd.DataFrame, trades: pd.DataFrame):
    """소액 DCA vs 대형 배팅 구분 분석."""
    print()
    print("=" * 70)
    print("  [분석 2] 금액 규모별 행동 패턴")
    print("=" * 70)

    buy_trades = trades[trades["action"] == "구매"].copy()
    buy_trades["size_class"] = pd.cut(
        buy_trades["amount_usd"],
        bins=[0, 20, 100, 500, 2000, 999999],
        labels=["미소액(<$20)", "소액($20-100)", "중형($100-500)", "대형($500-2K)", "초대형(>$2K)"],
    )

    print(f"\n  [매수 건별 금액 분포]")
    print(f"  {'규모':<18s} {'건수':>6s} {'비율':>6s} {'총금액':>12s} {'평균단가':>10s}")
    print("  " + "-" * 55)
    for cls in buy_trades["size_class"].cat.categories:
        sub = buy_trades[buy_trades["size_class"] == cls]
        if len(sub) == 0:
            continue
        pct = len(sub) / len(buy_trades) * 100
        total = sub["amount_usd"].sum()
        avg_price = sub["price_usd"].mean()
        print(f"  {cls:<18s} {len(sub):>6d} {pct:>5.1f}% ${total:>11,.0f} ${avg_price:>9.2f}")

    # 대형 매수(>$500) 시점의 시장 상태
    heavy = buy_trades[buy_trades["amount_usd"] >= 500]
    light = buy_trades[buy_trades["amount_usd"] < 100]

    if len(heavy) > 5 and len(light) > 5:
        print(f"\n  [대형 매수 vs 소액 매수 — 시장 상태 비교]")
        print(f"  {'지표':<16s} {'대형(>$500)':>12s} {'소액(<$100)':>12s} {'차이':>10s}")
        print("  " + "-" * 52)
        for col in ["SPY_pct", "QQQ_pct", "GLD_pct", "poly_btc_up", "ticker_pct"]:
            if col not in heavy.columns:
                continue
            h_mean = heavy[col].mean()
            l_mean = light[col].mean()
            if pd.isna(h_mean) or pd.isna(l_mean):
                continue
            diff = h_mean - l_mean
            print(f"  {col:<16s} {h_mean:>+12.3f} {l_mean:>+12.3f} {diff:>+10.3f}")

    # 종목별 평균 매수 금액
    print(f"\n  [종목별 평균 1회 매수 금액 (상위 10)]")
    ticker_amt = buy_trades.groupby("yf_ticker")["amount_usd"].agg(["mean", "sum", "count"])
    ticker_amt = ticker_amt.sort_values("sum", ascending=False).head(10)
    print(f"  {'종목':<8s} {'평균/건':>10s} {'총액':>12s} {'건수':>6s}")
    print("  " + "-" * 40)
    for ticker, row in ticker_amt.iterrows():
        print(f"  {ticker:<8s} ${row['mean']:>9,.0f} ${row['sum']:>11,.0f} {int(row['count']):>6d}")


# ════════════════════════════════════════════════════════════════
# 4. 심화 Decision Tree (갭 + BTC 중심)
# ════════════════════════════════════════════════════════════════
def enhanced_decision_tree(daily: pd.DataFrame):
    """갭 변수 + btc_up 중심의 Decision Tree."""
    print()
    print("=" * 70)
    print("  [분석 3] 심화 Decision Tree (갭 + Polymarket)")
    print("=" * 70)

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report
    except ImportError:
        print("\n  [SKIP] scikit-learn 미설치")
        return

    features = [
        "SPY_pct", "QQQ_pct", "GLD_pct",
        "poly_btc_up",
        "fear_signal",
        "SPY_3d_streak", "QQQ_3d_streak",
        "SPY_vol_5d",
    ]

    # 갭 변수 추가
    gap_cols = [c for c in daily.columns if c.startswith("gap_") and not c.endswith("_abs")]
    features += gap_cols

    # 종목 등락 추가
    ticker_cols = [c for c in daily.columns if c.endswith("_pct") and c.split("_")[0] in
                   ("BITU", "MSTU", "CONL", "ROBN", "NVDL", "AMDL")]
    features += ticker_cols

    available = [f for f in features if f in daily.columns]

    X = daily[available].copy()
    y = daily["is_buy"].copy()

    # NaN 처리: 갭이나 poly가 없는 날은 0으로 (중립)
    X = X.fillna(0)

    valid = X.index
    X = X.loc[valid]
    y = y.loc[valid]

    print(f"\n  학습 데이터: {len(X)}일 (매수 {y.sum()}일, 관망 {(1-y).sum():.0f}일)")
    print(f"  Features ({len(available)}): {available}")

    # ── Full dataset tree ──
    print(f"\n  [전체 데이터 규칙 트리 (depth=4)]")
    tree = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    tree.fit(X, y)

    scores = cross_val_score(tree, X, y, cv=5, scoring="accuracy")
    print(f"  5-fold 정확도: {scores.mean():.3f} (±{scores.std():.3f})")
    print()
    print(export_text(tree, feature_names=available, max_depth=4))

    # Feature importance
    print(f"\n  [Feature 중요도 (상위)]")
    importances = sorted(zip(available, tree.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        if imp > 0.02:
            bar = "#" * int(imp * 40)
            print(f"  {feat:<28s} {imp:.4f}  {bar}")

    return tree, available


# ════════════════════════════════════════════════════════════════
# 5. Walk-Forward 검증
# ════════════════════════════════════════════════════════════════
def walk_forward_validation(daily: pd.DataFrame):
    """전반기 학습 → 후반기 테스트."""
    print()
    print("=" * 70)
    print("  [분석 4] Walk-Forward 검증")
    print("=" * 70)

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError:
        print("\n  [SKIP] scikit-learn 미설치")
        return

    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    mid = len(daily_sorted) // 2
    train = daily_sorted.iloc[:mid]
    test = daily_sorted.iloc[mid:]

    print(f"\n  Train: {len(train)}일 ({train['date'].iloc[0]} ~ {train['date'].iloc[-1]})")
    print(f"  Test:  {len(test)}일 ({test['date'].iloc[0]} ~ {test['date'].iloc[-1]})")

    features = [
        "SPY_pct", "QQQ_pct", "GLD_pct", "poly_btc_up",
        "fear_signal", "SPY_vol_5d",
    ]
    gap_cols = [c for c in daily.columns if c.startswith("gap_") and not c.endswith("_abs")]
    features += gap_cols
    ticker_cols = [c for c in daily.columns if c.endswith("_pct") and c.split("_")[0] in
                   ("BITU", "MSTU", "CONL", "ROBN")]
    features += ticker_cols

    available = [f for f in features if f in daily.columns]

    X_train = train[available].fillna(0)
    y_train = train["is_buy"]
    X_test = test[available].fillna(0)
    y_test = test["is_buy"]

    print(f"\n  Train 매수율: {y_train.mean()*100:.1f}%")
    print(f"  Test  매수율: {y_test.mean()*100:.1f}%")

    for depth in [2, 3, 4]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight="balanced")
        tree.fit(X_train, y_train)
        preds = tree.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\n  depth={depth}: 정확도={acc:.3f}  정밀도={prec:.3f}  재현율={rec:.3f}  F1={f1:.3f}")

        if depth == 3:
            print(f"\n  [depth=3 규칙 — Out-of-Sample]")
            print(export_text(tree, feature_names=available, max_depth=3))

            # 중요도
            importances = sorted(zip(available, tree.feature_importances_), key=lambda x: x[1], reverse=True)
            print(f"  [OOS Feature 중요도]")
            for feat, imp in importances:
                if imp > 0.02:
                    print(f"  {feat:<28s} {imp:.4f}")


# ════════════════════════════════════════════════════════════════
# 6. 종합 규칙 후보 정리
# ════════════════════════════════════════════════════════════════
def summarize_rule_candidates(daily: pd.DataFrame, trades: pd.DataFrame):
    """발견된 패턴을 규칙 후보로 정리."""
    print()
    print("=" * 70)
    print("  [종합] 규칙 후보 리스트")
    print("=" * 70)

    buy_days = daily[daily["is_buy"] == 1]
    hold_days = daily[daily["is_buy"] == 0]
    buy_trades = trades[trades["action"] == "구매"]
    sell_trades = trades[trades["action"] == "판매"]

    candidates = []

    # ── R1: GLD 필터 ──
    gld_buy = buy_days["GLD_pct"].dropna().mean()
    gld_hold = hold_days["GLD_pct"].dropna().mean()
    gld_hi_buy_rate = daily[(daily["GLD_pct"] > 1.0) & daily["GLD_pct"].notna()]["is_buy"].mean()
    gld_lo_buy_rate = daily[(daily["GLD_pct"] < -0.5) & daily["GLD_pct"].notna()]["is_buy"].mean()
    candidates.append({
        "id": "R1",
        "name": "GLD 시황 필터",
        "condition": "GLD > +1.0% → 매수 억제",
        "evidence": f"GLD>1% 매수율 {gld_hi_buy_rate*100:.0f}% vs GLD<-0.5% {gld_lo_buy_rate*100:.0f}%",
        "type": "진입 필터",
    })

    # ── R2: 쌍둥이 갭 진입 ──
    for col in [c for c in daily.columns if c.startswith("gap_") and not c.endswith("_abs")]:
        vals = buy_trades[col].dropna()
        if len(vals) < 10:
            continue
        pos_rate = (vals > 0).mean()
        candidates.append({
            "id": f"R2_{col}",
            "name": f"쌍둥이 갭 진입 ({col})",
            "condition": f"갭 > 0% 일 때 매수 집중 (양갭 매수 비율 {pos_rate*100:.0f}%)",
            "evidence": f"평균 갭 {vals.mean():+.2f}%, 중앙값 {vals.median():+.2f}%",
            "type": "진입 조건",
        })

    # ── R3: BTC 확률 ──
    btc_buy = buy_days["poly_btc_up"].dropna().mean()
    btc_hold = hold_days["poly_btc_up"].dropna().mean()
    candidates.append({
        "id": "R3",
        "name": "BTC 확률 필터",
        "condition": f"BTC up 확률 기반 (매수일 평균 {btc_buy:.2f} vs 관망일 {btc_hold:.2f})",
        "evidence": f"차이 {btc_buy - btc_hold:+.3f}",
        "type": "시황 판단",
    })

    # ── R4: 매도 기준 ──
    sell_ticker_pct = sell_trades["ticker_pct"].dropna()
    if len(sell_ticker_pct) > 0:
        profit_sells = (sell_ticker_pct > 0).mean()
        median_at_sell = sell_ticker_pct.median()
        candidates.append({
            "id": "R4",
            "name": "이익실현 매도",
            "condition": f"종목 +{median_at_sell:.1f}% 이상 시 매도 (이익실현 {profit_sells*100:.0f}%)",
            "evidence": f"매도 시점 종목 등락 중앙값 {median_at_sell:+.2f}%",
            "type": "청산 조건",
        })

    # ── R5: DCA 제한 ──
    daily_dca = buy_trades.groupby(["date", "yf_ticker"]).size()
    max_dca = daily_dca.max()
    avg_dca = daily_dca.mean()
    candidates.append({
        "id": "R5",
        "name": "DCA 횟수 제한",
        "condition": f"일일 동일종목 매수 상한 (현재 최대 {max_dca}회, 평균 {avg_dca:.1f}회)",
        "evidence": "과다 DCA 시기에 손실 집중 가능성",
        "type": "사이징 규칙",
    })

    # ── R6: 하락 매수 (역발상) ──
    spy_down_buy_rate = daily[(daily["SPY_pct"] < -1.0) & daily["SPY_pct"].notna()]["is_buy"].mean()
    base_rate = daily["is_buy"].mean()
    candidates.append({
        "id": "R6",
        "name": "하락장 역발상 매수",
        "condition": f"SPY < -1.0% 시 매수 강화 (매수율 {spy_down_buy_rate*100:.0f}% vs 기저 {base_rate*100:.0f}%)",
        "evidence": f"기저 대비 +{(spy_down_buy_rate - base_rate)*100:.1f}%p",
        "type": "진입 조건",
    })

    # 출력
    print()
    for c in candidates:
        print(f"  [{c['id']}] {c['name']} ({c['type']})")
        print(f"      조건: {c['condition']}")
        print(f"      근거: {c['evidence']}")
        print()

    # CSV 저장
    cdf = pd.DataFrame(candidates)
    cdf.to_csv(ANALYSIS_DIR / "rule_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"  저장: {ANALYSIS_DIR / 'rule_candidates.csv'}")

    return candidates


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  규칙 추출 v2 — 심화 분석 (갭 + 금액 + WF)     ║")
    print("  ╚══════════════════════════════════════════════════╝")

    daily, trades = load_and_enrich()
    print(f"  일별 로그: {len(daily)}일 (갭 변수 추가)")
    print(f"  거래 로그: {len(trades)}건")

    # 1. 쌍둥이 갭 분석
    twin_gap_analysis(daily, trades)

    # 2. 금액 가중 분석
    amount_weighted_analysis(daily, trades)

    # 3. 심화 Decision Tree
    enhanced_decision_tree(daily)

    # 4. Walk-Forward 검증
    walk_forward_validation(daily)

    # 5. 종합 규칙 후보
    summarize_rule_candidates(daily, trades)

    print("\n  완료!")
    print()


if __name__ == "__main__":
    main()

"""
실제 거래내역 vs Polymarket 급락 신호 비교 분석
================================================
질문:
  1. 실제 매도일의 crash_score 분포 — 신호와 일치했나?
  2. 실제 매수일의 crash_score 분포 — 저점 매수 타이밍?
  3. 모델 포지션 vs 실제 보유 종목 일치도
  4. 신호가 높았을 때 실제로 손실이 발생했나?

종목명 매핑:
  '프로셰어즈 QQQ 3배*'  → TQQQ
  '디렉시온 미국 반도체'  → SOXL
  '비트코인 전략 2배 ETF' → BITU
  '그래닛셰어즈 코인베이스'→ COIN
  '일드맥스 테슬라 옵션*' → TSLL
  '디렉시온 테슬라 2배*'  → TSLL
  'T-REX 테슬라 데일리'   → TSLL
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.polymarket_crash_model import (
    load_poly_signals,
    load_market_data,
    compute_crash_score,
    assign_position,
    mode_label,
    TRAIN_START, TARGET_END,
)

HISTORY_CSV = ROOT / "history/거래내역_20231006_20260212.csv"
CHART_DIR   = ROOT / "docs/charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── 종목 매핑 ──────────────────────────────────
TICKER_MAP = {
    "프로셰어즈 QQQ 3배":      "TQQQ",
    "프로셰어즈 QQQ 3배 ETF":  "TQQQ",
    "프로셰어즈 QQQ 3배 ET":   "TQQQ",
    "디렉시온 미국 반도체":     "SOXL",
    "비트코인 전략 2배 ETF(U":  "BITU",
    "비트코인 전략 2배 ET":     "BITU",
    "비트코인 전략 2배 ETF":    "BITU",
    "그래닛셰어즈 코인베이스":  "COIN",
    "일드맥스 코인베이스 옵션": "COIN",
    "일드맥스 테슬라 옵션 인":  "TSLL",
    "일드맥스 테슬라 옵션 인컴":"TSLL",
    "디렉시온 테슬라 2배 ETF":  "TSLL",
    "디렉시온 테슬라 2배 E":    "TSLL",
    "T-REX 테슬라 데일리":      "TSLL",
    "프로셰어즈 비트코인 선물": "BITU",
    "JP모건 커버드콜 옵션 ETF": "JEPQ",
    "JP모건 커버드콜 옵션":     "JEPQ",
    "JP모건 커버드콜 옵션 E":   "JEPQ",
    "JP모건 나스닥 프리미":     "JEPQ",
    "JP모건 나스닥 프리미엄":   "JEPQ",
    "애플":                     "AAPL",
}

RISK_ASSETS = {"TQQQ", "SOXL", "BITU", "TSLL", "COIN"}


# ── 데이터 로드 ────────────────────────────────

def load_signal() -> pd.DataFrame:
    poly_df   = load_poly_signals()
    market_df = load_market_data()
    merged    = poly_df.join(market_df, how="left")
    merged    = compute_crash_score(merged)
    return merged


def load_history() -> pd.DataFrame:
    df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    df["거래일자"] = pd.to_datetime(df["거래일자"])
    df["ticker"] = df["종목명"].map(TICKER_MAP).fillna("기타")
    df["is_risk"] = df["ticker"].isin(RISK_ASSETS)
    df["is_buy"]  = df["거래구분"] == "구매"
    df["is_sell"] = df["거래구분"] == "판매"
    return df


# ── 분석 ───────────────────────────────────────

def analyze(signal: pd.DataFrame, history: pd.DataFrame) -> dict:
    # 거래내역과 crash_score 병합
    hist = history.copy()
    hist = hist.set_index("거래일자")
    hist["crash_score"] = hist.index.map(
        lambda d: signal.loc[d, "crash_score"]
        if d in signal.index else np.nan
    )
    hist["mode"] = hist["crash_score"].apply(
        lambda s: mode_label(s) if not pd.isna(s) else "N/A"
    )
    hist["model_soxl_w"] = hist["crash_score"].apply(
        lambda s: assign_position(s).get("SOXL", 0) if not pd.isna(s) else np.nan
    )

    # 위험자산만
    risk = hist[hist["is_risk"]].copy()

    # --- 1. 매수/매도일 crash_score 분포 ---
    buy_scores  = risk[risk["is_buy"]]["crash_score"].dropna()
    sell_scores = risk[risk["is_sell"]]["crash_score"].dropna()

    # --- 2. 월별 거래 규모와 crash_score ---
    daily_buy  = risk[risk["is_buy"]].groupby(level=0)["거래대금_달러"].sum()
    daily_sell = risk[risk["is_sell"]].groupby(level=0)["거래대금_달러"].sum()

    # --- 3. 신호 수준별 실제 거래 성향 ---
    bins = [0, 0.15, 0.30, 0.50, 0.70, 1.0]
    labels = ["BULL\n<0.15", "LONG-\n0.15-0.30", "NEUTRAL\n0.30-0.50",
              "SHORT+\n0.50-0.70", "CRASH\n>0.70"]
    risk["score_bin"] = pd.cut(risk["crash_score"], bins=bins, labels=labels)
    bin_stats = risk.groupby(["score_bin", "거래구분"])["거래대금_달러"].sum().unstack(fill_value=0)

    # --- 4. 고신호(>0.35) 구간 이후 실제 시장 손익 ---
    high_signal_days = signal[signal["crash_score"] > 0.35].index
    next_day_soxl = []
    for dt in high_signal_days:
        idx = signal.index.get_loc(dt)
        if idx + 1 < len(signal):
            next_ret = signal["SOXL_ret"].iloc[idx + 1]
            if not pd.isna(next_ret):
                next_day_soxl.append(next_ret)

    # --- 5. 모델 포지션 vs 실제 보유 일치도 ---
    match_days = []
    for dt in signal.index:
        sc = float(signal.loc[dt, "crash_score"])
        pos = assign_position(sc)
        dominant = max(pos, key=pos.get)
        day_risk = risk[risk.index == dt]
        actual_buys  = day_risk[day_risk["is_buy"]]["ticker"].tolist()
        actual_sells = day_risk[day_risk["is_sell"]]["ticker"].tolist()
        if actual_buys or actual_sells:
            match_days.append({
                "date": dt,
                "model_dominant": dominant,
                "actual_buy": actual_buys,
                "actual_sell": actual_sells,
                "crash_score": sc,
            })

    return {
        "hist": hist,
        "risk": risk,
        "buy_scores": buy_scores,
        "sell_scores": sell_scores,
        "daily_buy": daily_buy,
        "daily_sell": daily_sell,
        "bin_stats": bin_stats,
        "high_signal_days": high_signal_days,
        "next_day_soxl": next_day_soxl,
        "match_days": match_days,
        "signal": signal,
    }


# ── 리포트 출력 ────────────────────────────────

def print_report(result: dict) -> None:
    buy_scores  = result["buy_scores"]
    sell_scores = result["sell_scores"]
    match_days  = result["match_days"]
    next_day_soxl = result["next_day_soxl"]

    print("\n" + "=" * 65)
    print("실제 거래내역 vs Polymarket 급락 신호 비교 분석")
    print("=" * 65)

    print(f"\n【1. 위험자산 매수/매도일 Crash Score 분포】")
    print(f"  매수일: n={len(buy_scores):3d}  평균={buy_scores.mean():.3f}  "
          f"중앙={buy_scores.median():.3f}  max={buy_scores.max():.3f}")
    print(f"  매도일: n={len(sell_scores):3d}  평균={sell_scores.mean():.3f}  "
          f"중앙={sell_scores.median():.3f}  max={sell_scores.max():.3f}")

    if sell_scores.mean() > buy_scores.mean():
        diff = sell_scores.mean() - buy_scores.mean()
        print(f"\n  → 매도일 평균 crash_score가 매수일보다 {diff:.3f} 높음")
        print(f"  → 실제 거래가 신호와 방향 일치 ✓")
    else:
        diff = buy_scores.mean() - sell_scores.mean()
        print(f"\n  → 매수일 crash_score가 오히려 {diff:.3f} 높음")
        print(f"  → 신호와 역방향 매매 패턴 존재")

    print(f"\n【2. Crash Score 구간별 매수/매도 거래 금액($)】")
    bs = result["bin_stats"]
    header = f"  {'구간':20s} {'매수($)':>12s} {'매도($)':>12s} {'매도비율':>8s}"
    print(header)
    print("  " + "-" * 56)
    for idx in bs.index:
        buy_amt  = bs.loc[idx, "구매"] if "구매" in bs.columns else 0
        sell_amt = bs.loc[idx, "판매"] if "판매" in bs.columns else 0
        total    = buy_amt + sell_amt
        sell_pct = sell_amt / total * 100 if total > 0 else 0
        print(f"  {str(idx):20s} ${buy_amt:>10,.0f} ${sell_amt:>10,.0f} {sell_pct:>7.0f}%")

    print(f"\n【3. 고신호(>0.35) 다음날 SOXL 수익률】")
    if next_day_soxl:
        arr = np.array(next_day_soxl)
        neg_pct = (arr < 0).mean() * 100
        print(f"  n={len(arr)}, 평균={arr.mean():+.2%}, 음수비율={neg_pct:.0f}%")
        print(f"  최대손실={arr.min():+.2%}, 최대이익={arr.max():+.2%}")
        if neg_pct >= 50:
            print(f"  → crash_score > 0.35 이후 SOXL 하락 빈도가 높음 (신호 유효)")
        else:
            print(f"  → crash_score > 0.35 이후 방향성 불명확")

    print(f"\n【4. 거래일 모델 신호 vs 실제 매매 (최근 30건)】")
    print(f"  {'날짜':12s} {'score':6s} {'모델':8s} {'실제 매수':20s} {'실제 매도':20s}")
    print("  " + "-" * 72)
    recent = sorted(match_days, key=lambda x: x["date"], reverse=True)[:30]
    for m in recent:
        buy_str  = ",".join(m["actual_buy"][:2])  or "-"
        sell_str = ",".join(m["actual_sell"][:2]) or "-"
        print(
            f"  {str(m['date'].date()):12s} "
            f"{m['crash_score']:5.2f} "
            f"{m['model_dominant']:8s} "
            f"{buy_str:20s} "
            f"{sell_str:20s}"
        )


# ── 시각화 ─────────────────────────────────────

def plot(result: dict) -> None:
    signal      = result["signal"]
    risk        = result["risk"]
    buy_scores  = result["buy_scores"]
    sell_scores = result["sell_scores"]
    daily_buy   = result["daily_buy"]
    daily_sell  = result["daily_sell"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "실제 거래내역 vs Polymarket Crash Signal 비교",
        fontsize=14, fontweight="bold",
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # ── A: Crash Score + 실제 매수/매도 오버레이 ─
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_facecolor("#fafafa")
    ax_a.fill_between(signal.index, signal["crash_score"],
                      alpha=0.2, color="#7b1fa2")
    ax_a.plot(signal.index, signal["crash_score"],
              color="#7b1fa2", lw=1.5, label="Crash Score")
    ax_a.axhline(0.35, color="#ff5722", lw=0.8, ls="--", alpha=0.8)

    # 매수 마커
    for dt, row in risk[risk["is_buy"]].iterrows():
        if dt in signal.index:
            sc = signal.loc[dt, "crash_score"]
            ax_a.scatter(dt, sc, color="#1565c0", s=40, zorder=5, alpha=0.7, marker="^")

    # 매도 마커
    for dt, row in risk[risk["is_sell"]].iterrows():
        if dt in signal.index:
            sc = signal.loc[dt, "crash_score"]
            ax_a.scatter(dt, sc, color="#c62828", s=40, zorder=5, alpha=0.7, marker="v")

    from matplotlib.lines import Line2D
    buy_marker  = Line2D([0],[0], marker="^", color="w", markerfacecolor="#1565c0", ms=8, label="실제 매수")
    sell_marker = Line2D([0],[0], marker="v", color="w", markerfacecolor="#c62828", ms=8, label="실제 매도")
    ax_a.legend(handles=[
        plt.Line2D([0],[0], color="#7b1fa2", lw=2, label="Crash Score"),
        buy_marker, sell_marker,
    ], fontsize=9, loc="upper left")
    ax_a.set_title("Crash Score 타임라인 + 실제 매수(▲)/매도(▼)", fontsize=11)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_a.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax_a.grid(axis="y", alpha=0.3)

    # ── B: 매수/매도일 Crash Score 히스토그램 ────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_facecolor("#fafafa")
    bins = np.linspace(0, 1, 21)
    ax_b.hist(buy_scores,  bins=bins, alpha=0.65, color="#1565c0", label=f"매수 (n={len(buy_scores)})")
    ax_b.hist(sell_scores, bins=bins, alpha=0.65, color="#c62828", label=f"매도 (n={len(sell_scores)})")
    ax_b.axvline(buy_scores.mean(),  color="#1565c0", lw=1.5, ls="--")
    ax_b.axvline(sell_scores.mean(), color="#c62828", lw=1.5, ls="--")
    ax_b.set_title("매수/매도일 Crash Score 분포", fontsize=11)
    ax_b.set_xlabel("Crash Score")
    ax_b.set_ylabel("거래 일수")
    ax_b.legend(fontsize=9)
    ax_b.grid(axis="y", alpha=0.3)

    # ── C: 구간별 매수/매도 금액 ─────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_facecolor("#fafafa")
    bs = result["bin_stats"]
    x   = np.arange(len(bs.index))
    w   = 0.35
    buy_vals  = [bs.loc[i, "구매"] if "구매" in bs.columns else 0 for i in bs.index]
    sell_vals = [bs.loc[i, "판매"] if "판매" in bs.columns else 0 for i in bs.index]
    ax_c.bar(x - w/2, buy_vals,  width=w, color="#1565c0", alpha=0.75, label="매수")
    ax_c.bar(x + w/2, sell_vals, width=w, color="#c62828", alpha=0.75, label="매도")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(i) for i in bs.index], fontsize=8)
    ax_c.set_title("Crash Score 구간별 매수/매도 금액($)", fontsize=11)
    ax_c.set_ylabel("거래금액 ($)")
    ax_c.legend(fontsize=9)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax_c.grid(axis="y", alpha=0.3)

    # ── D: 월별 거래 금액과 Crash Score ──────────
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.set_facecolor("#fafafa")
    monthly_buy  = daily_buy.resample("ME").sum()
    monthly_sell = daily_sell.resample("ME").sum()
    monthly_score = signal["crash_score"].resample("ME").mean()

    ax_d.bar(monthly_buy.index,  monthly_buy.values,  width=15, color="#1565c0", alpha=0.7, label="월 매수액")
    ax_d.bar(monthly_sell.index, -monthly_sell.values, width=15, color="#c62828", alpha=0.7, label="월 매도액")
    ax_d.axhline(0, color="#333", lw=0.8)

    ax_d2 = ax_d.twinx()
    ax_d2.plot(monthly_score.index, monthly_score.values,
               color="#7b1fa2", lw=2, marker="o", ms=4, label="월평균 Crash Score")
    ax_d2.set_ylabel("Crash Score", color="#7b1fa2", fontsize=10)
    ax_d2.tick_params(axis="y", labelcolor="#7b1fa2")
    ax_d2.set_ylim(0, 0.8)

    ax_d.set_title("월별 거래 금액 vs 평균 Crash Score", fontsize=11)
    ax_d.set_ylabel("거래금액 ($)")
    ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${abs(x)/1000:.0f}K"))

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax_d.grid(axis="y", alpha=0.3)

    out_path = CHART_DIR / "trade_history_vs_signal.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"\n차트 저장: {out_path}")


# ── 메인 ───────────────────────────────────────

def main() -> None:
    print("데이터 로드 중...")
    signal  = load_signal()
    history = load_history()
    print(f"  신호: {len(signal)}일, 거래내역: {len(history)}건")
    print(f"  위험자산 거래: {history['is_risk'].sum()}건 "
          f"(매수 {history[history['is_buy'] & history['is_risk']].shape[0]}건 "
          f"/ 매도 {history[history['is_sell'] & history['is_risk']].shape[0]}건)")

    result = analyze(signal, history)
    print_report(result)
    plot(result)


if __name__ == "__main__":
    main()

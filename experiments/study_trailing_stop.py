#!/usr/bin/env python3
"""
축 2 스터디: 트레일링 스탑 EDA
=================================
backtest_1min_3y.parquet (1분봉)를 사용해 각 포지션의 실제 최고점을 측정하고,
다양한 트레일링 스탑 시나리오별 기대 수익을 비교한다.

핵심 질문:
  - 각 포지션이 보유 기간 중 실제로 몇 % 최고점까지 올랐나?
  - 현재 +5.9% 고정 익절 대신 트레일링 스탑을 쓰면 얼마나 더 가져올 수 있나?
  - score 0.95+ 포지션에서 효과가 특히 큰가?

시나리오 비교:
  - v1 (현재): 고정 +5.9% 익절 or 7일 강제청산
  - T-2%: 고점 대비 -2% 하락 시 청산 (단, 고점이 +5% 이상일 때만 활성화)
  - T-3%: 고점 대비 -3% 하락 시 청산
  - T-5%: 고점 대비 -5% 하락 시 청산
  - Fix+8%, Fix+10%, Fix+12%: 고정 threshold 상향 시나리오

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_trailing_stop.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ============================================================
# 경로 상수
# ============================================================
TRADES_CSV   = _ROOT / "data" / "results" / "backtests" / "d2s_trades.csv"
MIN1_PARQUET = _ROOT / "data" / "market" / "ohlcv" / "backtest_1min_3y.parquet"
OUT_CSV      = _ROOT / "data" / "results" / "analysis" / "trailing_stop_eda.csv"

# 트레일링 스탑 시나리오 정의
# (trail_pct, min_profit_to_activate)
TRAIL_SCENARIOS = {
    "T-2%": (2.0, 5.0),
    "T-3%": (3.0, 5.0),
    "T-5%": (5.0, 5.0),
}
# 고정 threshold 시나리오 (이 이상 달성 시 청산)
FIXED_SCENARIOS = [8.0, 10.0, 12.0]

# 트레일링 스탑 대상 최소 score
TRAIL_SCORE_MIN = 0.95


# ============================================================
# 1분봉 데이터 로드 (D2S 4종목만)
# ============================================================

def load_1min(tickers: list[str]) -> pd.DataFrame:
    print("[1/4] 1분봉 데이터 로드...")
    df = pd.read_parquet(MIN1_PARQUET)
    df = df[df["symbol"].isin(tickers)].copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["ts"].dt.date
    print(f"  {len(df):,}행 로드 완료 ({', '.join(tickers)})")
    return df


# ============================================================
# 포지션 단위 데이터 구성 (BUY/SELL 매핑)
# ============================================================

def build_positions(trades_csv: Path) -> pd.DataFrame:
    """d2s_trades.csv → 포지션 단위 DataFrame.

    각 포지션:
      - entry_date  : 첫 번째 BUY 날짜
      - exit_date   : SELL 날짜
      - ticker
      - entry_price : DCA 평균 진입가 (마지막 BUY의 가중평균)
      - entry_score : 첫 번째 BUY의 score (진입 판단 시점)
      - dca_count   : BUY 횟수
      - actual_pnl_pct : 실제 청산 수익률 (v1 백테스트 결과)
      - exit_type   : take_profit / forced
    """
    print("[2/4] 포지션 단위 데이터 구성...")
    df = pd.read_csv(trades_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    positions = []
    pos_buys: dict[str, list] = {}  # ticker -> buy rows

    for _, row in df.sort_values("date").iterrows():
        t = row["ticker"]
        if row["side"] == "BUY":
            pos_buys.setdefault(t, []).append(row)
        elif row["side"] == "SELL":
            if t in pos_buys and pos_buys[t]:
                buys = pos_buys[t]
                # 가중평균 진입가
                total_cost = sum(b["amount"] - b["fee"] for b in buys)
                total_qty  = sum((b["amount"] - b["fee"]) / b["price"] for b in buys)
                avg_entry  = total_cost / total_qty if total_qty > 0 else buys[0]["price"]

                positions.append({
                    "ticker":         t,
                    "entry_date":     buys[0]["date"],
                    "exit_date":      row["date"],
                    "entry_price":    avg_entry,
                    "entry_score":    buys[0]["score"],   # 첫 진입 score
                    "max_score":      max(b["score"] for b in buys),
                    "dca_count":      len(buys),
                    "actual_pnl_pct": row["pnl_pct"],
                    "actual_pnl":     row["pnl"],
                    "exit_type":      "forced" if "hold_days" in str(row["reason"]) else "take_profit",
                    "entry_reason":   buys[0]["reason"],
                })
                pos_buys[t] = []

    pos_df = pd.DataFrame(positions)
    print(f"  포지션 {len(pos_df)}개 구성 완료")
    return pos_df


# ============================================================
# 트레일링 스탑 시뮬레이션 (포지션별, 일봉 단위)
# ============================================================

def simulate_trailing_stop(
    entry_price: float,
    hold_data: pd.DataFrame,   # date별 OHLCV (1분봉 그룹화된 일봉)
    trail_pct: float,
    min_profit: float,
) -> tuple[float, date | None]:
    """일봉 단위 트레일링 스탑 시뮬레이션.

    Parameters
    ----------
    entry_price : 평균 진입가
    hold_data   : 보유 기간 일별 (date, open, high, low, close) DataFrame
    trail_pct   : 고점 대비 하락 허용 % (예: 3.0 → -3%)
    min_profit  : 트레일링 활성화 최소 수익률 % (예: 5.0 → +5% 달성 후 활성화)

    Returns
    -------
    (exit_pct, exit_date) : 청산 수익률과 청산일 (미청산이면 None)
    """
    peak_price = entry_price
    trail_active = False

    for _, row in hold_data.iterrows():
        daily_high  = row["high"]
        daily_close = row["close"]

        # 고점 갱신
        if daily_high > peak_price:
            peak_price = daily_high

        # 트레일링 활성화 조건: peak가 min_profit 이상 도달
        peak_gain = (peak_price / entry_price - 1) * 100
        if peak_gain >= min_profit:
            trail_active = True

        # 트레일링 스탑 체크: close가 peak 대비 trail_pct 이상 하락
        if trail_active:
            drop_from_peak = (daily_close / peak_price - 1) * 100
            if drop_from_peak <= -trail_pct:
                exit_pct = (daily_close / entry_price - 1) * 100
                return exit_pct, row["date"]

    # 보유 기간 내 미청산
    if len(hold_data) > 0:
        last_close = hold_data.iloc[-1]["close"]
        exit_pct = (last_close / entry_price - 1) * 100
        return exit_pct, hold_data.iloc[-1]["date"]
    return 0.0, None


# ============================================================
# 메인 분석
# ============================================================

def main():
    tickers = ["AMDL", "CONL", "MSTU", "ROBN"]

    # 데이터 로드
    min1_df = load_1min(tickers)
    pos_df  = build_positions(TRADES_CSV)

    print("[3/4] 포지션별 고점 추적 및 시나리오 시뮬레이션...")

    # 1분봉 → 일봉 집계 (date별 OHLCV)
    daily_1min = (
        min1_df.groupby(["symbol", "date"])
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"),
             volume=("volume", "sum"))
        .reset_index()
    )

    records = []
    skipped = 0

    for _, pos in pos_df.iterrows():
        ticker      = pos["ticker"]
        entry_date  = pos["entry_date"]
        exit_date   = pos["exit_date"]
        score       = pos["entry_score"]

        # 보유 기간 1분봉 데이터
        mask = (
            (daily_1min["symbol"] == ticker) &
            (daily_1min["date"] >= entry_date) &
            (daily_1min["date"] <= exit_date)
        )
        hold_data = daily_1min[mask].sort_values("date").reset_index(drop=True)

        if len(hold_data) == 0:
            skipped += 1
            continue

        # ── entry_price: 1분봉 진입 당일 종가 기준 ──
        # market_daily(수정가) vs 1분봉(미수정가) 불일치 해결.
        # 레버리지 ETF는 reverse split이 잦아 두 가격이 크게 다를 수 있음.
        # 1분봉 첫째 날 close를 기준가로 사용해 내부 일관성 확보.
        entry_price = hold_data.iloc[0]["close"]

        # 실제 고점 (보유 기간 최대 high)
        peak_high       = hold_data["high"].max()
        peak_return_pct = (peak_high / entry_price - 1) * 100
        peak_day        = hold_data.loc[hold_data["high"].idxmax(), "date"]
        last_close      = hold_data.iloc[-1]["close"]

        # 고점에서 실제 exit까지 하락폭
        drop_from_peak = (pos["actual_pnl_pct"] - peak_return_pct)  # 음수 = 고점 대비 손실

        rec = {
            "ticker":           ticker,
            "entry_date":       entry_date,
            "exit_date":        exit_date,
            "hold_days":        (exit_date - entry_date).days,
            "entry_score":      score,
            "max_score":        pos["max_score"],
            "dca_count":        pos["dca_count"],
            "exit_type":        pos["exit_type"],
            "actual_pnl_pct":   round(pos["actual_pnl_pct"], 2),
            "actual_pnl":       round(pos["actual_pnl"], 1),
            "peak_return_pct":  round(peak_return_pct, 2),
            "peak_day":         peak_day,
            "days_to_peak":     (peak_day - entry_date).days,
            "drop_from_peak":   round(drop_from_peak, 2),
            "is_trail_target":  score >= TRAIL_SCORE_MIN,
        }

        # 시나리오별 시뮬레이션
        for name, (trail_pct, min_profit) in TRAIL_SCENARIOS.items():
            sim_pct, sim_date = simulate_trailing_stop(
                entry_price, hold_data, trail_pct, min_profit
            )
            rec[f"{name}_pnl_pct"] = round(sim_pct, 2)
            # 개선액: 트레일링 스탑 - 실제 결과
            rec[f"{name}_gain_vs_actual"] = round(sim_pct - pos["actual_pnl_pct"], 2)

        # 고정 threshold 시나리오: peak가 threshold 이상이면 threshold에서 나온 것으로 간주
        for thresh in FIXED_SCENARIOS:
            if peak_return_pct >= thresh:
                rec[f"fix{thresh:.0f}_pnl_pct"]         = thresh
                rec[f"fix{thresh:.0f}_gain_vs_actual"]   = round(thresh - pos["actual_pnl_pct"], 2)
            else:
                # peak가 threshold에 못 미침 → 실제와 동일하게 나옴
                rec[f"fix{thresh:.0f}_pnl_pct"]         = round(pos["actual_pnl_pct"], 2)
                rec[f"fix{thresh:.0f}_gain_vs_actual"]   = 0.0

        records.append(rec)

    result_df = pd.DataFrame(records)

    if skipped > 0:
        print(f"  ⚠️  1분봉 데이터 없어서 스킵된 포지션: {skipped}개")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"  결과 저장: {OUT_CSV}")

    # ============================================================
    # 결과 출력
    # ============================================================
    print("\n" + "=" * 65)
    print("  축 2 스터디: 트레일링 스탑 EDA 결과")
    print("=" * 65)

    total = len(result_df)
    trail_target = result_df[result_df["is_trail_target"]]

    print(f"\n  분석 포지션: {total}개")
    print(f"  트레일링 스탑 대상 (score≥{TRAIL_SCORE_MIN}): {len(trail_target)}개")

    # --- 전체 고점 분포 ---
    print("\n  [1] 보유 기간 중 실제 고점 수익률 분포")
    print(f"    {'전체':12s}  평균 {result_df['peak_return_pct'].mean():+.1f}%  "
          f"중앙값 {result_df['peak_return_pct'].median():+.1f}%  "
          f"최대 {result_df['peak_return_pct'].max():+.1f}%")
    print(f"    {'score<0.95':12s}  평균 "
          f"{result_df[~result_df['is_trail_target']]['peak_return_pct'].mean():+.1f}%")
    if len(trail_target) > 0:
        print(f"    {'score≥0.95':12s}  평균 "
              f"{trail_target['peak_return_pct'].mean():+.1f}%  "
              f"(실제 청산 평균 {trail_target['actual_pnl_pct'].mean():+.1f}%)")

    # 고점 구간별 분포
    print()
    bins = [(-100, 0, "손실권(<0%)"), (0, 5.9, "0~5.9%"),
            (5.9, 10, "5.9~10%"), (10, 20, "10~20%"), (20, 200, "20%+")]
    for lo, hi, label in bins:
        n = ((result_df["peak_return_pct"] >= lo) &
             (result_df["peak_return_pct"] < hi)).sum()
        print(f"    고점 {label:12s}: {n:3d}건 ({n/total*100:.1f}%)")

    # --- 시나리오 비교 ---
    print("\n  [2] 시나리오별 평균 수익률 비교")
    print(f"    {'시나리오':18s}  {'전체 평균':>10s}  {'score≥0.95':>12s}  {'vs v1 개선':>10s}")
    print(f"    {'-'*55}")

    # v1 baseline
    v1_all  = result_df["actual_pnl_pct"].mean()
    v1_top  = trail_target["actual_pnl_pct"].mean() if len(trail_target) > 0 else 0
    print(f"    {'v1 (현재 고정+5.9%)':18s}  {v1_all:>+9.1f}%  {v1_top:>+11.1f}%  {'기준':>10s}")

    for name in TRAIL_SCENARIOS:
        col     = f"{name}_pnl_pct"
        avg_all = result_df[col].mean()
        avg_top = trail_target[col].mean() if len(trail_target) > 0 else 0
        diff    = avg_all - v1_all
        print(f"    {name:18s}  {avg_all:>+9.1f}%  {avg_top:>+11.1f}%  {diff:>+9.1f}%p")

    for thresh in FIXED_SCENARIOS:
        col     = f"fix{thresh:.0f}_pnl_pct"
        avg_all = result_df[col].mean()
        avg_top = trail_target[col].mean() if len(trail_target) > 0 else 0
        diff    = avg_all - v1_all
        print(f"    {'Fix+'+f'{thresh:.0f}%':18s}  {avg_all:>+9.1f}%  {avg_top:>+11.1f}%  {diff:>+9.1f}%p")

    # --- score 0.95+ 상세 ---
    if len(trail_target) > 0:
        print(f"\n  [3] score≥{TRAIL_SCORE_MIN} 포지션 상세 ({len(trail_target)}건)")
        print(f"    {'ticker':6s}  {'entry':10s}  {'score':6s}  "
              f"{'실제%':7s}  {'고점%':7s}  {'T-3%':7s}  {'exit_type':12s}")
        print(f"    {'-'*65}")
        for _, row in trail_target.sort_values("actual_pnl_pct", ascending=False).iterrows():
            print(
                f"    {row['ticker']:6s}  {str(row['entry_date']):10s}  "
                f"{row['entry_score']:.2f}    "
                f"{row['actual_pnl_pct']:>+6.1f}%  "
                f"{row['peak_return_pct']:>+6.1f}%  "
                f"{row['T-3%_pnl_pct']:>+6.1f}%  "
                f"{row['exit_type']}"
            )

    # --- 트레일링 스탑 적용 시 추가 이익 시뮬레이션 ---
    print(f"\n  [4] score≥{TRAIL_SCORE_MIN} 포지션에만 T-3% 적용 시 총 PnL 변화 추정")
    # 트레일링 대상: actual_pnl을 T-3% 결과로 대체
    # 나머지: actual_pnl 그대로
    if len(trail_target) > 0:
        non_trail = result_df[~result_df["is_trail_target"]]
        total_pnl_v1  = result_df["actual_pnl"].sum()

        # T-3%: pnl을 비율로 환산 (actual_pnl × T-3%_pnl_pct / actual_pnl_pct)
        trail_pnl_t3 = trail_target.apply(
            lambda r: r["actual_pnl"] * (r["T-3%_pnl_pct"] / r["actual_pnl_pct"])
            if r["actual_pnl_pct"] != 0 else r["actual_pnl"],
            axis=1
        ).sum()
        total_pnl_t3 = non_trail["actual_pnl"].sum() + trail_pnl_t3

        print(f"    v1 총 PnL: ${total_pnl_v1:+,.0f}")
        print(f"    T-3% 적용 총 PnL (추정): ${total_pnl_t3:+,.0f}  (Δ ${total_pnl_t3 - total_pnl_v1:+,.0f})")
        print(f"    ※ 추정치: v1 비율 기준 환산. 실제 백테스트로 검증 필요.")

    print("\n" + "=" * 65)
    print(f"  상세 데이터: {OUT_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()

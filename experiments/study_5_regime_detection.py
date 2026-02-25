#!/usr/bin/env python3
"""
Study 5 — 레짐 감지 스터디 (Regime Detection)
================================================
attach v2 부록 H 핵심 발견:
  hd=4, tp=5.0% → OOS +25.97%p (기준 -20.1% → +5.87%)
  단, IS에서 -16.73%p 손실 (bull market 상승 기회 절단)
  → 레짐 조건부 적용이 필요: Bull=hd7/tp5.9%, Bear=hd4/tp5.0%

레짐 감지 3차원 신호:
  1. SPY streak: 연속 상승/하락일 수 (이미 R13에서 활용)
  2. SPY SMA: 가격/이평선 비율 (추세 방향)
  3. Polymarket BTC up 확률: Risk-on/off 시장 심리 (이미 R3에서 활용)
  → 3차원 다수결로 bull/bear/neutral 판정

핵심 질문:
  Q1. 레짐 감지 신호(SPY streak, SMA, Polymarket)가 IS/OOS 기간을 올바르게 구분하는가?
  Q2. 레짐 조건부 청산이 고정 청산 파라미터보다 우월한가?
  Q3. 최적 레짐 감지 임계값은?
  Q4. Polymarket BTC 확률이 SPY 기반 레짐 감지를 보완하는가?
  Q5. R19 BB 하드 필터와의 시너지 효과?

3단계 구성:
  Phase 1: EDA — IS/OOS 레짐 분포 분석 + 레짐 신호 검증
  Phase 2: Grid — 레짐 임계값 × 청산 파라미터 2D 탐색
  Phase 3: 최적 조합 IS/OOS 비교

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_5_regime_detection.py
    pyenv shell ptj_stock_lab && python experiments/study_5_regime_detection.py --phase 1
    pyenv shell ptj_stock_lab && python experiments/study_5_regime_detection.py --phase 2 --n-jobs 8
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import date
from multiprocessing import Pool
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2, D2S_ENGINE_V3
from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2
from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3

# ============================================================
# 상수
# ============================================================

IS_START  = date(2025, 3, 3)
IS_END    = date(2025, 9, 30)
OOS_START = date(2025, 10, 1)
OOS_END   = date(2026, 2, 17)
FULL_START = IS_START
FULL_END   = OOS_END

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Phase 1: EDA — 레짐 분포 분석
# ============================================================

def phase1_eda() -> None:
    """IS/OOS 기간 레짐 분포 + 레짐별 시장 특성 분석."""
    print("=" * 70)
    print("  Phase 1 EDA — IS/OOS 레짐 분포 분석")
    print("=" * 70)

    # 시장 데이터 로드
    market_path = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily.parquet"
    if not market_path.exists():
        print(f"  [!] 데이터 없음: {market_path}")
        return

    market = pd.read_parquet(market_path)
    market.index = pd.to_datetime(market.index).tz_localize(None)

    # SPY 종가 + 등락률
    spy_close = market[("SPY", "Close")].dropna()
    spy_pct   = spy_close.pct_change() * 100

    # SMA20 계산
    spy_sma20 = spy_close.rolling(20).mean()

    def classify_regime(spy_streak: int, spy_down_streak: int,
                        spy_c: float, sma: float,
                        bull_streak: int = 3, bear_streak: int = 2,
                        sma_bull_pct: float = 1.0, sma_bear_pct: float = 1.0) -> str:
        streak_regime = "neutral"
        if spy_streak >= bull_streak:
            streak_regime = "bull"
        elif spy_down_streak >= bear_streak:
            streak_regime = "bear"

        sma_regime = "neutral"
        if not np.isnan(sma):
            if spy_c > sma * (1 + sma_bull_pct / 100):
                sma_regime = "bull"
            elif spy_c < sma * (1 - sma_bear_pct / 100):
                sma_regime = "bear"

        if streak_regime == "bull" and sma_regime != "bear":
            return "bull"
        if streak_regime == "bear" and sma_regime != "bull":
            return "bear"
        if sma_regime != "neutral":
            return sma_regime
        return "neutral"

    # 레짐 시계열 구성
    regimes = []
    spy_streak_cnt = 0
    spy_down_streak_cnt = 0

    for dt, pct in spy_pct.items():
        if pd.isna(pct):
            regimes.append(("neutral", 0, 0))
            continue
        sc = spy_close.get(dt, np.nan)
        sma = spy_sma20.get(dt, np.nan)
        r = classify_regime(spy_streak_cnt, spy_down_streak_cnt, sc, sma)
        regimes.append((r, spy_streak_cnt, spy_down_streak_cnt))
        if pct > 0:
            spy_streak_cnt += 1
            spy_down_streak_cnt = 0
        else:
            spy_streak_cnt = 0
            spy_down_streak_cnt += 1

    regime_df = pd.DataFrame(regimes, index=spy_pct.index,
                              columns=["regime", "spy_streak", "spy_down_streak"])

    print("\n  [1-1] 전체 기간 레짐 분포")
    total = len(regime_df)
    for r in ["bull", "neutral", "bear"]:
        n = (regime_df["regime"] == r).sum()
        print(f"    {r:7s}: {n:4d}일 ({n/total*100:.0f}%)")

    for period_name, start, end in [
        ("IS  (2025-03~09)", IS_START, IS_END),
        ("OOS (2025-10~02)", OOS_START, OOS_END),
    ]:
        sub = regime_df[
            (regime_df.index >= pd.Timestamp(start))
            & (regime_df.index <= pd.Timestamp(end))
        ]
        n_total = len(sub)
        print(f"\n  [1-2] {period_name} 레짐 분포 (총 {n_total}일)")
        for r in ["bull", "neutral", "bear"]:
            n = (sub["regime"] == r).sum()
            avg_spy = spy_pct[sub[sub["regime"] == r].index].mean()
            print(f"    {r:7s}: {n:4d}일 ({n/max(n_total,1)*100:.0f}%)  "
                  f"평균 SPY: {avg_spy:+.2f}%")

    # 레짐별 수익률 패턴
    print("\n  [1-3] 레짐별 D2S 매수 대상 종목 등락률")
    tickers = ["ROBN", "CONL", "MSTU", "AMDL"]
    for ticker in tickers:
        try:
            t_pct = market[(ticker, "Close")].pct_change() * 100
            for r in ["bull", "bear"]:
                days = regime_df[regime_df["regime"] == r].index
                avg = t_pct[t_pct.index.isin(days)].mean()
                print(f"    {ticker} [{r:7s}]: 평균 {avg:+.2f}%/일")
        except KeyError:
            continue

    print("\n  [1-4] OOS 기간 특성 (왜 하락장인가?)")
    oos_spy = spy_pct[
        (spy_pct.index >= pd.Timestamp(OOS_START))
        & (spy_pct.index <= pd.Timestamp(OOS_END))
    ]
    print(f"    SPY 평균 등락률: {oos_spy.mean():+.3f}%/일")
    print(f"    SPY 양봉 비율:   {(oos_spy > 0).mean()*100:.1f}%")
    print(f"    SPY 음봉 비율:   {(oos_spy < 0).mean()*100:.1f}%")
    print(f"    SPY -1%↓ 일수:  {(oos_spy < -1).sum()}일")

    # 레짐 전환 포인트
    changes = regime_df["regime"] != regime_df["regime"].shift(1)
    transitions = regime_df[changes][["regime"]].copy()
    transitions = transitions[
        (transitions.index >= pd.Timestamp(IS_START))
        & (transitions.index <= pd.Timestamp(OOS_END))
    ]
    print(f"\n  [1-5] 레짐 전환 포인트 ({len(transitions)}회)")
    for dt, row in list(transitions.iterrows())[:15]:
        print(f"    {dt.date()}  → {row['regime']}")

    print("\n  Phase 1 완료")


# ============================================================
# Phase 2: Grid — 레짐 임계값 × 청산 파라미터 탐색
# ============================================================

def _run_grid_combo(args: tuple) -> dict:
    """멀티프로세싱 워커: 단일 파라미터 조합 백테스트."""
    (combo_id, params, start, end) = args
    try:
        bt = D2SBacktestV3(params=params, start_date=start, end_date=end, use_fees=True)
        bt.run(verbose=False)
        r = bt.report()
        return {
            "combo_id": combo_id,
            "total_return_pct": r.get("total_return_pct", 0),
            "win_rate": r.get("win_rate", 0),
            "mdd_pct": r.get("mdd_pct", 0),
            "sharpe_ratio": r.get("sharpe_ratio", 0),
            "sell_trades": r.get("sell_trades", 0),
        }
    except Exception as e:
        return {"combo_id": combo_id, "error": str(e),
                "total_return_pct": -999, "win_rate": 0,
                "mdd_pct": 0, "sharpe_ratio": -99, "sell_trades": 0}


def phase2_grid(n_jobs: int = 4) -> pd.DataFrame:
    """레짐 임계값 × 청산 파라미터 2D 그리드 탐색.

    탐색 축:
      - bull_spy_streak:   2, 3, 4  (Bull 판정 연속 상승일 수)
      - bear_spy_streak:   1, 2, 3  (Bear 판정 연속 하락일 수)
      - sma_bull_pct:      0.5, 1.0, 1.5  (SMA 강도)
      - bull_take_profit:  5.9, 7.0  (Bull TP)
      - bull_hold_days:    7, 10     (Bull HD)
      - bear_take_profit:  4.5, 5.0, 5.5  (Bear TP)
      - bear_hold_days:    3, 4, 5   (Bear HD)
    """
    print("=" * 70)
    print("  Phase 2 Grid — 레짐 임계값 × 청산 파라미터 탐색")
    print("=" * 70)

    # 탐색 축 정의
    bull_streak_options  = [2, 3, 4]
    bear_streak_options  = [1, 2, 3]
    sma_bull_pcts        = [0.5, 1.0, 1.5]
    bull_tp_options      = [5.9, 7.0]
    bull_hd_options      = [7, 10]
    bear_tp_options      = [4.5, 5.0, 5.5]
    bear_hd_options      = [3, 4, 5]

    # 기준선: v2 고정 파라미터 (레짐 없음)
    baselines = [
        ("baseline_v2",   D2S_ENGINE_V2, IS_START, IS_END),
        ("baseline_v2",   D2S_ENGINE_V2, OOS_START, OOS_END),
        # Study H 최적 (비레짐)
        ("hd4_tp50",      {**D2S_ENGINE_V2,
                           "take_profit_pct": 5.0,
                           "optimal_hold_days_max": 4}, IS_START, IS_END),
        ("hd4_tp50",      {**D2S_ENGINE_V2,
                           "take_profit_pct": 5.0,
                           "optimal_hold_days_max": 4}, OOS_START, OOS_END),
    ]

    # 그리드 조합 생성
    combos = []
    combo_id = 0
    for b_str in bull_streak_options:
        for d_str in bear_streak_options:
            for sma_pct in sma_bull_pcts:
                for b_tp in bull_tp_options:
                    for b_hd in bull_hd_options:
                        for d_tp in bear_tp_options:
                            for d_hd in bear_hd_options:
                                p = deepcopy(D2S_ENGINE_V3)
                                p.update({
                                    "regime_bull_spy_streak": b_str,
                                    "regime_bear_spy_streak": d_str,
                                    "regime_spy_sma_bull_pct": sma_pct,
                                    "regime_spy_sma_bear_pct": -sma_pct,
                                    "bull_take_profit_pct": b_tp,
                                    "bull_hold_days_max": b_hd,
                                    "bear_take_profit_pct": d_tp,
                                    "bear_hold_days_max": d_hd,
                                    # R19 비활성화 (레짐 효과만 측정)
                                    "bb_entry_hard_filter": False,
                                })
                                label = (f"bs{b_str}_ds{d_str}_sma{sma_pct:.0f}_"
                                         f"btp{b_tp}_bhd{b_hd}_dtp{d_tp}_dhd{d_hd}")
                                for start, end, period in [
                                    (IS_START, IS_END, "IS"),
                                    (OOS_START, OOS_END, "OOS"),
                                ]:
                                    combos.append((
                                        combo_id,
                                        label,
                                        b_str, d_str, sma_pct,
                                        b_tp, b_hd, d_tp, d_hd,
                                        period,
                                        deepcopy(p), start, end,
                                    ))
                                combo_id += 1

    total_combos = combo_id
    total_runs = len(combos)
    print(f"  조합 수: {total_combos}개 × IS/OOS = {total_runs}회")
    print(f"  병렬 워커: {n_jobs}")

    # 기준선 실행
    print("\n  기준선 실행...")
    baseline_results = []
    for label, params, start, end in baselines:
        period = "IS" if end == IS_END else "OOS"
        bt = D2SBacktestV2(params=params, start_date=start, end_date=end, use_fees=True)
        bt.run(verbose=False)
        r = bt.report()
        baseline_results.append({
            "label": label,
            "period": period,
            "total_return_pct": r.get("total_return_pct", 0),
            "win_rate": r.get("win_rate", 0),
            "mdd_pct": r.get("mdd_pct", 0),
            "sharpe_ratio": r.get("sharpe_ratio", 0),
            "sell_trades": r.get("sell_trades", 0),
        })
        print(f"  {label:20s} [{period}] 수익률: {r.get('total_return_pct', 0):+.2f}%  "
              f"MDD: {r.get('mdd_pct', 0):.1f}%  Sharpe: {r.get('sharpe_ratio', 0):.3f}")

    # 그리드 실행
    print(f"\n  그리드 실행 중... ({total_runs}회)")
    worker_args = [
        (c[0], c[10], c[11], c[12]) for c in combos
    ]

    if n_jobs <= 1:
        raw_results = [_run_grid_combo(a) for a in worker_args]
    else:
        with Pool(processes=n_jobs) as pool:
            raw_results = pool.map(_run_grid_combo, worker_args)

    # 결과 조합
    rows = []
    for combo, raw in zip(combos, raw_results):
        rows.append({
            "combo_id": combo[0],
            "label": combo[1],
            "bull_spy_streak": combo[2],
            "bear_spy_streak": combo[3],
            "sma_bull_pct": combo[4],
            "bull_tp": combo[5],
            "bull_hd": combo[6],
            "bear_tp": combo[7],
            "bear_hd": combo[8],
            "period": combo[9],
            **{k: v for k, v in raw.items() if k != "combo_id"},
        })

    df = pd.DataFrame(rows)

    # IS/OOS 피벗
    is_df  = df[df["period"] == "IS"].set_index("combo_id")
    oos_df = df[df["period"] == "OOS"].set_index("combo_id")

    merged = is_df[[
        "label", "bull_spy_streak", "bear_spy_streak", "sma_bull_pct",
        "bull_tp", "bull_hd", "bear_tp", "bear_hd",
        "total_return_pct", "mdd_pct", "sharpe_ratio", "sell_trades"
    ]].copy()
    merged.columns = (
        ["label", "bull_streak", "bear_streak", "sma_pct",
         "bull_tp", "bull_hd", "bear_tp", "bear_hd",
         "IS_ret", "IS_mdd", "IS_sharpe", "IS_trades"]
    )
    merged["OOS_ret"] = oos_df["total_return_pct"]
    merged["OOS_mdd"] = oos_df["mdd_pct"]
    merged["OOS_sharpe"] = oos_df["sharpe_ratio"]
    merged["FULL_score"] = merged["IS_sharpe"] * 0.5 + merged["OOS_sharpe"] * 0.5

    # 기준선 참조값
    bl_v2_is  = next((r["total_return_pct"] for r in baseline_results
                      if r["label"] == "baseline_v2" and r["period"] == "IS"), 0)
    bl_v2_oos = next((r["total_return_pct"] for r in baseline_results
                      if r["label"] == "baseline_v2" and r["period"] == "OOS"), 0)

    print(f"\n  기준선 v2: IS={bl_v2_is:+.2f}%  OOS={bl_v2_oos:+.2f}%")

    # 상위 10개 (FULL_score 기준)
    top10 = merged.nlargest(10, "FULL_score")
    print(f"\n  Top 10 (IS/OOS Sharpe 평균 기준):")
    print(f"  {'#':>3}  {'IS%':>7}  {'OOS%':>7}  {'IS_Shp':>6}  {'OOS_Shp':>7}  "
          f"{'bStr':>4}  {'dStr':>4}  {'bTP':>5}  {'bHD':>4}  {'dTP':>5}  {'dHD':>4}")
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        print(
            f"  {idx:3d}  "
            f"{row['IS_ret']:+6.2f}%  {row['OOS_ret']:+6.2f}%  "
            f"{row['IS_sharpe']:+5.3f}  {row['OOS_sharpe']:+6.3f}  "
            f"{int(row['bull_streak']):4d}  {int(row['bear_streak']):4d}  "
            f"{row['bull_tp']:5.1f}  {int(row['bull_hd']):4d}  "
            f"{row['bear_tp']:5.1f}  {int(row['bear_hd']):4d}"
        )

    # OOS 양수 + IS 기준 이상인 조합
    print(f"\n  OOS 양수 & IS ≥ {bl_v2_is * 0.7:.1f}% 조합:")
    good = merged[
        (merged["OOS_ret"] > 0)
        & (merged["IS_ret"] >= bl_v2_is * 0.7)
    ].sort_values("OOS_ret", ascending=False)
    if len(good) > 0:
        print(f"  → {len(good)}개 발견")
        for _, row in good.head(5).iterrows():
            print(f"    IS={row['IS_ret']:+.2f}% OOS={row['OOS_ret']:+.2f}%  "
                  f"bStr={int(row['bull_streak'])} dStr={int(row['bear_streak'])}  "
                  f"bTP={row['bull_tp']} bHD={int(row['bull_hd'])}  "
                  f"dTP={row['bear_tp']} dHD={int(row['bear_hd'])}")
    else:
        print("  → 없음 (조건 완화 필요)")

    # 저장
    out_path = RESULTS_DIR / "study_5_regime_grid.csv"
    merged.to_csv(out_path)
    print(f"\n  그리드 결과 저장: {out_path}")

    print("\n  Phase 2 완료")
    return merged


# ============================================================
# Phase 3: 최적 조합 IS/OOS + R19 시너지
# ============================================================

def phase3_best_combo(grid_df: pd.DataFrame | None = None) -> None:
    """최적 조합 전체 검증 + R19 시너지."""
    print("=" * 70)
    print("  Phase 3 — 최적 조합 검증 + R19 시너지")
    print("=" * 70)

    # 그리드 결과 로드 or 인자 사용
    if grid_df is None:
        csv_path = RESULTS_DIR / "study_5_regime_grid.csv"
        if not csv_path.exists():
            print("  [!] Phase 2 결과 없음. Phase 2 먼저 실행하세요.")
            return
        grid_df = pd.read_csv(csv_path, index_col=0)

    best = grid_df.nlargest(1, "FULL_score").iloc[0]
    print(f"\n  Best 조합:")
    print(f"    bull_streak={int(best['bull_streak'])}  bear_streak={int(best['bear_streak'])}")
    print(f"    bull_tp={best['bull_tp']}%  bull_hd={int(best['bull_hd'])}일")
    print(f"    bear_tp={best['bear_tp']}%  bear_hd={int(best['bear_hd'])}일")
    print(f"    IS={best['IS_ret']:+.2f}%  OOS={best['OOS_ret']:+.2f}%  "
          f"IS_Shp={best['IS_sharpe']:.3f}  OOS_Shp={best['OOS_sharpe']:.3f}")

    best_params = deepcopy(D2S_ENGINE_V3)
    best_params.update({
        "regime_bull_spy_streak": int(best["bull_streak"]),
        "regime_bear_spy_streak": int(best["bear_streak"]),
        "bull_take_profit_pct": best["bull_tp"],
        "bull_hold_days_max": int(best["bull_hd"]),
        "bear_take_profit_pct": best["bear_tp"],
        "bear_hold_days_max": int(best["bear_hd"]),
    })

    scenarios = [
        ("v2 기준선",          D2S_ENGINE_V2, False),
        ("v3 레짐 (R19 OFF)",  {**best_params, "bb_entry_hard_filter": False}, False),
        ("v3 레짐 (R19 ON)",   {**best_params, "bb_entry_hard_filter": True},  True),
    ]

    print(f"\n  비교표 (IS / OOS / FULL):")
    hdr = f"  {'시나리오':25s}  {'IS':>8}  {'OOS':>8}  {'FULL':>8}  {'IS_MDD':>7}  {'Shp_IS':>7}  {'Shp_OOS':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, params, is_v3 in scenarios:
        cls = D2SBacktestV3 if is_v3 else (
            D2SBacktestV3 if "regime_enabled" in params else D2SBacktestV2
        )
        results = {}
        for period, start, end in [
            ("IS",   IS_START,   IS_END),
            ("OOS",  OOS_START,  OOS_END),
            ("FULL", FULL_START, FULL_END),
        ]:
            bt = cls(params=params, start_date=start, end_date=end, use_fees=True)
            bt.run(verbose=False)
            results[period] = bt.report()

        r_is   = results["IS"]
        r_oos  = results["OOS"]
        r_full = results["FULL"]
        print(
            f"  {label:25s}  "
            f"{r_is['total_return_pct']:+7.2f}%  "
            f"{r_oos['total_return_pct']:+7.2f}%  "
            f"{r_full['total_return_pct']:+7.2f}%  "
            f"{r_is['mdd_pct']:6.1f}%  "
            f"{r_is['sharpe_ratio']:+6.3f}  "
            f"{r_oos['sharpe_ratio']:+7.3f}"
        )

    print("\n  Phase 3 완료")
    print("\n  ★ v3 후보 규칙 요약:")
    print(f"    [R19] BB 진입 하드 필터: %B ≤ {best_params['bb_entry_hard_max']} (Study G F2)")
    print(f"    [R20] 레짐 조건부 TP: Bull={best['bull_tp']}%  Bear={best['bear_tp']}%")
    print(f"    [R21] 레짐 조건부 HD: Bull={int(best['bull_hd'])}일  Bear={int(best['bear_hd'])}일")
    print(f"    [레짐 감지] bull_streak≥{int(best['bull_streak'])}  bear_streak≥{int(best['bear_streak'])}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Study 5 — 레짐 감지 스터디")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="실행 단계 (1: EDA, 2: Grid, 3: 최적 검증)")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Phase 2 병렬 프로세스 수 (기본: 4)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Study 5 — 레짐 감지 스터디")
    print("  IS: 2025-03-03 ~ 2025-09-30  |  OOS: 2025-10-01 ~ 2026-02-17")
    print("=" * 70)

    if args.phase == 1:
        phase1_eda()
    elif args.phase == 2:
        phase2_grid(n_jobs=args.n_jobs)
    elif args.phase == 3:
        phase3_best_combo()
    else:
        # 전체 실행
        phase1_eda()
        print()
        grid_df = phase2_grid(n_jobs=args.n_jobs)
        print()
        phase3_best_combo(grid_df)

    print("\n  Study 5 완료!")


if __name__ == "__main__":
    main()

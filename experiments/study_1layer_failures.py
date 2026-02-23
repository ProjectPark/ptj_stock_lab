#!/usr/bin/env python3
"""
축 1 스터디: 1레이어 실패 패턴 분석
======================================
DCA 1레이어 진입의 강제청산(실패) 케이스 공통 조건을 찾아
v2 진입 필터 강화 가능성을 평가한다.

핵심 질문:
  - 1레이어 실패 7건의 공통 조건은 무엇인가?
  - 성공(43건) vs 실패(7건)을 가르는 진입 필터는?
  - 새로운 필터 적용 시 진입 건수 감소 대비 실패 감소 효과는?

가설 검증:
  [H1] score < 0.70 진입은 실패율이 높다
  [H2] bb_at_entry >= 0.20 이면 강제청산 위험이 높다
  [H3] GLD 상승일 갭 진입은 취약하다
  [H4] R14(리스크오프) 없는 갭 진입 = 가장 약한 조합

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_1layer_failures.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ============================================================
# 경로
# ============================================================
TRADES_CSV   = _ROOT / "data" / "results" / "backtests" / "d2s_trades.csv"
DAILY_PARQ   = _ROOT / "data" / "market" / "daily" / "market_daily.parquet"
OUT_CSV      = _ROOT / "data" / "results" / "analysis" / "1layer_failure_eda.csv"


# ============================================================
# 데이터 준비
# ============================================================

def load_market_context() -> pd.DataFrame:
    """SPY / GLD 일별 등락률 로드."""
    daily = pd.read_parquet(DAILY_PARQ)
    spy_pct = daily[("SPY", "Close")].pct_change() * 100
    gld_pct = daily[("GLD", "Close")].pct_change() * 100
    ctx = pd.DataFrame({"spy_pct": spy_pct, "gld_pct": gld_pct})
    ctx.index = ctx.index.date
    return ctx


def build_positions(trades_csv: Path, market_ctx: pd.DataFrame) -> pd.DataFrame:
    """d2s_trades.csv → 포지션 단위 DataFrame (시황 컨텍스트 포함)."""
    df = pd.read_csv(trades_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    positions = []
    pos_buys: dict[str, list] = {}

    for _, row in df.sort_values("date").iterrows():
        t = row["ticker"]
        if row["side"] == "BUY":
            pos_buys.setdefault(t, []).append(row)
        elif row["side"] == "SELL":
            if t in pos_buys and pos_buys[t]:
                bl = pos_buys[t]
                entry_date = bl[0]["date"]

                # reason 파싱
                reason = str(bl[0]["reason"])
                bb_m    = re.search(r"R8:bb_low\(([-\d.]+)\)", reason)
                crash_m = re.search(r"contrarian:([-\d.]+)%", reason)
                gap_m   = re.search(r"gap:(\w+)\(([-+\d.]+)%\)", reason)

                # 진입일 시황
                ctx = market_ctx.loc[entry_date] if entry_date in market_ctx.index else None

                positions.append({
                    "ticker":        t,
                    "entry_date":    entry_date,
                    "exit_date":     row["date"],
                    "hold_days":     (row["date"] - entry_date).days,
                    "dca_count":     len(bl),
                    "score":         bl[0]["score"],
                    "pnl":           row["pnl"],
                    "pnl_pct":       row["pnl_pct"],
                    "exit_type":     "forced" if "hold_days" in str(row["reason"]) else "take_profit",
                    "reason":        reason,
                    # 신호 플래그
                    "has_r14":       "R14:riskoff" in reason,
                    "has_r16":       "R16:atr"     in reason,
                    "has_r6":        "R6:spy"      in reason,
                    "has_r9":        "R9:combo"    in reason,
                    "has_r15":       "R15:friday"  in reason,
                    "is_gap":        reason.strip().startswith("gap:"),
                    # 수치 특성
                    "bb_at_entry":   float(bb_m.group(1))    if bb_m    else None,
                    "crash_pct":     float(crash_m.group(1)) if crash_m else None,
                    "gap_pct":       float(gap_m.group(2))   if gap_m   else None,
                    # 시황
                    "spy_pct":       float(ctx["spy_pct"]) if ctx is not None else None,
                    "gld_pct":       float(ctx["gld_pct"]) if ctx is not None else None,
                })
                pos_buys[t] = []

    return pd.DataFrame(positions)


# ============================================================
# 필터 시뮬레이션 — 특정 조건으로 진입을 막을 때 효과 측정
# ============================================================

def simulate_filter(one: pd.DataFrame, mask_blocked: pd.Series, label: str) -> dict:
    """1레이어 진입에서 mask_blocked에 해당하는 진입을 차단했을 때 효과.

    Parameters
    ----------
    one           : 1레이어 전체 포지션
    mask_blocked  : True = 이 조건에서 진입 차단
    label         : 필터 이름

    Returns
    -------
    dict: blocked_n, blocked_fail, blocked_succ, remain_fail, remain_succ, ...
    """
    blocked = one[mask_blocked]
    remain  = one[~mask_blocked]

    return {
        "label":          label,
        "총 진입":        len(one),
        "차단 건수":       len(blocked),
        "차단 중 실패":    (blocked["exit_type"] == "forced").sum(),
        "차단 중 성공":    (blocked["exit_type"] == "take_profit").sum(),
        "잔여 진입":       len(remain),
        "잔여 실패":       (remain["exit_type"] == "forced").sum(),
        "잔여 승률":       f"{(remain['pnl'] > 0).mean()*100:.0f}%" if len(remain) > 0 else "-",
        "잔여 평균PnL":    f"${remain['pnl'].mean():+.0f}" if len(remain) > 0 else "-",
        "효과 (실패 감소)": (blocked["exit_type"] == "forced").sum(),
        "비용 (성공 포기)": (blocked["exit_type"] == "take_profit").sum(),
    }


# ============================================================
# 메인
# ============================================================

def main():
    print("[1/3] 데이터 로드...")
    market_ctx = load_market_context()
    pos_df     = build_positions(TRADES_CSV, market_ctx)

    one  = pos_df[pos_df["dca_count"] == 1].copy()
    fail = one[one["exit_type"] == "forced"]
    succ = one[one["exit_type"] == "take_profit"]

    print(f"  1레이어 포지션: {len(one)}건  성공: {len(succ)}건  실패: {len(fail)}건")

    print("\n[2/3] 성공 vs 실패 특성 비교...")

    # ============================================================
    # 섹션 1: 실패 케이스 프로파일
    # ============================================================
    print("\n" + "=" * 65)
    print("  [1] 1레이어 실패 케이스 프로파일")
    print("=" * 65)

    print(f"\n  {'특성':20s}  {'성공(43건)':>12s}  {'실패(7건)':>10s}  {'차이'}")
    print(f"  {'-'*55}")

    compare_num = [
        ("score",        "진입 점수"),
        ("bb_at_entry",  "BB %B"),
        ("crash_pct",    "종목 하락폭(%)"),
        ("spy_pct",      "진입일 SPY(%)"),
        ("gld_pct",      "진입일 GLD(%)"),
    ]
    for col, label in compare_num:
        sv = succ[col].mean()
        fv = fail[col].mean()
        diff = fv - sv
        flag = " ◀ 유의" if abs(diff) > 0.3 else ""
        print(f"  {label:20s}  {sv:>+10.2f}   {fv:>+8.2f}  {diff:+.2f}{flag}")

    print()
    compare_bool = [
        ("has_r14", "R14 리스크오프"),
        ("has_r16", "R16 ATR 고변동"),
        ("has_r6",  "R6 SPY 약세"),
        ("has_r9",  "R9 MACD 콤보"),
        ("is_gap",  "갭 진입"),
    ]
    for col, label in compare_bool:
        sv = succ[col].mean() * 100
        fv = fail[col].mean() * 100
        diff = fv - sv
        flag = " ◀ 유의" if abs(diff) > 15 else ""
        print(f"  {label:20s}  {sv:>+10.0f}%  {fv:>+8.0f}%  {diff:+.0f}%p{flag}")

    # ============================================================
    # 섹션 2: 핵심 발견
    # ============================================================
    print("\n" + "=" * 65)
    print("  [2] 핵심 발견 — bb_at_entry 구간별 실패율")
    print("=" * 65)
    bins = [(-99, 0.0, "BB < 0"),
            (0.0,  0.2, "BB 0.0~0.2"),
            (0.2,  0.4, "BB 0.2~0.4"),
            (0.4,  0.6, "BB 0.4~0.6"),
            (0.6,  9.9, "BB >= 0.6")]
    for lo, hi, label in bins:
        sub = one[one["bb_at_entry"].between(lo, hi, inclusive="left")]
        if len(sub) == 0:
            continue
        fc  = (sub["exit_type"] == "forced").sum()
        avg = sub["pnl"].mean()
        wr  = (sub["pnl"] > 0).mean() * 100
        star = " ★ 최적 구간" if fc == 0 and len(sub) >= 5 else ""
        print(f"  {label:12s}: {len(sub):3d}건  강제청산 {fc}건 ({fc/len(sub)*100:.0f}%)  "
              f"승률 {wr:.0f}%  평균 PnL ${avg:+.0f}{star}")

    print()
    print("  score 구간별 실패율")
    score_bins = [(0.0, 0.7, "score < 0.70"),
                  (0.7, 0.8, "score 0.70~0.80"),
                  (0.8, 0.9, "score 0.80~0.90"),
                  (0.9, 1.1, "score >= 0.90")]
    for lo, hi, label in score_bins:
        sub = one[one["score"].between(lo, hi, inclusive="left")]
        if len(sub) == 0:
            continue
        fc  = (sub["exit_type"] == "forced").sum()
        avg = sub["pnl"].mean()
        wr  = (sub["pnl"] > 0).mean() * 100
        print(f"  {label:18s}: {len(sub):3d}건  강제청산 {fc}건 ({fc/len(sub)*100:.0f}%)  "
              f"승률 {wr:.0f}%  평균 PnL ${avg:+.0f}")

    # ============================================================
    # 섹션 3: 가설 검증 — 필터 시뮬레이션
    # ============================================================
    print("\n" + "=" * 65)
    print("  [3] 가설 검증 — 필터별 효과 시뮬레이션")
    print("=" * 65)
    print(f"  ※ '차단 건수'는 막히는 진입, '효과'는 줄어드는 실패, '비용'은 포기하는 성공")
    print()

    filters = [
        # (label, mask_blocked)
        ("H1: score < 0.70 차단",
         one["score"] < 0.70),

        ("H2: bb >= 0.20 차단",
         one["bb_at_entry"].fillna(1.0) >= 0.20),

        ("H3: GLD > 0 + 갭 진입 차단",
         one["is_gap"] & one["gld_pct"].fillna(0).gt(0)),

        ("H4: R14 없는 갭 차단",
         one["is_gap"] & ~one["has_r14"]),

        ("H1+H2 복합: score<0.70 OR bb>=0.20",
         (one["score"] < 0.70) | (one["bb_at_entry"].fillna(1.0) >= 0.20)),

        ("H1+H2 교집합: score<0.70 AND bb>=0.20",
         (one["score"] < 0.70) & (one["bb_at_entry"].fillna(1.0) >= 0.20)),
    ]

    results = []
    for label, mask in filters:
        r = simulate_filter(one, mask, label)
        results.append(r)
        print(f"  [{label}]")
        print(f"    차단: {r['차단 건수']}건  (실패 {r['차단 중 실패']}건 포함 / 성공 {r['차단 중 성공']}건 포기)")
        print(f"    잔여: {r['잔여 진입']}건  실패 {r['잔여 실패']}건  승률 {r['잔여 승률']}  평균 PnL {r['잔여 평균PnL']}")
        ratio = r['효과 (실패 감소)'] / r['비용 (성공 포기)'] if r['비용 (성공 포기)'] > 0 else float('inf')
        print(f"    효과/비용 비율: {ratio:.1f}  (높을수록 좋음)")
        print()

    # ============================================================
    # 섹션 4: 실패 케이스 상세
    # ============================================================
    print("=" * 65)
    print("  [4] 실패 케이스 상세")
    print("=" * 65)
    cols = ["ticker", "entry_date", "score", "bb_at_entry",
            "spy_pct", "gld_pct", "crash_pct", "has_r14", "is_gap", "pnl_pct"]
    print(fail[cols].to_string(index=False))

    # ============================================================
    # 섹션 5: 최적 필터 도출
    # ============================================================
    print("\n" + "=" * 65)
    print("  [5] 권고 필터 — 효과/비용 비율 기준")
    print("=" * 65)

    best = max(results, key=lambda r: (
        r["효과 (실패 감소)"] / max(r["비용 (성공 포기)"], 0.1)
    ))
    print(f"  최적 단일 필터: [{best['label']}]")
    print(f"    실패 {best['효과 (실패 감소)']}건 제거  /  성공 {best['비용 (성공 포기)']}건 포기")
    print(f"    잔여 승률: {best['잔여 승률']}  평균 PnL: {best['잔여 평균PnL']}")

    # 권고 파라미터 값
    print()
    print("  권고 파라미터 (v2 강화안):")

    # score 최적 임계값
    for thresh in [0.65, 0.70, 0.75]:
        sub = one[one["score"] >= thresh]
        fc = (sub["exit_type"] == "forced").sum()
        wr = (sub["pnl"] > 0).mean() * 100
        print(f"    score_min = {thresh:.2f} → {len(sub)}건 잔여, "
              f"강제청산 {fc}건, 승률 {wr:.0f}%")

    print()
    # bb 최적 임계값
    for thresh in [0.20, 0.25, 0.30]:
        sub = one[one["bb_at_entry"].fillna(1.0) < thresh]
        fc = (sub["exit_type"] == "forced").sum()
        wr = (sub["pnl"] > 0).mean() * 100
        print(f"    bb_entry_max = {thresh:.2f} → {len(sub)}건 잔여, "
              f"강제청산 {fc}건, 승률 {wr:.0f}%")

    print("=" * 65)

    # 저장
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    one.to_csv(OUT_CSV, index=False)
    print(f"\n  상세 데이터: {OUT_CSV}")


if __name__ == "__main__":
    main()

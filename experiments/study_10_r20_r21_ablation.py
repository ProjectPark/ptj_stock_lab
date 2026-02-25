#!/usr/bin/env python3
"""
Study 10 — R20(TP 조건부) vs R21(HD 조건부) 단독 Ablation
===========================================================
목적: Study B에서 no_regime이 full_3signal보다 좋았던 원인 특정.
     R20과 R21 중 어느 것이 역효과인지 분리 실험.

실험 설계 (no-ROBN 1.5년 전체 기간):
  A. full_regime : Bull(tp=5.0%, hd=12d) / Bear+Neutral(tp=6.5%, hd=8d)  ← 현재 상태
  B. tp_only     : TP 조건부 유지 / HD 8일 고정 (R21 제거)
  C. hd_only     : TP 5.0% 고정 / HD 조건부 유지 (R20 제거)
  D. no_regime   : TP 5.0% 고정 / HD 8일 고정 (Study B best)
  E. no_regime_6 : TP 6.0% 고정 / HD 10일 고정 (균일 대안)

각 실험에서 레짐 감지(streak+SMA+Poly)는 모두 동일하게 작동.
R20/R21 청산 파라미터값만 변경.

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_10_r20_r21_ablation.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN
from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 기간 설정 (no-ROBN 1.5년) ─────────────────────────────────
START = date(2024, 9, 18)
END   = date(2026, 2, 17)

# ── 실험 설계 ─────────────────────────────────────────────────
EXPERIMENTS = {
    "A_full_regime": {
        "desc": "현재 상태 (Bull tp=5.0%/hd=12d, Bear tp=6.5%/hd=8d)",
        "params": {
            "bull_take_profit_pct": 5.0,
            "bear_take_profit_pct": 6.5,
            "bull_hold_days_max":   12,
            "bear_hold_days_max":   8,
        },
    },
    "B_tp_only": {
        "desc": "R20(TP 조건부) 유지 / R21 제거 (HD=8일 고정)",
        "params": {
            "bull_take_profit_pct": 5.0,
            "bear_take_profit_pct": 6.5,
            "bull_hold_days_max":   8,   # Bear와 동일 고정
            "bear_hold_days_max":   8,
        },
    },
    "C_hd_only": {
        "desc": "R21(HD 조건부) 유지 / R20 제거 (TP=5.0% 고정)",
        "params": {
            "bull_take_profit_pct": 5.0,
            "bear_take_profit_pct": 5.0,  # Bull과 동일 고정
            "bull_hold_days_max":   12,
            "bear_hold_days_max":   8,
        },
    },
    "D_no_regime": {
        "desc": "R20+R21 모두 제거 (TP=5.0% / HD=8일 고정) ← Study B best",
        "params": {
            "bull_take_profit_pct": 5.0,
            "bear_take_profit_pct": 5.0,
            "bull_hold_days_max":   8,
            "bear_hold_days_max":   8,
        },
    },
    "E_no_regime_alt": {
        "desc": "R20+R21 제거 대안 (TP=6.0% / HD=10일 고정)",
        "params": {
            "bull_take_profit_pct": 6.0,
            "bear_take_profit_pct": 6.0,
            "bull_hold_days_max":   10,
            "bear_hold_days_max":   10,
        },
    },
}


def run_experiment(name: str, desc: str, param_overrides: dict) -> dict:
    params = {**D2S_ENGINE_V3_NO_ROBN, **param_overrides}
    bt = D2SBacktestV3(params=params, start_date=START, end_date=END)
    bt.run(verbose=False)
    r = bt.report()
    return {
        "name":       name,
        "desc":       desc,
        "overrides":  param_overrides,
        "return_pct": r.get("total_return_pct", 0),
        "mdd_pct":    r.get("mdd_pct", 0),
        "sharpe":     r.get("sharpe_ratio", 0),
        "win_rate":   r.get("win_rate", 0),
        "sell_trades": r.get("sell_trades", 0),
        "regime_days": bt._regime_days,
        "r20_r21_triggers": sum(bt._regime_applied.values()) if hasattr(bt, "_regime_applied") else None,
    }


def main():
    print("=" * 70)
    print("  Study 10 — R20 vs R21 Ablation (no-ROBN 1.5년)")
    print(f"  기간: {START} ~ {END}")
    print("=" * 70)

    results = {}
    for name, cfg in EXPERIMENTS.items():
        print(f"\n[{name}] {cfg['desc']}")
        r = run_experiment(name, cfg["desc"], cfg["params"])
        results[name] = r
        print(f"  수익률: {r['return_pct']:+.2f}%  MDD: {r['mdd_pct']:.2f}%  "
              f"Sharpe: {r['sharpe']:.3f}  승률: {r['win_rate']:.1f}%  "
              f"청산: {r['sell_trades']}건")
        rd = r.get("regime_days", {})
        if rd:
            total = sum(rd.values()) or 1
            print(f"  레짐: bull {rd.get('bull',0)}일({rd.get('bull',0)/total*100:.0f}%) "
                  f"bear {rd.get('bear',0)}일({rd.get('bear',0)/total*100:.0f}%) "
                  f"neutral {rd.get('neutral',0)}일({rd.get('neutral',0)/total*100:.0f}%)")

    # ── 결과 테이블 ─────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  결과 비교 테이블")
    print("=" * 70)
    header = f"  {'실험':<22} {'수익률':>8} {'MDD':>8} {'Sharpe':>8} {'승률':>7} {'청산':>5}"
    print(header)
    print("  " + "-" * 60)
    for name, r in results.items():
        mark = " ★" if name == "D_no_regime" else ""
        print(f"  {name:<22} {r['return_pct']:>7.2f}% {r['mdd_pct']:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['win_rate']:>6.1f}% {r['sell_trades']:>5}{mark}")

    # ── 결론 분석 ───────────────────────────────────────────────
    a = results["A_full_regime"]["return_pct"]
    b = results["B_tp_only"]["return_pct"]
    c = results["C_hd_only"]["return_pct"]
    d = results["D_no_regime"]["return_pct"]

    print("\n  원인 분석:")
    r20_effect = b - a   # R21 제거 시 변화 → R21의 영향
    r21_effect = c - a   # R20 제거 시 변화 → R20의 영향
    both_effect = d - a  # 둘 다 제거 시 변화

    print(f"  R21(HD) 제거 효과 (B-A): {r20_effect:+.2f}%p → "
          + ("HD 조건부가 손해" if r20_effect > 0 else "HD 조건부가 도움"))
    print(f"  R20(TP) 제거 효과 (C-A): {r21_effect:+.2f}%p → "
          + ("TP 조건부가 손해" if r21_effect > 0 else "TP 조건부가 도움"))
    print(f"  둘 다 제거 효과 (D-A):   {both_effect:+.2f}%p")

    if r20_effect > r21_effect:
        print("\n  → 주요 원인: R21(HD 조건부) — Bull hold_days=12일이 과도")
    elif r21_effect > r20_effect:
        print("\n  → 주요 원인: R20(TP 조건부) — Bull/Bear TP 차등이 역효과")
    else:
        print("\n  → R20/R21 둘 다 문제, 레짐 조건부 자체 재검토 필요")

    # ── 저장 ────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"study_10_r20_r21_ablation_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()

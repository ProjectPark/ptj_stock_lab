#!/usr/bin/env python3
"""
Study 8 — Polymarket 레짐 단독 기여도 분리 (Ablation)
=====================================================
목적: 레짐 감지 3개 신호 중 각각의 기여도를 ablation으로 측정

Ablation 4가지:
  full_3signal: 기본 v3 (streak + SMA + Poly 다수결)
  no_poly:      streak + SMA만 (Poly 신호 neutral 처리)
  poly_only:    Poly 신호만 (streak/SMA neutral 처리)
  no_regime:    레짐 감지 완전 비활성 (tp=5.0% / hd=8일 고정)
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = date(2024, 9, 18)
END_DATE   = date(2026, 2, 17)


class D2SBacktestAblation(D2SBacktestV3):
    """레짐 감지 ablation 버전 — ablation_mode 파라미터로 신호 선택."""

    def __init__(self, ablation_mode: str = "full", **kwargs):
        super().__init__(**kwargs)
        self.ablation_mode = ablation_mode  # full / no_poly / poly_only / no_regime

    def _detect_regime(self, spy_streak, spy_close, poly_btc_up=None):
        if self.ablation_mode == "no_regime":
            return "neutral"

        # 기존 _detect_regime 내부 로직을 ablation_mode에 따라 선택적 비활성화
        if not self.params.get("regime_enabled", True):
            return "neutral"

        bull_streak_th = self.params.get("regime_bull_spy_streak", 5)
        bear_streak_th = self.params.get("regime_bear_spy_streak", 1)

        # streak 신호
        streak_regime = "neutral"
        if self.ablation_mode in ("full", "no_poly"):
            if spy_streak >= bull_streak_th:
                streak_regime = "bull"
            elif self._spy_down_streak >= bear_streak_th:
                streak_regime = "bear"

        # SMA 신호
        sma_regime = "neutral"
        if self.ablation_mode in ("full", "no_poly"):
            if spy_close is not None and len(self._spy_closes) >= 5:
                sma = float(np.mean(self._spy_closes))
                bull_pct = self.params.get("regime_spy_sma_bull_pct", 1.1) / 100.0
                bear_pct = abs(self.params.get("regime_spy_sma_bear_pct", -1.5)) / 100.0
                if spy_close > sma * (1 + bull_pct):
                    sma_regime = "bull"
                elif spy_close < sma * (1 - bear_pct):
                    sma_regime = "bear"

        # Poly 신호
        poly_regime = "neutral"
        if self.ablation_mode in ("full", "poly_only"):
            if poly_btc_up is not None:
                btc_bull_th = self.params.get("regime_btc_bull_threshold", 0.55)
                btc_bear_th = self.params.get("regime_btc_bear_threshold", 0.35)
                if poly_btc_up > btc_bull_th:
                    poly_regime = "bull"
                elif poly_btc_up < btc_bear_th:
                    poly_regime = "bear"

        # 다수결 (poly_only는 단독 신호만)
        if self.ablation_mode == "poly_only":
            return poly_regime

        signals = [streak_regime, sma_regime, poly_regime]
        bull_cnt = signals.count("bull")
        bear_cnt = signals.count("bear")
        if bull_cnt >= 2:
            return "bull"
        if bear_cnt >= 2:
            return "bear"
        if streak_regime != "neutral":
            return streak_regime
        if sma_regime != "neutral":
            return sma_regime
        return "neutral"


def run_ablation(label, mode, params, start, end):
    print(f"\n[{label}] ablation_mode={mode}")
    bt = D2SBacktestAblation(
        ablation_mode=mode,
        params=params,
        start_date=start,
        end_date=end,
    )
    bt.run(verbose=False)
    bt.print_report()
    return bt.report()


def main():
    print("=" * 60)
    print("  Study 8: Polymarket 레짐 단독 기여도 Ablation")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    results = {}
    configs = [
        ("full_3signal", "full",      "streak+SMA+Poly (기본 v3)"),
        ("no_poly",      "no_poly",   "streak+SMA만 (Poly 제거)"),
        ("poly_only",    "poly_only", "Poly 신호만"),
        ("no_regime",    "no_regime", "레짐 감지 없음 (tp=5% hd=8일)"),
    ]

    for key, mode, _ in configs:
        results[key] = run_ablation(key, mode, D2S_ENGINE_V3_NO_ROBN, START_DATE, END_DATE)

    print("\n" + "=" * 60)
    print("  Ablation 비교 요약")
    print("=" * 60)
    print(f"  {'모드':16s}  {'수익률':>10}  {'MDD':>8}  {'Sharpe':>8}  {'승률':>8}")
    print(f"  {'-'*55}")
    for key, _, desc in configs:
        r = results[key]
        print(
            f"  {key:16s}  "
            f"{r.get('total_return_pct',0):>9.1f}%  "
            f"{r.get('mdd_pct',0):>7.1f}%  "
            f"{r.get('sharpe_ratio',0):>8.3f}  "
            f"{r.get('win_rate',0):>7.1f}%"
        )
    print("=" * 60)

    result = {
        "study": "study_8_regime_ablation",
        "run_date": datetime.now().isoformat(),
        "period": {"start": str(START_DATE), "end": str(END_DATE)},
        **results,
    }
    out_path = RESULTS_DIR / f"study_8_regime_ablation_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()

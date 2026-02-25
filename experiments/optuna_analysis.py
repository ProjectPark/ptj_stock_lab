#!/usr/bin/env python3
"""
Optuna 최적화 결과 분석 스크립트
==================================
모든 Optuna DB를 로드하여 best trial, param 비교, importance를 출력한다.

Usage:
    pyenv shell ptj_stock_lab && python experiments/optuna_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import config

# ── 현재 config.py 파라미터 스냅샷 ─────────────────────────────────
CURRENT_CONFIG = {
    # v2 shared
    "COIN_TRIGGER_PCT": config.COIN_TRIGGER_PCT,
    "CONL_TRIGGER_PCT": config.CONL_TRIGGER_PCT,
    "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
    "STOP_LOSS_BULLISH_PCT": config.STOP_LOSS_BULLISH_PCT,
    "DCA_MAX_COUNT": config.DCA_MAX_COUNT,
    "PAIR_GAP_SELL_THRESHOLD_V2": config.PAIR_GAP_SELL_THRESHOLD_V2,
    "MAX_HOLD_HOURS": config.MAX_HOLD_HOURS,
    "COIN_SELL_PROFIT_PCT": config.COIN_SELL_PROFIT_PCT,
    "COIN_SELL_BEARISH_PCT": config.COIN_SELL_BEARISH_PCT,
    "CONL_SELL_PROFIT_PCT": config.CONL_SELL_PROFIT_PCT,
    "CONL_SELL_AVG_PCT": config.CONL_SELL_AVG_PCT,
    "DCA_DROP_PCT": config.DCA_DROP_PCT,
    "TAKE_PROFIT_PCT": config.TAKE_PROFIT_PCT,
    "PAIR_SELL_FIRST_PCT": config.PAIR_SELL_FIRST_PCT,
    # v3
    "V3_PAIR_GAP_ENTRY_THRESHOLD": config.V3_PAIR_GAP_ENTRY_THRESHOLD,
    "V3_DCA_MAX_COUNT": config.V3_DCA_MAX_COUNT,
    "V3_COIN_TRIGGER_PCT": config.V3_COIN_TRIGGER_PCT,
    "V3_CONL_TRIGGER_PCT": config.V3_CONL_TRIGGER_PCT,
    "V3_SPLIT_BUY_INTERVAL_MIN": config.V3_SPLIT_BUY_INTERVAL_MIN,
    "V3_SIDEWAYS_MIN_SIGNALS": config.V3_SIDEWAYS_MIN_SIGNALS,
    "V3_SIDEWAYS_POLY_LOW": config.V3_SIDEWAYS_POLY_LOW,
    "V3_SIDEWAYS_POLY_HIGH": config.V3_SIDEWAYS_POLY_HIGH,
    "V3_SIDEWAYS_GLD_THRESHOLD": config.V3_SIDEWAYS_GLD_THRESHOLD,
    "V3_SIDEWAYS_INDEX_THRESHOLD": config.V3_SIDEWAYS_INDEX_THRESHOLD,
    # v4
    "V4_PAIR_GAP_ENTRY_THRESHOLD": config.V4_PAIR_GAP_ENTRY_THRESHOLD,
    "V4_DCA_MAX_COUNT": config.V4_DCA_MAX_COUNT,
    "V4_COIN_TRIGGER_PCT": config.V4_COIN_TRIGGER_PCT,
    "V4_CONL_TRIGGER_PCT": config.V4_CONL_TRIGGER_PCT,
    "V4_SPLIT_BUY_INTERVAL_MIN": config.V4_SPLIT_BUY_INTERVAL_MIN,
    "V4_CB_VIX_SPIKE_PCT": config.V4_CB_VIX_SPIKE_PCT,
    "V4_CB_VIX_COOLDOWN_DAYS": config.V4_CB_VIX_COOLDOWN_DAYS,
    "V4_CB_GLD_SPIKE_PCT": config.V4_CB_GLD_SPIKE_PCT,
    "V4_CB_GLD_COOLDOWN_DAYS": config.V4_CB_GLD_COOLDOWN_DAYS,
    "V4_CB_BTC_CRASH_PCT": config.V4_CB_BTC_CRASH_PCT,
    "V4_CB_BTC_SURGE_PCT": config.V4_CB_BTC_SURGE_PCT,
    "V4_HIGH_VOL_MOVE_PCT": config.V4_HIGH_VOL_MOVE_PCT,
    "V4_HIGH_VOL_HIT_COUNT": config.V4_HIGH_VOL_HIT_COUNT,
    "V4_HIGH_VOL_STOP_LOSS_PCT": config.V4_HIGH_VOL_STOP_LOSS_PCT,
    "V4_CONL_ADX_MIN": config.V4_CONL_ADX_MIN,
    "V4_CONL_EMA_SLOPE_MIN_PCT": config.V4_CONL_EMA_SLOPE_MIN_PCT,
    "V4_PAIR_IMMEDIATE_SELL_PCT": config.V4_PAIR_IMMEDIATE_SELL_PCT,
    "V4_PAIR_FIXED_TP_PCT": config.V4_PAIR_FIXED_TP_PCT,
    "V4_SIDEWAYS_MIN_SIGNALS": config.V4_SIDEWAYS_MIN_SIGNALS,
    "V4_SIDEWAYS_POLY_LOW": config.V4_SIDEWAYS_POLY_LOW,
    "V4_SIDEWAYS_POLY_HIGH": config.V4_SIDEWAYS_POLY_HIGH,
    "V4_SIDEWAYS_GLD_THRESHOLD": config.V4_SIDEWAYS_GLD_THRESHOLD,
    "V4_SIDEWAYS_INDEX_THRESHOLD": config.V4_SIDEWAYS_INDEX_THRESHOLD,
    "V4_SIDEWAYS_ATR_DECLINE_PCT": config.V4_SIDEWAYS_ATR_DECLINE_PCT,
    "V4_SIDEWAYS_VOLUME_DECLINE_PCT": config.V4_SIDEWAYS_VOLUME_DECLINE_PCT,
    "V4_SIDEWAYS_EMA_SLOPE_MAX": config.V4_SIDEWAYS_EMA_SLOPE_MAX,
    "V4_SIDEWAYS_RSI_LOW": config.V4_SIDEWAYS_RSI_LOW,
    "V4_SIDEWAYS_RSI_HIGH": config.V4_SIDEWAYS_RSI_HIGH,
    "V4_SIDEWAYS_RANGE_MAX_PCT": config.V4_SIDEWAYS_RANGE_MAX_PCT,
}


def load_study_safe(study_name: str, storage: str) -> optuna.Study | None:
    """에러 시 None 반환."""
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"  [WARN] {study_name} 로드 실패: {e}")
        return None


def get_best_single(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Single-objective best trial."""
    try:
        return study.best_trial
    except Exception:
        return None


def get_best_multi(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Multi-objective: score = return - 0.5 * MDD."""
    pareto = study.best_trials
    if not pareto:
        return None
    return max(pareto, key=lambda t: t.values[0] - 0.5 * t.values[1])


def get_importance_top5(study: optuna.Study) -> dict:
    """파라미터 중요도 Top-5 반환."""
    try:
        importance = optuna.importance.get_param_importances(study)
        return dict(list(importance.items())[:5])
    except Exception:
        return {}


def fmt_delta(best_val, current_val) -> str:
    """Δ 포맷."""
    if best_val is None or current_val is None:
        return "N/A"
    try:
        delta = float(best_val) - float(current_val)
        return f"{delta:+.3f}" if abs(delta) >= 0.001 else "0"
    except (TypeError, ValueError):
        return "N/A"


def print_param_diff(best_params: dict, label: str = ""):
    """Best params vs Current config 차이 테이블 출력."""
    if label:
        print(f"\n  {label}")
    print(f"  {'파라미터':<45} {'Best':>10} {'Current':>10} {'Δ':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10}")
    for k, v in sorted(best_params.items()):
        cur = CURRENT_CONFIG.get(k, None)
        cur_str = f"{cur}" if cur is not None else "-"
        v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
        delta = fmt_delta(v, cur) if cur is not None else "-"
        print(f"  {k:<45} {v_str:>10} {cur_str:>10} {delta:>10}")


# ── DB 정의 ───────────────────────────────────────────────────────────
DB_DEFS = [
    {
        "label": "v2 Multi-obj",
        "db": str(config.OPTUNA_DIR / "optuna_v2.db"),
        "study_name": "ptj_v2_expanded",
        "mode": "multi",
        "version": "v2",
    },
    {
        "label": "v3 Full",
        "db": str(config.OPTUNA_DIR / "optuna_v3.db"),
        "study_name": "ptj_v3_full",
        "mode": "single",
        "version": "v3",
    },
    {
        "label": "v3 Phase2",
        "db": str(config.OPTUNA_DIR / "optuna_v3_phase2.db"),
        "study_name": "ptj_v3_phase2",
        "mode": "single",
        "version": "v3",
    },
    {
        "label": "v3 TrainTest",
        "db": str(config.OPTUNA_DIR / "optuna_v3_train_test.db"),
        "study_name": "ptj_v3_train_test",
        "mode": "single",
        "version": "v3",
    },
    {
        "label": "v3 TrainTest v2",
        "db": str(config.OPTUNA_DIR / "optuna_v3_train_test_v2.db"),
        "study_name": "ptj_v3_train_test_v2",
        "mode": "single",
        "version": "v3",
    },
    {
        "label": "v3 Wide",
        "db": str(config.OPTUNA_DIR / "optuna_v3_train_test_wide.db"),
        "study_name": "ptj_v3_train_test_wide",
        "mode": "single",
        "version": "v3",
    },
    {
        "label": "v4 Phase1",
        "db": str(config.OPTUNA_DIR / "optuna_v4_phase1.db"),
        "study_name": "ptj_v4_phase1",
        "mode": "single",
        "version": "v4",
    },
    {
        "label": "crash_model v1",
        "db": str(config.OPTUNA_DIR / "crash_model_params.db"),
        "study_name": "crash_model_v1",
        "mode": "single",
        "version": "misc",
    },
]


def main():
    print("=" * 70)
    print("  PTJ Optuna 최적화 결과 분석")
    print("=" * 70)

    # ── 요약 테이블 ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  요약 테이블")
    print(f"{'─'*70}")
    print(f"  {'DB 파일':<35} {'Study':<30} {'#Trials':>8} {'Complete':>9} {'Best Score':>12}")
    print(f"  {'-'*35} {'-'*30} {'-'*8} {'-'*9} {'-'*12}")

    results = []
    for d in DB_DEFS:
        db_path = Path(d["db"])
        if not db_path.exists():
            print(f"  {db_path.name:<35} {'(파일 없음)':<30} {'-':>8} {'-':>9} {'-':>12}")
            continue

        storage = f"sqlite:///{db_path}"
        study = load_study_safe(d["study_name"], storage)
        if study is None:
            print(f"  {db_path.name:<35} {d['study_name']:<30} {'ERR':>8} {'-':>9} {'-':>12}")
            continue

        total = len(study.trials)
        complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        if d["mode"] == "multi":
            best = get_best_multi(study)
            score_str = f"{best.values[0]:+.2f}% ret" if best else "-"
        else:
            best = get_best_single(study)
            score_str = f"{best.value:+.4f}" if best else "-"

        print(f"  {db_path.name:<35} {d['study_name']:<30} {total:>8} {complete:>9} {score_str:>12}")
        results.append({**d, "study": study, "best": best, "complete": complete, "total": total})

    # ── 버전별 상세 분석 ─────────────────────────────────────────────
    for r in results:
        study = r["study"]
        best = r["best"]
        label = r["label"]

        print(f"\n{'='*70}")
        print(f"  [{label}] {r['study_name']}")
        print(f"{'='*70}")
        print(f"  DB     : {Path(r['db']).name}")
        print(f"  Version: {r['version']}")
        print(f"  Trials : {r['complete']} 완료 / {r['total']} 전체")

        if best is None:
            print("  Best trial 없음")
            continue

        if r["mode"] == "multi":
            print(f"  Best   : Trial #{best.number} | Return={best.values[0]:+.2f}% | MDD=-{best.values[1]:.2f}%")
        else:
            print(f"  Best   : Trial #{best.number} | Score={best.value:+.4f}")
            attrs = best.user_attrs
            if "total_return_pct" in attrs:
                print(f"           Return={attrs['total_return_pct']:+.2f}% | MDD=-{attrs.get('mdd', '?'):.2f}%"
                      f" | Sharpe={attrs.get('sharpe', '?'):.4f}")

        print_param_diff(best.params, label="파라미터 비교 (Best vs Current config):")

        # Importance
        importance = get_importance_top5(study)
        if importance:
            print(f"\n  Top-5 파라미터 중요도 (fANOVA):")
            for k, v in importance.items():
                bar = "█" * int(v * 30)
                print(f"    {k:<45} {v:.4f}  {bar}")

    # ── v5 상태 ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  v5 Optuna 최적화 상태")
    print(f"{'='*70}")
    v5_db = config.OPTUNA_DIR / "optuna_v5.db"
    if v5_db.exists():
        v5_study = load_study_safe("ptj_v5_opt", f"sqlite:///{v5_db}")
        if v5_study:
            c = len([t for t in v5_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"  v5 DB 발견: {c}개 완료 trial")
        else:
            print("  v5 DB 존재하지만 study 로드 실패")
    else:
        print("  v5 Optuna DB 없음 — 최적화 미실행")
        print("  권장: pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v5_optuna.py --stage 2 --n-trials 400")

    print(f"\n{'='*70}")
    print("  분석 완료")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

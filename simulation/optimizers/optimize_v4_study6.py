#!/usr/bin/env python3
"""
PTJ v4 — Study 6: 전체 파라미터 종합 최적화 (Comprehensive)
============================================================
목적:
  Study 1~5 확정 파라미터를 warm start로, Group A~F 전체를 동시 탐색.
  파라미터 탐색 범위를 이전 study 대비 대폭 확대하여
  글로벌 최적점을 찾는다.

배경:
  - Study 1 (3y): Group A~E 최적화 완료
  - Study 2: Swing 파라미터 최적화 (TRIGGER=27.5, STAGE1_DRAWDOWN=-11.0)
  - Study 3/4: OOS 실패 (진입 억제 / 횡보장 강화 → 거래 소멸)
  - Study 5: 매도/BTC CB 5개 최적화 완료
  → Study 6: 확정 값들을 warm start로 전체 파라미터 재탐색

옵션 D (--swing-stage1-revival):
  Swing Stage1 파라미터 범위를 대폭 확장하여 Stage1을 살리는 방안 탐색.
  현재 Stage1: 2회 발동, 2회 손절 (-$2,255). 더 넓은 범위로 재도전.

병렬 전략:
  SQLite DB — mp.Pool(n_jobs=19) + db=sqlite:///data/optuna/optuna_v4_study6.db

스코어:
  balanced (phase=2): return - 2.0×max(0, mdd-7.5) - MDD>12 페널티
                       - 0.8×max(0, 40-win_rate) - 과소거래 페널티

Usage:
    # 전체 실행 (baseline → Optuna → OOS)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study6.py \\
      --n-trials 500 --n-jobs 19 \\
      --train-start 2023-01-03 --oos-start 2026-01-01

    # Swing Stage1 살리기 옵션 D 포함
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study6.py \\
      --n-trials 500 --n-jobs 19 --swing-stage1-revival \\
      --train-start 2023-01-03 --oos-start 2026-01-01

    # Stage 2만 (baseline 이미 실행된 경우)
    pyenv shell ptj_stock_lab && python simulation/optimizers/optimize_v4_study6.py --stage 2 \\
      --n-trials 500 --n-jobs 19 --train-start 2023-01-03 --oos-start 2026-01-01
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import config
from simulation.optimizers.optimize_v4_optuna import _calc_score
from simulation.optimizers.optimize_v4_study5 import V4Study5Optimizer, _STUDY2_BEST_SWING
from simulation.optimizers.optimize_v4_study1 import _PHASE1_BEST_OVERRIDES, _run_oos_eval
from simulation.optimizers.optimizer_base import TrialResult


# ============================================================
# 상수
# ============================================================

STUDY6_STUDY_NAME  = "ptj_v4_study6"
STUDY6_JOURNAL     = "data/optuna/optuna_v4_study6.log"   # JournalFileBackend (병렬 안전)
STUDY5_DB          = "sqlite:///data/optuna/optuna_v4_study5.db"

_REPORTS_DIR = _PROJECT_ROOT / "docs" / "reports" / "optuna"


# ============================================================
# Study 6 확정 warm-start 값 (Study 1+2+5 베스트 통합)
# ============================================================

_STUDY6_WARM_START: dict[str, Any] = {
    # ── Study 1 (3y) Phase1 Best ─────────────────────────────
    **_PHASE1_BEST_OVERRIDES,

    # ── Study 2 Swing Best ───────────────────────────────────
    **_STUDY2_BEST_SWING,

    # ── Study 5 매도/BTC CB Best (config.py 반영값) ──────────
    "V4_PAIR_FIXED_TP_PCT":        config.V4_PAIR_FIXED_TP_PCT,       # 5.0
    "V4_PAIR_IMMEDIATE_SELL_PCT":  config.V4_PAIR_IMMEDIATE_SELL_PCT,  # 0.40
    "PAIR_GAP_SELL_THRESHOLD_V2":  config.PAIR_GAP_SELL_THRESHOLD_V2,  # 9.0
    "V4_CB_BTC_CRASH_PCT":         config.V4_CB_BTC_CRASH_PCT,         # -6.0
    "V4_CB_BTC_SURGE_PCT":         config.V4_CB_BTC_SURGE_PCT,         # 13.5

    # ── Phase1 CB Best ───────────────────────────────────────
    "V4_CB_VIX_SPIKE_PCT":         config.V4_CB_VIX_SPIKE_PCT,         # 3.0
    "V4_CB_VIX_COOLDOWN_DAYS":     config.V4_CB_VIX_COOLDOWN_DAYS,     # 13
    "V4_CB_GLD_SPIKE_PCT":         config.V4_CB_GLD_SPIKE_PCT,         # 3.0
    "V4_CB_GLD_COOLDOWN_DAYS":     config.V4_CB_GLD_COOLDOWN_DAYS,     # 3
}


# ============================================================
# V4Study6Optimizer
# ============================================================


class V4Study6Optimizer(V4Study5Optimizer):
    """Study 6: 전체 파라미터 종합 최적화 (Group A~F + 옵션D 스윙Stage1).

    Study 5 확정 파라미터를 warm start로, 모든 그룹을 동시 탐색한다.
    탐색 범위를 이전 study 대비 대폭 확대.
    """

    version = "v4"

    def __init__(
        self,
        phase1_db: str = "sqlite:///data/optuna/optuna_v4_phase1.db",
        objective_mode: str = "balanced",
        train_start: str | None = None,
        swing_stage1_revival: bool = False,
    ):
        super().__init__(
            phase1_db=phase1_db,
            objective_mode=objective_mode,
            train_start=train_start,
        )
        self.swing_stage1_revival = swing_stage1_revival
        self._optuna_reports_dir = _REPORTS_DIR

    # ── Baseline: Study5 확정값 전체 ─────────────────────────

    def get_baseline_params(self) -> dict:
        """Study 1+2+5 확정값 전체를 baseline으로 사용."""
        base = super().get_baseline_params()  # Study5가 Study1+2 포함
        # Study6에서 추가 탐색할 Swing Stage2 파라미터 기본값 보완
        base.setdefault("V4_SWING_STAGE2_HOLD_DAYS",   config.V4_SWING_STAGE2_HOLD_DAYS)
        base.setdefault("V4_SWING_STAGE2_STOP_PCT",    config.V4_SWING_STAGE2_STOP_PCT)
        base.setdefault("V4_SWING_STAGE2_WEIGHT_PCT",  config.V4_SWING_STAGE2_WEIGHT_PCT)
        base.setdefault("V4_SIDEWAYS_BB_WIDTH_PERCENTILE", config.V4_SIDEWAYS_BB_WIDTH_PERCENTILE)
        return base

    # ── 탐색 공간: Group A~F 전체 ────────────────────────────

    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Study 6 전체 탐색 공간 (Group A~F + 옵션D Stage1 살리기)."""
        params = self.get_baseline_params()

        # ── Group A: 공유 매도/손절 파라미터 ─────────────────
        params.update({
            "STOP_LOSS_PCT": trial.suggest_float(
                "STOP_LOSS_PCT", -7.0, -2.0, step=0.25),
            "STOP_LOSS_BULLISH_PCT": trial.suggest_float(
                "STOP_LOSS_BULLISH_PCT", -22.0, -8.0, step=0.5),
            "COIN_SELL_PROFIT_PCT": trial.suggest_float(
                "COIN_SELL_PROFIT_PCT", 2.0, 9.0, step=0.5),
            "COIN_SELL_BEARISH_PCT": trial.suggest_float(
                "COIN_SELL_BEARISH_PCT", 0.1, 1.0, step=0.1),
            "CONL_SELL_PROFIT_PCT": trial.suggest_float(
                "CONL_SELL_PROFIT_PCT", 1.0, 7.0, step=0.5),
            "CONL_SELL_AVG_PCT": trial.suggest_float(
                "CONL_SELL_AVG_PCT", 0.25, 3.0, step=0.25),
            "DCA_DROP_PCT": trial.suggest_float(
                "DCA_DROP_PCT", -3.0, -0.2, step=0.05),
            "MAX_HOLD_HOURS": trial.suggest_int(
                "MAX_HOLD_HOURS", 2, 12),
            "TAKE_PROFIT_PCT": trial.suggest_float(
                "TAKE_PROFIT_PCT", 1.5, 9.0, step=0.5),
            "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float(
                "PAIR_GAP_SELL_THRESHOLD_V2", 4.0, 18.0, step=0.5),
            "PAIR_SELL_FIRST_PCT": trial.suggest_float(
                "PAIR_SELL_FIRST_PCT", 0.4, 1.0, step=0.05),
        })

        # ── Group B: 횡보장 감지 파라미터 ────────────────────
        params.update({
            "V4_SIDEWAYS_MIN_SIGNALS": trial.suggest_int(
                "V4_SIDEWAYS_MIN_SIGNALS", 2, 6),
            "V4_SIDEWAYS_POLY_LOW": trial.suggest_float(
                "V4_SIDEWAYS_POLY_LOW", 0.20, 0.55, step=0.05),
            "V4_SIDEWAYS_POLY_HIGH": trial.suggest_float(
                "V4_SIDEWAYS_POLY_HIGH", 0.45, 0.80, step=0.05),
            "V4_SIDEWAYS_GLD_THRESHOLD": trial.suggest_float(
                "V4_SIDEWAYS_GLD_THRESHOLD", 0.1, 1.2, step=0.1),
            "V4_SIDEWAYS_INDEX_THRESHOLD": trial.suggest_float(
                "V4_SIDEWAYS_INDEX_THRESHOLD", 0.2, 2.0, step=0.1),
            "V4_SIDEWAYS_GAP_FAIL_COUNT": trial.suggest_int(
                "V4_SIDEWAYS_GAP_FAIL_COUNT", 1, 6),
            "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": trial.suggest_int(
                "V4_SIDEWAYS_TRIGGER_FAIL_COUNT", 1, 6),
            "V4_SIDEWAYS_ATR_DECLINE_PCT": trial.suggest_float(
                "V4_SIDEWAYS_ATR_DECLINE_PCT", 5.0, 55.0, step=5.0),
            "V4_SIDEWAYS_VOLUME_DECLINE_PCT": trial.suggest_float(
                "V4_SIDEWAYS_VOLUME_DECLINE_PCT", 10.0, 65.0, step=5.0),
            "V4_SIDEWAYS_EMA_SLOPE_MAX": trial.suggest_float(
                "V4_SIDEWAYS_EMA_SLOPE_MAX", 0.02, 0.40, step=0.02),
            "V4_SIDEWAYS_RSI_LOW": trial.suggest_float(
                "V4_SIDEWAYS_RSI_LOW", 30.0, 55.0, step=5.0),
            "V4_SIDEWAYS_RSI_HIGH": trial.suggest_float(
                "V4_SIDEWAYS_RSI_HIGH", 45.0, 70.0, step=5.0),
            "V4_SIDEWAYS_RANGE_MAX_PCT": trial.suggest_float(
                "V4_SIDEWAYS_RANGE_MAX_PCT", 0.5, 10.0, step=0.5),
            "V4_SIDEWAYS_BB_WIDTH_PERCENTILE": trial.suggest_float(
                "V4_SIDEWAYS_BB_WIDTH_PERCENTILE", 10.0, 50.0, step=5.0),
        })

        # ── Group C: CB / 고변동성 파라미터 ──────────────────
        params.update({
            "V4_CB_GLD_SPIKE_PCT": trial.suggest_float(
                "V4_CB_GLD_SPIKE_PCT", 1.5, 7.0, step=0.5),
            "V4_CB_GLD_COOLDOWN_DAYS": trial.suggest_int(
                "V4_CB_GLD_COOLDOWN_DAYS", 1, 12),
            "V4_CB_BTC_CRASH_PCT": trial.suggest_float(
                "V4_CB_BTC_CRASH_PCT", -15.0, -3.0, step=0.5),
            "V4_CB_BTC_SURGE_PCT": trial.suggest_float(
                "V4_CB_BTC_SURGE_PCT", 3.0, 25.0, step=0.5),
            "V4_CB_VIX_SPIKE_PCT": trial.suggest_float(
                "V4_CB_VIX_SPIKE_PCT", 2.0, 15.0, step=1.0),
            "V4_CB_VIX_COOLDOWN_DAYS": trial.suggest_int(
                "V4_CB_VIX_COOLDOWN_DAYS", 3, 25),
            "V4_HIGH_VOL_MOVE_PCT": trial.suggest_float(
                "V4_HIGH_VOL_MOVE_PCT", 5.0, 25.0, step=1.0),
            "V4_HIGH_VOL_HIT_COUNT": trial.suggest_int(
                "V4_HIGH_VOL_HIT_COUNT", 1, 5),
            "V4_HIGH_VOL_STOP_LOSS_PCT": trial.suggest_float(
                "V4_HIGH_VOL_STOP_LOSS_PCT", -12.0, -2.0, step=0.5),
        })

        # ── Group D: CONL 필터 / 매도 파라미터 ───────────────
        params.update({
            "V4_CONL_ADX_MIN": trial.suggest_float(
                "V4_CONL_ADX_MIN", 5.0, 40.0, step=2.5),
            "V4_CONL_EMA_SLOPE_MIN_PCT": trial.suggest_float(
                "V4_CONL_EMA_SLOPE_MIN_PCT", -1.5, 1.5, step=0.1),
            "V4_PAIR_IMMEDIATE_SELL_PCT": trial.suggest_float(
                "V4_PAIR_IMMEDIATE_SELL_PCT", 0.1, 0.9, step=0.1),
            "V4_PAIR_FIXED_TP_PCT": trial.suggest_float(
                "V4_PAIR_FIXED_TP_PCT", 2.0, 18.0, step=0.5),
        })

        # ── Group E: 진입 타이밍 / 자금 파라미터 ─────────────
        params.update({
            "V4_PAIR_GAP_ENTRY_THRESHOLD": trial.suggest_float(
                "V4_PAIR_GAP_ENTRY_THRESHOLD", 1.0, 18.0, step=0.5),
            "V4_DCA_MAX_COUNT": trial.suggest_int(
                "V4_DCA_MAX_COUNT", 1, 15),
            "V4_MAX_PER_STOCK": trial.suggest_int(
                "V4_MAX_PER_STOCK", 2_000, 15_000, step=500),
            "V4_INITIAL_BUY": trial.suggest_int(
                "V4_INITIAL_BUY", 800, 5_000, step=200),
            "V4_DCA_BUY": trial.suggest_int(
                "V4_DCA_BUY", 200, 2_500, step=100),
            "V4_COIN_TRIGGER_PCT": trial.suggest_float(
                "V4_COIN_TRIGGER_PCT", 2.0, 12.0, step=0.5),
            "V4_CONL_TRIGGER_PCT": trial.suggest_float(
                "V4_CONL_TRIGGER_PCT", 2.0, 14.0, step=0.5),
            "V4_SPLIT_BUY_INTERVAL_MIN": trial.suggest_int(
                "V4_SPLIT_BUY_INTERVAL_MIN", 5, 60, step=5),
            "V4_ENTRY_CUTOFF_HOUR": trial.suggest_int(
                "V4_ENTRY_CUTOFF_HOUR", 8, 15),
            "V4_ENTRY_CUTOFF_MINUTE": trial.suggest_categorical(
                "V4_ENTRY_CUTOFF_MINUTE", [0, 30]),
        })

        # ── Group F: Swing Stage2 파라미터 ───────────────────
        params.update({
            "V4_SWING_TRIGGER_PCT": trial.suggest_float(
                "V4_SWING_TRIGGER_PCT", 10.0, 45.0, step=2.5),
            "V4_SWING_STAGE2_HOLD_DAYS": trial.suggest_int(
                "V4_SWING_STAGE2_HOLD_DAYS", 42, 210, step=7),
            "V4_SWING_STAGE2_STOP_PCT": trial.suggest_float(
                "V4_SWING_STAGE2_STOP_PCT", -15.0, -2.0, step=0.5),
            "V4_SWING_STAGE2_WEIGHT_PCT": trial.suggest_float(
                "V4_SWING_STAGE2_WEIGHT_PCT", 40.0, 100.0, step=10.0),
        })

        # ── Group F_D (옵션): Swing Stage1 살리기 ────────────
        #   --swing-stage1-revival 플래그 시 훨씬 넓은 범위로 탐색
        if self.swing_stage1_revival:
            params.update({
                "V4_SWING_STAGE1_DRAWDOWN_PCT": trial.suggest_float(
                    "V4_SWING_STAGE1_DRAWDOWN_PCT", -35.0, -3.0, step=0.5),
                "V4_SWING_STAGE1_ATR_MULT": trial.suggest_float(
                    "V4_SWING_STAGE1_ATR_MULT", 0.25, 8.0, step=0.25),
                "V4_SWING_STAGE1_HOLD_DAYS": trial.suggest_int(
                    "V4_SWING_STAGE1_HOLD_DAYS", 7, 210, step=7),
                "V4_SWING_STAGE1_WEIGHT_PCT": trial.suggest_float(
                    "V4_SWING_STAGE1_WEIGHT_PCT", 30.0, 100.0, step=10.0),
            })
        else:
            # Stage1 비활성: Study2 확정값 고정
            params.update({
                "V4_SWING_STAGE1_DRAWDOWN_PCT": _STUDY2_BEST_SWING["V4_SWING_STAGE1_DRAWDOWN_PCT"],
                "V4_SWING_STAGE1_ATR_MULT":     _STUDY2_BEST_SWING["V4_SWING_STAGE1_ATR_MULT"],
                "V4_SWING_STAGE1_HOLD_DAYS":    _STUDY2_BEST_SWING["V4_SWING_STAGE1_HOLD_DAYS"],
            })

        return params

    # ── 목적함수 ─────────────────────────────────────────────

    def calc_score(self, result: TrialResult) -> float:
        """Study 6: balanced (phase=2). Stage1 살리기 모드 시 Stage1 페널티 완화."""
        base_score = _calc_score(result.to_dict(), self.objective_mode, phase=2)

        # Stage1 살리기 모드: Stage1 손실 페널티 (과도 페널티 제거)
        # Stage1이 살아있되 과도 손실 방지 → 약한 패널티만 적용
        if self.swing_stage1_revival:
            s1_pnl = result.sig_stats.get("swing_stage1", {}).get("pnl", 0.0)
            if s1_pnl < -500_000:  # -50만원 초과 손실 시만 소프트 페널티
                base_score -= 1.0
        return base_score

    # ── warm start: Study5 확정값 ────────────────────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if already_done > 0:
            print(f"  기존 완료 trial {already_done}개 발견 — warm start skip")
            return already_done

        # warm start #0: Study 1+2+5 확정 best
        ws = dict(_STUDY6_WARM_START)
        if self.swing_stage1_revival:
            ws["V4_SWING_STAGE1_DRAWDOWN_PCT"] = -11.0
            ws["V4_SWING_STAGE1_ATR_MULT"]     = 2.5
            ws["V4_SWING_STAGE1_HOLD_DAYS"]    = 21
            ws["V4_SWING_STAGE1_WEIGHT_PCT"]   = config.V4_SWING_STAGE1_WEIGHT_PCT
        ws.setdefault("V4_SWING_STAGE2_HOLD_DAYS",  config.V4_SWING_STAGE2_HOLD_DAYS)
        ws.setdefault("V4_SWING_STAGE2_STOP_PCT",   config.V4_SWING_STAGE2_STOP_PCT)
        ws.setdefault("V4_SWING_STAGE2_WEIGHT_PCT", config.V4_SWING_STAGE2_WEIGHT_PCT)
        ws.setdefault("V4_SIDEWAYS_BB_WIDTH_PERCENTILE", config.V4_SIDEWAYS_BB_WIDTH_PERCENTILE)
        study.enqueue_trial(ws)
        print("  warm start #0: Study 1+2+5 확정 best enqueue")

        # warm start #1: baseline (config 현재값)
        study.enqueue_trial(baseline_params)
        print("  warm start #1: baseline (config 현재값) enqueue")

        # warm-up 2개 순차 실행
        print("  warm-up 2개 trial 순차 실행 중...")
        obj = self._make_objective(**kwargs)
        study.optimize(obj, n_trials=2, show_progress_bar=False)
        already_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        print(f"  warm-up 완료 (총 완료: {already_done}회)")
        return already_done

    # ── 리포트 ───────────────────────────────────────────────

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs):
        oos_start = kwargs.get("oos_start")
        if oos_start:
            _run_oos_eval(self, study, oos_start, top_n=5)

        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        label = "study6_revival" if self.swing_stage1_revival else "study6"
        report_path = _REPORTS_DIR / f"v4_{label}_optuna_report.md"
        report_path.write_text(md, encoding="utf-8")
        print(f"\n  Study 6 리포트: {report_path}")

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return
        best = study.best_trial
        print("\n" + "=" * 65)
        mode_label = " [Stage1 살리기 모드]" if self.swing_stage1_revival else ""
        print(f"  [Study 6 결과]{mode_label}  Best Trial #{best.number}")
        print("=" * 65)
        key_params = [
            "PAIR_GAP_SELL_THRESHOLD_V2", "V4_PAIR_FIXED_TP_PCT",
            "V4_SWING_TRIGGER_PCT", "V4_CB_BTC_SURGE_PCT", "V4_CB_VIX_COOLDOWN_DAYS",
            "STOP_LOSS_BULLISH_PCT", "V4_SIDEWAYS_ATR_DECLINE_PCT",
        ]
        if self.swing_stage1_revival:
            key_params += [
                "V4_SWING_STAGE1_DRAWDOWN_PCT", "V4_SWING_STAGE1_ATR_MULT",
                "V4_SWING_STAGE1_HOLD_DAYS",
            ]
        for k in key_params:
            v = best.params.get(k, "N/A")
            cur = getattr(config, k, "N/A")
            print(f"  {k:<44s} = {v}  (현재: {cur})")
        print("=" * 65)


# ============================================================
# 병렬 워커
# ============================================================


def _study6_worker(
    study_name: str,
    journal_path_str: str,
    n_trials: int,
    phase1_db: str,
    objective: str,
    train_start: str | None,
    train_end: str | None,
    swing_stage1_revival: bool,
) -> None:
    """병렬 워커: 독립 프로세스에서 study.optimize()를 실행한다."""
    import optuna as _optuna
    from optuna.samplers import TPESampler as _TPE
    from optuna.storages import JournalStorage as _JS
    from optuna.storages.journal import JournalFileBackend as _JFB
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    _storage = _JS(_JFB(journal_path_str))
    sampler = _TPE(n_startup_trials=30, seed=None)
    study = _optuna.load_study(
        study_name=study_name,
        storage=_storage,
        sampler=sampler,
    )
    opt = V4Study6Optimizer(
        phase1_db=phase1_db,
        objective_mode=objective,
        train_start=train_start,
        swing_stage1_revival=swing_stage1_revival,
    )
    obj_kwargs: dict = {}
    if train_start:
        obj_kwargs["start_date"] = train_start
    if train_end:
        obj_kwargs["end_date"] = train_end
    obj = opt._make_objective(**obj_kwargs)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)


# ============================================================
# Stage 1 (Baseline)
# ============================================================


def run_stage1(args: argparse.Namespace) -> tuple[dict, dict]:
    """Stage 1: Study 1+2+5 확정값으로 baseline 실행."""
    opt = V4Study6Optimizer(
        phase1_db=args.phase1_db,
        objective_mode=args.objective,
        train_start=args.train_start,
        swing_stage1_revival=args.swing_stage1_revival,
    )

    print("\n" + "=" * 70)
    print("  [Stage 1] Baseline — Study 1+2+5 확정값 기준")
    if args.train_start:
        print(f"  start_date = {args.train_start}")
    if args.swing_stage1_revival:
        print("  [옵션 D] Swing Stage1 살리기 모드 활성")
    print("=" * 70)

    params = opt.get_baseline_params()

    run_kwargs: dict = {}
    if args.train_start:
        run_kwargs["start_date"] = args.train_start

    print("\n  실행 중...")
    t0 = time.time()
    result = opt.run_single_trial(params, **run_kwargs)
    elapsed = time.time() - t0
    print(f"  완료 ({elapsed:.1f}초)")

    opt._print_result_summary(result)
    print(f"    CB 차단   : {result.cb_buy_blocks}회")
    print(f"    횡보장 일수: {result.sideways_days}일 / {result.total_trading_days}일")

    opt.save_baseline_json(result, params)

    label = "study6_revival" if args.swing_stage1_revival else "study6"
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = opt._generate_baseline_report(result, params)
    report_path = _REPORTS_DIR / f"v4_{label}_baseline_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"  Baseline 리포트: {report_path}")

    return result.to_dict(), params


# ============================================================
# Stage 2 (Optuna)
# ============================================================


def run_stage2(
    args: argparse.Namespace,
    baseline: dict | None = None,
    baseline_params: dict | None = None,
) -> None:
    """Stage 2: 전체 파라미터 Optuna 최적화."""
    train_start  = args.train_start
    train_end    = getattr(args, "train_end", None)
    oos_start    = getattr(args, "oos_start", None)
    n_jobs       = args.n_jobs
    n_trials     = args.n_trials

    opt = V4Study6Optimizer(
        phase1_db=args.phase1_db,
        objective_mode=args.objective,
        train_start=train_start,
        swing_stage1_revival=args.swing_stage1_revival,
    )

    if baseline is None or baseline_params is None:
        baseline, baseline_params = opt.load_baseline_json()
        print(f"  Baseline 로드: {opt._baseline_json}")
    print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")

    mode_label = " [Stage1 살리기]" if args.swing_stage1_revival else ""
    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Study 6{mode_label} Optuna ({n_trials} trials, {n_jobs} workers)")
    print(f"  탐색: Group A~F 전체 ({_count_search_params(args)} 파라미터)")
    print(f"  DB: {args.db}")
    if train_start:
        print(f"  Train: {train_start} ~{(' '+train_end) if train_end else ''}")
    if oos_start:
        print(f"  OOS  : {oos_start} ~")
    print(f"{'=' * 70}")

    journal_path = Path(args.db)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal_path)))

    sampler = TPESampler(seed=42, n_startup_trials=min(40, n_trials))
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    obj_kwargs: dict = {}
    if train_start:
        obj_kwargs["start_date"] = train_start
    if train_end:
        obj_kwargs["end_date"] = train_end

    already_done = opt._pre_optimize_setup(study, baseline_params, **obj_kwargs)
    remaining = n_trials - already_done

    if remaining <= 0:
        print(f"  목표 trial {n_trials}회 이미 완료됨 (완료: {already_done}회)")
    else:
        n_workers = min(n_jobs, remaining)
        base_n = remaining // n_workers
        extra  = remaining % n_workers

        worker_args = [
            (
                args.study_name,
                args.db,
                base_n + (1 if i < extra else 0),
                args.phase1_db,
                args.objective,
                train_start,
                train_end,
                args.swing_stage1_revival,
            )
            for i in range(n_workers)
            if base_n + (1 if i < extra else 0) > 0
        ]

        print(f"  워커 {len(worker_args)}개 × ~{base_n}개 trial 병렬 실행 중...")
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            pool.starmap(_study6_worker, worker_args)
        elapsed = time.time() - t0
        print(f"  실행 완료: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

    # 결과 로드 + 요약
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\n  완료된 trial 없음")
        return

    best = study.best_trial
    diff = best.value - baseline["total_return_pct"]
    print(f"\n  BEST Trial #{best.number}")
    print(f"  스코어  : {best.value:+.4f}  (baseline 대비 {diff:+.2f})")
    print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
    print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")
    print(f"  CB 차단 : {best.user_attrs.get('cb_buy_blocks', 0)}회")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print(f"\n  Top 5:")
    print(f"  {'#':>4s}  {'스코어':>8s}  {'MDD':>7s}  {'WR':>6s}  {'Sharpe':>8s}  {'CB차단':>7s}")
    for t in top5:
        print(
            f"  {t.number:4d}  {t.value:+8.4f}"
            f"  -{t.user_attrs.get('mdd', 0):5.2f}%"
            f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
            f"  {t.user_attrs.get('cb_buy_blocks', 0):>6}"
        )

    elapsed_total = time.time() - t0 if remaining > 0 else 0
    opt._post_optimize(
        study, baseline, baseline_params, elapsed_total, n_jobs,
        oos_start=oos_start,
    )


# ============================================================
# 유틸리티
# ============================================================


def _count_search_params(args: argparse.Namespace) -> int:
    """탐색 파라미터 수를 반환한다."""
    # Group A(11) + B(14) + C(9) + D(4) + E(10) + F(4) = 52
    # + Stage1 살리기(4) = 56
    base = 52
    return base + (4 if args.swing_stage1_revival else 0)


# ============================================================
# 메인
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="PTJ v4 Study 6 — 전체 파라미터 종합 최적화 (Group A~F)"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 1→2 연속)")
    parser.add_argument("--n-trials",  type=int, default=500,
                        help="Optuna trial 수 (기본: 500)")
    parser.add_argument("--n-jobs",    type=int, default=19,
                        help="병렬 워커 수 (기본: 19 — SLURM 20 CPU 기준)")
    parser.add_argument("--timeout",   type=int, default=None,
                        help="최대 실행 시간(초, 기본: 없음)")
    parser.add_argument("--study-name", type=str, default=STUDY6_STUDY_NAME,
                        help=f"Optuna study 이름 (기본: {STUDY6_STUDY_NAME})")
    parser.add_argument("--db",        type=str, default=STUDY6_JOURNAL,
                        help=f"JournalStorage 경로 (기본: {STUDY6_JOURNAL})")
    parser.add_argument("--phase1-db", type=str,
                        default="sqlite:///data/optuna/optuna_v4_phase1.db")
    parser.add_argument("--objective", type=str, choices=["return", "balanced"],
                        default="balanced")
    parser.add_argument("--train-start", type=str, default="2023-01-03",
                        help="훈련 시작일 (기본: 2023-01-03, 3y 데이터)")
    parser.add_argument("--train-end",   type=str, default=None)
    parser.add_argument("--oos-start",   type=str, default="2026-01-01",
                        help="OOS 평가 시작일 (기본: 2026-01-01)")
    parser.add_argument("--swing-stage1-revival", action="store_true",
                        help="[옵션 D] Swing Stage1 살리기 — Stage1 파라미터 대폭 확장 탐색")
    args = parser.parse_args()

    print("=" * 70)
    print("  PTJ v4 — Study 6: 전체 파라미터 종합 최적화")
    print(f"  Group A~F ({_count_search_params(args)}개 파라미터)")
    if args.swing_stage1_revival:
        print("  [옵션 D] Swing Stage1 살리기 모드 활성")
    print(f"  objective={args.objective} | n-trials={args.n_trials} | n-jobs={args.n_jobs}")
    print(f"  DB: {args.db}")
    print(f"  Train: {args.train_start} ~{(' '+args.train_end) if args.train_end else ''}")
    print(f"  OOS  : {args.oos_start} ~")
    print("=" * 70)

    if args.stage == 1:
        run_stage1(args)
    elif args.stage == 2:
        run_stage2(args)
    else:
        baseline, baseline_params = run_stage1(args)
        print()
        run_stage2(args, baseline=baseline, baseline_params=baseline_params)

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

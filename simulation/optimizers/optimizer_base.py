"""
PTJ 매매법 — Optuna 최적화 베이스 클래스
=========================================
v2~v5 모든 Optuna 최적화의 공통 로직을 추출한 ABC.

- TrialResult: 백테스트 1회 결과 데이터클래스
- BaseOptimizer: Stage 1 (baseline) + Stage 2 (Optuna) 공통 워크플로우
- 각 버전은 get_baseline_params(), create_engine(), define_search_space()만 구현

Usage:
    from optimizers.optimizer_base import BaseOptimizer, TrialResult

    class V5Optimizer(BaseOptimizer):
        version = "v5"
        ...
"""
from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from simulation.optimizers import report_sections as rs

# ============================================================
# TrialResult
# ============================================================


@dataclass
class TrialResult:
    """백테스트 1회 실행 결과 — 모든 버전 공통 지표."""

    final_equity: float = 0.0
    total_return_pct: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0
    total_fees: float = 0.0
    total_sells: int = 0
    total_buys: int = 0
    win_rate: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_pnl: float = 0.0
    stop_loss_count: int = 0
    stop_loss_pnl: float = 0.0
    time_stop_count: int = 0
    eod_close_count: int = 0
    sideways_days: int = 0
    sideways_blocks: int = 0
    entry_cutoff_blocks: int = 0
    daily_limit_blocks: int = 0
    cb_buy_blocks: int = 0
    cb_sell_halt_bars: int = 0
    total_trading_days: int = 0
    sig_stats: dict = field(default_factory=dict)
    exit_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """dict 변환 (JSON 직렬화용)."""
        return {
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "mdd": self.mdd,
            "sharpe": self.sharpe,
            "total_fees": self.total_fees,
            "total_sells": self.total_sells,
            "total_buys": self.total_buys,
            "win_rate": self.win_rate,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_pnl": self.total_pnl,
            "stop_loss_count": self.stop_loss_count,
            "stop_loss_pnl": self.stop_loss_pnl,
            "time_stop_count": self.time_stop_count,
            "eod_close_count": self.eod_close_count,
            "sideways_days": self.sideways_days,
            "sideways_blocks": self.sideways_blocks,
            "entry_cutoff_blocks": self.entry_cutoff_blocks,
            "daily_limit_blocks": self.daily_limit_blocks,
            "cb_buy_blocks": self.cb_buy_blocks,
            "cb_sell_halt_bars": self.cb_sell_halt_bars,
            "total_trading_days": self.total_trading_days,
            "sig_stats": self.sig_stats,
            "exit_stats": self.exit_stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrialResult:
        """dict에서 생성."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# 공통 지표 추출
# ============================================================


def extract_metrics(engine: Any) -> TrialResult:
    """백테스트 엔진에서 TrialResult 지표를 추출한다.

    v2~v5 모든 엔진에 공통으로 적용된다. 엔진의 attribute 이름이
    일관되므로 (initial_capital_krw, equity_curve, trades, ...) 단일
    함수로 추출 가능하다.
    """
    from simulation.backtests import backtest_common

    initial = engine.initial_capital_krw
    final = engine.equity_curve[-1][1] if engine.equity_curve else initial
    total_ret = (final - initial) / initial * 100
    mdd = backtest_common.calc_mdd(engine.equity_curve)
    sharpe = backtest_common.calc_sharpe(engine.equity_curve)
    total_fees = engine.total_buy_fees_krw + engine.total_sell_fees_krw

    sells = [t for t in engine.trades if t.side == "SELL"]
    buys = [t for t in engine.trades if t.side == "BUY"]
    stop_losses = [t for t in sells if t.exit_reason == "stop_loss"]
    time_stops = [t for t in sells if t.exit_reason == "time_stop"]
    eod_closes = [t for t in sells if t.exit_reason == "eod_close"]
    wins = [t for t in sells if t.pnl_krw > 0]
    losses = [t for t in sells if t.pnl_krw < 0]

    # 시그널별 통계
    sig_stats: dict = {}
    for t in sells:
        key = t.signal_type
        if key not in sig_stats:
            sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
        sig_stats[key]["count"] += 1
        sig_stats[key]["pnl"] += t.pnl_krw
        if t.pnl_krw > 0:
            sig_stats[key]["wins"] += 1

    # 매도 사유별 통계
    exit_stats: dict = {}
    for t in sells:
        key = t.exit_reason or "unknown"
        if key not in exit_stats:
            exit_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
        exit_stats[key]["count"] += 1
        exit_stats[key]["pnl"] += t.pnl_krw
        if t.pnl_krw > 0:
            exit_stats[key]["wins"] += 1

    return TrialResult(
        final_equity=final,
        total_return_pct=total_ret,
        mdd=mdd,
        sharpe=sharpe,
        total_fees=total_fees,
        total_sells=len(sells),
        total_buys=len(buys),
        win_rate=len(wins) / len(sells) * 100 if sells else 0,
        win_count=len(wins),
        loss_count=len(losses),
        avg_win=sum(t.pnl_krw for t in wins) / len(wins) if wins else 0,
        avg_loss=sum(t.pnl_krw for t in losses) / len(losses) if losses else 0,
        total_pnl=sum(t.pnl_krw for t in sells),
        stop_loss_count=len(stop_losses),
        stop_loss_pnl=sum(t.pnl_krw for t in stop_losses),
        time_stop_count=len(time_stops),
        eod_close_count=len(eod_closes),
        sideways_days=getattr(engine, "sideways_days", 0),
        sideways_blocks=getattr(engine, "sideways_blocks", 0),
        entry_cutoff_blocks=getattr(engine, "entry_cutoff_blocks", 0),
        daily_limit_blocks=getattr(engine, "daily_limit_blocks", 0),
        cb_buy_blocks=getattr(engine, "cb_buy_blocks", 0),
        cb_sell_halt_bars=getattr(engine, "cb_sell_halt_bars", 0),
        total_trading_days=getattr(engine, "total_trading_days", 0),
        sig_stats=sig_stats,
        exit_stats=exit_stats,
    )


def extract_metrics_usd(engine: Any) -> TrialResult:
    """USD 기반 v2 엔진에서 TrialResult 지표를 추출한다."""
    from simulation.backtests import backtest_common

    initial = engine.initial_capital_usd
    final = engine.equity_curve[-1][1] if engine.equity_curve else initial
    total_ret = (final - initial) / initial * 100
    mdd = backtest_common.calc_mdd(engine.equity_curve)
    sharpe = backtest_common.calc_sharpe(engine.equity_curve)
    total_fees = engine.total_buy_fees_usd + engine.total_sell_fees_usd

    sells = [t for t in engine.trades if t.side == "SELL"]
    buys = [t for t in engine.trades if t.side == "BUY"]
    stop_losses = [t for t in sells if t.exit_reason == "stop_loss"]
    wins = [t for t in sells if getattr(t, "pnl_usd", getattr(t, "pnl_krw", 0)) > 0]
    losses = [t for t in sells if getattr(t, "pnl_usd", getattr(t, "pnl_krw", 0)) < 0]

    return TrialResult(
        final_equity=final,
        total_return_pct=total_ret,
        mdd=mdd,
        sharpe=sharpe,
        total_fees=total_fees,
        total_sells=len(sells),
        total_buys=len(buys),
        win_rate=len(wins) / len(sells) * 100 if sells else 0,
        win_count=len(wins),
        loss_count=len(losses),
        stop_loss_count=len(stop_losses),
        total_trading_days=getattr(engine, "total_trading_days", 0),
    )


# ============================================================
# BaseOptimizer ABC
# ============================================================


class BaseOptimizer(ABC):
    """v2~v5 Optuna 최적화 공통 워크플로우.

    서브클래스는 version, get_baseline_params(), create_engine(),
    define_search_space()를 구현한다.
    """

    version: str  # "v2", "v3", "v4", "v5"

    def __init__(self):
        ver = self.version
        self._docs_dir = Path(__file__).resolve().parent / "docs"
        self._baseline_json = config.RESULTS_DIR / "baselines" / f"{ver}_baseline_result.json"
        self._baseline_report = self._docs_dir / f"{ver}_baseline_report.md"
        self._optuna_report = self._docs_dir / f"{ver}_optuna_report.md"

    # ── 서브클래스 필수 구현 ───────────────────────────────────

    @abstractmethod
    def get_baseline_params(self) -> dict:
        """현재 config.py 기본값을 config key -> value dict로 반환."""

    @abstractmethod
    def create_engine(self, params: dict, **kwargs) -> Any:
        """파라미터를 config에 주입 + 엔진 인스턴스를 생성한다.

        Parameters
        ----------
        params : dict
            config key -> value 매핑 (예: {"V5_PAIR_GAP_ENTRY_THRESHOLD": 2.2})
        **kwargs
            start_date, end_date 등 엔진별 추가 인자
        """

    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> dict:
        """Optuna 탐색 공간을 정의하고 params dict를 반환."""

    # ── 서브클래스 선택 오버라이드 ─────────────────────────────

    def calc_score(self, result: TrialResult) -> float:
        """최적화 스코어를 계산한다. 기본: total_return_pct."""
        return result.total_return_pct

    def get_trial_user_attrs(self, result: TrialResult) -> dict[str, Any]:
        """trial.set_user_attr()에 기록할 지표 키를 반환한다."""
        return {
            "final_equity": result.final_equity,
            "mdd": result.mdd,
            "sharpe": result.sharpe,
            "total_fees": result.total_fees,
            "total_sells": result.total_sells,
            "total_buys": result.total_buys,
            "win_rate": result.win_rate,
            "stop_loss_count": result.stop_loss_count,
            "time_stop_count": result.time_stop_count,
            "sideways_days": result.sideways_days,
            "sideways_blocks": result.sideways_blocks,
            "entry_cutoff_blocks": result.entry_cutoff_blocks,
            "daily_limit_blocks": result.daily_limit_blocks,
            "cb_buy_blocks": result.cb_buy_blocks,
        }

    # ── 공통 실행 함수 ────────────────────────────────────────

    def run_single_trial(self, params: dict, **kwargs) -> TrialResult:
        """파라미터로 백테스트 1회를 실행하고 지표를 반환한다.

        config에 setattr/restore 패턴을 사용한다.
        """
        import config as _config

        originals = {}
        for key, value in params.items():
            originals[key] = getattr(_config, key)
            setattr(_config, key, value)

        try:
            engine = self.create_engine(params, **kwargs)
            engine.run(verbose=False)
            return extract_metrics(engine)
        finally:
            for key, value in originals.items():
                setattr(_config, key, value)

    # ── Stage 1: Baseline ─────────────────────────────────────

    def save_baseline_json(self, result: TrialResult, params: dict) -> None:
        """baseline 결과를 JSON으로 저장."""
        (config.RESULTS_DIR / "baselines").mkdir(parents=True, exist_ok=True)
        payload = {
            "result": result.to_dict(),
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self._baseline_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Baseline JSON: {self._baseline_json}")

    def load_baseline_json(self) -> tuple[dict, dict]:
        """저장된 baseline JSON을 로드한다. (result dict, params dict)"""
        if not self._baseline_json.exists():
            raise FileNotFoundError(
                f"Baseline 결과 없음: {self._baseline_json}\n"
                "  먼저 --stage 1을 실행하세요."
            )
        with open(self._baseline_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["result"], data["params"]

    def run_stage1(self) -> tuple[TrialResult, dict]:
        """Stage 1: baseline 실행 -> JSON 저장 -> 마크다운 리포트."""
        print("\n" + "=" * 70)
        print(f"  [Stage 1] Baseline - 현재 config 가중치 시뮬레이션 ({self.version})")
        print("=" * 70)

        params = self.get_baseline_params()

        print("\n  실행 중...")
        t0 = time.time()
        result = self.run_single_trial(params)
        elapsed = time.time() - t0
        print(f"  완료 ({elapsed:.1f}초)")

        # 콘솔 요약
        self._print_result_summary(result)

        # JSON 저장
        self.save_baseline_json(result, params)

        # 마크다운 리포트
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        md = self._generate_baseline_report(result, params)
        self._baseline_report.write_text(md, encoding="utf-8")
        print(f"  Baseline 리포트: {self._baseline_report}")

        return result, params

    # ── 훅 메서드 (서브클래스 오버라이드 가능) ──────────────────

    def _pre_optimize_setup(self, study, baseline_params, **kwargs) -> int:
        """study.optimize() 전 준비 (warm start, enqueue 등).

        Returns: 이미 실행 완료된 trial 수.
        """
        study.enqueue_trial(baseline_params)
        return 0

    def _get_progress_callbacks(self, n_trials: int) -> list:
        """progress callback 리스트. 기본: 빈 리스트."""
        return []

    def _post_optimize(self, study, baseline, baseline_params, elapsed, n_jobs, **kwargs) -> None:
        """optimize 후 추가 작업. 기본: 마크다운 리포트 저장."""
        md = self._generate_optuna_report(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        self._docs_dir.mkdir(parents=True, exist_ok=True)
        self._optuna_report.write_text(md, encoding="utf-8")
        print(f"\n  Optuna 리포트: {self._optuna_report}")

    # ── Stage 2: Optuna ───────────────────────────────────────

    def run_stage2(
        self,
        n_trials: int = 20,
        n_jobs: int = 6,
        timeout: int | None = None,
        study_name: str | None = None,
        db: str | None = None,
        baseline: dict | None = None,
        baseline_params: dict | None = None,
        **kwargs,
    ) -> None:
        """Stage 2: Optuna 최적화 실행 -> 마크다운 리포트."""
        if study_name is None:
            study_name = f"ptj_{self.version}_opt"

        # baseline 로드
        if baseline is None or baseline_params is None:
            baseline, baseline_params = self.load_baseline_json()
            print(f"  Baseline 로드: {self._baseline_json}")
            print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")

        print(f"\n{'=' * 70}")
        print(f"  [Stage 2] Optuna 최적화 ({n_trials} trials, {n_jobs} workers)")
        print(f"{'=' * 70}")

        sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials))
        storage = db if db else None

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        # 훅 1: pre-optimize (warm start / enqueue)
        already_done = self._pre_optimize_setup(study, baseline_params, **kwargs)

        # 훅 2: objective (kwargs 전달)
        obj_kwargs = {k: v for k, v in kwargs.items() if k in ("end_date",)}
        objective = self._make_objective(**obj_kwargs)

        # 훅 3: callbacks
        callbacks = self._get_progress_callbacks(n_trials)

        t0 = time.time()

        remaining = n_trials - already_done
        if remaining <= 0:
            print(f"  목표 trial {n_trials}회 이미 완료됨 (완료: {already_done}회)")
        elif n_jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_jobs) as pool:
                self._pool = pool
                study.optimize(
                    objective,
                    n_trials=remaining,
                    timeout=timeout,
                    show_progress_bar=True,
                    callbacks=callbacks or None,
                )
                self._pool = None
        else:
            self._pool = None
            study.optimize(
                objective,
                n_trials=remaining,
                timeout=timeout,
                show_progress_bar=True,
                callbacks=callbacks or None,
            )

        elapsed = time.time() - t0

        # 콘솔 요약
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            best = study.best_trial
            diff = best.value - baseline["total_return_pct"]
            print(f"\n  실행 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
            print(f"  Trial당 평균: {elapsed / max(len(study.trials), 1):.1f}초")
            print(f"\n  BEST Trial #{best.number}")
            print(f"  수익률  : {best.value:+.2f}%  (baseline 대비 {diff:+.2f}%)")
            print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
            print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
            print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")

            top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
            print(f"\n  Top 5:")
            print(f"  {'#':>4s}  {'수익률':>8s}  {'MDD':>8s}  {'Sharpe':>8s}  {'승률':>6s}")
            for t in top5:
                print(
                    f"  {t.number:4d}  {t.value:+7.2f}%"
                    f"  -{t.user_attrs.get('mdd', 0):6.2f}%"
                    f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
                    f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
                )
        else:
            print("\n  완료된 trial 없음")
            return

        # 훅 4: post-optimize (리포트 + 추가 작업)
        self._post_optimize(study, baseline, baseline_params, elapsed, n_jobs, **kwargs)

    # ── Objective 생성 ────────────────────────────────────────

    def _make_objective(self, **kwargs):
        """Optuna objective callable을 반환한다.

        **kwargs는 run_single_trial()에 전달된다 (예: end_date).
        """
        optimizer = self

        class _Objective:
            def __call__(self_obj, trial: optuna.Trial) -> float:
                params = optimizer.define_search_space(trial)
                result = optimizer.run_single_trial(params, **kwargs)
                # 지표 기록
                for attr_key, value in optimizer.get_trial_user_attrs(result).items():
                    trial.set_user_attr(attr_key, value)
                return optimizer.calc_score(result)

        return _Objective()

    # ── 콘솔 요약 출력 ────────────────────────────────────────

    def _print_result_summary(self, result: TrialResult) -> None:
        """baseline/trial 결과 콘솔 요약."""
        r = result
        print(f"\n  수익률  : {r.total_return_pct:+.2f}%")
        print(f"  자산    : {r.final_equity:,.0f}원")
        print(f"  MDD     : -{r.mdd:.2f}%")
        print(f"  Sharpe  : {r.sharpe:.4f}")
        print(f"  승률    : {r.win_rate:.1f}%  ({r.win_count}W / {r.loss_count}L)")
        print(f"  매수/매도: {r.total_buys} / {r.total_sells}")
        total_blocks = r.sideways_blocks + r.entry_cutoff_blocks + r.daily_limit_blocks
        print(
            f"  차단    : {total_blocks}회 "
            f"(횡보 {r.sideways_blocks} / 시간 {r.entry_cutoff_blocks} / 일일 {r.daily_limit_blocks})"
        )

    # ── 마크다운 리포트 생성 (Baseline) ───────────────────────

    def _generate_baseline_report(self, result: TrialResult, params: dict) -> str:
        """Baseline 마크다운 리포트를 생성한다."""
        r = result
        ver = self.version
        total_blocks = r.sideways_blocks + r.entry_cutoff_blocks + r.daily_limit_blocks

        lines = [
            f"# PTJ {ver} Baseline 시뮬레이션 리포트",
            "",
            f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 1. 개요",
            "",
            f"현재 `config.py`에 설정된 {ver} 파라미터로 백테스트를 실행한 결과입니다.",
            "",
            "## 2. 핵심 지표",
            "",
            "| 지표 | 값 |",
            "|---|---|",
            f"| 총 수익률 | **{r.total_return_pct:+.2f}%** |",
            f"| 최종 자산 | {r.final_equity:,.0f}원 |",
            f"| MDD | -{r.mdd:.2f}% |",
            f"| Sharpe Ratio | {r.sharpe:.4f} |",
            f"| 총 손익 | {r.total_pnl:+,.0f}원 |",
            f"| 총 수수료 | {r.total_fees:,.0f}원 |",
            "",
            "## 3. 매매 통계",
            "",
            "| 지표 | 값 |",
            "|---|---|",
            f"| 거래일 | {r.total_trading_days}일 |",
            f"| 매수 횟수 | {r.total_buys}회 |",
            f"| 매도 횟수 | {r.total_sells}회 |",
            f"| 승/패 | {r.win_count}W / {r.loss_count}L |",
            f"| 승률 | {r.win_rate:.1f}% |",
            f"| 평균 수익 | {r.avg_win:+,.0f}원 |",
            f"| 평균 손실 | {r.avg_loss:+,.0f}원 |",
            "",
            f"## 4. {ver} 선별 매매 효과",
            "",
            "| 지표 | 값 |",
            "|---|---|",
            f"| 횡보장 감지일 | {r.sideways_days}일 / {r.total_trading_days}일 |",
            f"| 횡보장 차단 매수 | {r.sideways_blocks}회 |",
            f"| 시간제한 차단 | {r.entry_cutoff_blocks}회 |",
            f"| 일일1회 차단 | {r.daily_limit_blocks}회 |",
        ]
        if r.cb_buy_blocks > 0:
            lines.append(f"| CB 차단 매수 | {r.cb_buy_blocks}회 |")
        lines.append(f"| **총 차단 매수** | **{total_blocks}회** |")

        lines += [
            "",
            "## 5. 매도 사유별 성과",
            "",
            "| 사유 | 횟수 | P&L | 승률 |",
            "|---|---|---|---|",
        ]
        for key in sorted(r.exit_stats.keys()):
            s = r.exit_stats[key]
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            lines.append(f"| {key} | {s['count']}회 | {s['pnl']:+,.0f}원 | {wr:.1f}% |")

        lines += [
            "",
            "## 6. 시그널별 성과",
            "",
            "| 시그널 | 횟수 | P&L | 승률 |",
            "|---|---|---|---|",
        ]
        for key in sorted(r.sig_stats.keys()):
            s = r.sig_stats[key]
            wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
            lines.append(f"| {key} | {s['count']}회 | {s['pnl']:+,.0f}원 | {wr:.1f}% |")

        # 파라미터 섹션
        ver_upper = ver.upper() + "_"
        ver_keys = [k for k in sorted(params.keys()) if k.startswith(ver_upper)]
        shared_keys = [k for k in sorted(params.keys()) if not k.startswith(ver_upper)]

        lines += [
            "",
            "## 7. 현재 파라미터 (config.py)",
            "",
            f"### {ver} 고유",
            "",
            "| 파라미터 | 값 |",
            "|---|---|",
        ]
        for k in ver_keys:
            lines.append(f"| `{k}` | {_fmt_param_value(params[k])} |")

        lines += [
            "",
            "### v2 공유",
            "",
            "| 파라미터 | 값 |",
            "|---|---|",
        ]
        for k in shared_keys:
            lines.append(f"| `{k}` | {_fmt_param_value(params[k])} |")

        lines.append("")
        return "\n".join(lines)

    # ── 마크다운 리포트 생성 (Optuna) ─────────────────────────

    def _get_report_sections(
        self,
        study: optuna.Study,
        baseline: dict,
        baseline_params: dict,
        elapsed: float,
        n_jobs: int,
        **kwargs,
    ) -> list[list[str]]:
        """리포트 섹션 리스트. 서브클래스에서 오버라이드하여 섹션 추가/교체."""
        return [
            rs.section_execution_info(study, elapsed, n_jobs, self.version),
            rs.section_baseline_vs_best(study, baseline),
            rs.section_best_params(study, baseline_params),
            rs.section_top5_table(study),
            rs.section_importance(study),
            rs.section_top5_detail(study),
            rs.section_config_code(study),
        ]

    def _generate_optuna_report(
        self,
        study: optuna.Study,
        baseline: dict,
        baseline_params: dict,
        elapsed: float,
        n_jobs: int,
        **kwargs,
    ) -> str:
        """Optuna 마크다운 리포트를 생성한다."""
        sections = self._get_report_sections(
            study, baseline, baseline_params, elapsed, n_jobs, **kwargs
        )
        return rs.build_report(self.version, sections)


# ============================================================
# 유틸리티
# ============================================================


def _fmt_param_value(v: Any) -> str:
    """파라미터 값을 리포트용 문자열로 포맷."""
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, int) and v >= 1_000_000:
        return f"{v:,}"
    return str(v)

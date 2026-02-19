"""
PTJ 매매법 — 통합 파이프라인
============================
버전 + 파라미터만 지정하면 백테스트/최적화가 돌아가는 단일 진입점.

Usage:
    from pipeline import run_backtest, run_optimize

    # 백테스트
    engine = run_backtest("v5")
    engine = run_backtest("v4", params={"stop_loss_pct": -4.0})

    # Optuna 최적화
    run_optimize("v5", stage=1)                         # baseline only
    run_optimize("v5", stage=2, n_trials=100, n_jobs=6) # Optuna
"""
from __future__ import annotations

from typing import Any

from simulation.strategies.params import (
    BaseParams,
    V3Params,
    V4Params,
    V5Params,
    v2_params_from_config,
    v3_params_from_config,
    v4_params_from_config,
    v5_params_from_config,
)
from simulation.optimizers.optimizer_base import TrialResult, extract_metrics, extract_metrics_usd

# ============================================================
# Lazy imports — 엔진/옵티마이저는 사용 시점에 로드
# ============================================================

def _import_engine(version: str):
    """버전별 백테스트 엔진 클래스를 lazy import."""
    if version == "v2":
        from simulation.backtests.backtest_v2 import BacktestEngineV2
        return BacktestEngineV2
    elif version == "v3":
        from simulation.backtests.backtest_v3 import BacktestEngineV3
        return BacktestEngineV3
    elif version == "v4":
        from simulation.backtests.backtest_v4 import BacktestEngineV4
        return BacktestEngineV4
    elif version == "v5":
        from simulation.backtests.backtest_v5 import BacktestEngineV5
        return BacktestEngineV5
    else:
        raise ValueError(f"Unknown engine version: {version}")


def _import_optimizer(version: str):
    """버전별 옵티마이저 클래스를 lazy import."""
    if version == "v2":
        from simulation.optimizers.optimize_v2_optuna import V2Optimizer
        return V2Optimizer
    elif version == "v3":
        from simulation.optimizers.optimize_v3_optuna import V3Optimizer
        return V3Optimizer
    elif version == "v4":
        from simulation.optimizers.optimize_v4_optuna import V4Optimizer
        return V4Optimizer
    elif version == "v5":
        from simulation.optimizers.optimize_v5_optuna import V5Optimizer
        return V5Optimizer
    else:
        raise ValueError(f"Unknown optimizer version: {version}")


# ============================================================
# Registry (lookup tables)
# ============================================================

PARAMS_REGISTRY: dict[str, tuple[type, callable]] = {
    "v2": (BaseParams, v2_params_from_config),
    "v3": (V3Params, v3_params_from_config),
    "v4": (V4Params, v4_params_from_config),
    "v5": (V5Params, v5_params_from_config),
}

SUPPORTED_VERSIONS = list(PARAMS_REGISTRY.keys())


# ============================================================
# Public API
# ============================================================

def run_backtest(
    version: str,
    params: dict | None = None,
    start_date=None,
    end_date=None,
    use_fees: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Any:
    """통합 백테스트 진입점.

    Args:
        version: "v2" | "v3" | "v4" | "v5"
        params: 파라미터 오버라이드 dict. None이면 config.py 기본값 사용.
        start_date: 백테스트 시작일
        end_date: 백테스트 종료일
        use_fees: 수수료 적용 여부
        verbose: 상세 출력 여부
        **kwargs: 엔진 생성자에 전달할 추가 인자

    Returns:
        BacktestEngine 인스턴스 (run 완료 상태)
    """
    if version not in PARAMS_REGISTRY:
        raise ValueError(f"Unknown version '{version}'. Supported: {SUPPORTED_VERSIONS}")

    params_cls, from_config_fn = PARAMS_REGISTRY[version]

    # 파라미터 구성: config 기본값 + 오버라이드
    base_params = from_config_fn()
    if params:
        from dataclasses import replace
        base_params = replace(base_params, **params)

    # 엔진 생성 + 실행
    engine_cls = _import_engine(version)
    engine_kwargs = {"params": base_params, "use_fees": use_fees, **kwargs}
    if start_date is not None:
        engine_kwargs["start_date"] = start_date
    if end_date is not None:
        engine_kwargs["end_date"] = end_date

    engine = engine_cls(**engine_kwargs)
    engine.run(verbose=verbose)
    return engine


def run_optimize(
    version: str,
    stage: int = 0,
    n_trials: int = 100,
    n_jobs: int = 6,
    **kwargs,
) -> Any:
    """통합 Optuna 최적화 진입점.

    Args:
        version: "v2" | "v3" | "v4" | "v5"
        stage: 0=both, 1=baseline only, 2=Optuna only
        n_trials: Optuna trial 수
        n_jobs: 병렬 worker 수
        **kwargs: 옵티마이저에 전달할 추가 인자

    Returns:
        Optimizer 인스턴스
    """
    if version not in PARAMS_REGISTRY:
        raise ValueError(f"Unknown version '{version}'. Supported: {SUPPORTED_VERSIONS}")

    opt_cls = _import_optimizer(version)
    opt = opt_cls(**kwargs)

    if stage in (0, 1):
        opt.run_stage1()
    if stage in (0, 2):
        opt.run_stage2(n_trials=n_trials, n_jobs=n_jobs)

    return opt


def get_metrics(engine: Any) -> TrialResult:
    """어느 엔진이든 공통 메트릭을 추출한다.

    Args:
        engine: run() 완료된 BacktestEngine 인스턴스

    Returns:
        TrialResult with all metrics
    """
    # v2는 USD 기반, v3/v4/v5는 KRW-style 기반
    if hasattr(engine, "initial_capital_usd") and not hasattr(engine, "initial_capital_krw"):
        return extract_metrics_usd(engine)
    return extract_metrics(engine)


def export_for_inference(version: str, params: dict | None = None) -> dict:
    """ptj_stock 배포 서버용 파라미터 포맷으로 변환한다.

    Args:
        version: 엔진 버전
        params: 파라미터 dict. None이면 config.py 기본값.

    Returns:
        서버에서 사용할 수 있는 flat dict
    """
    if version not in PARAMS_REGISTRY:
        raise ValueError(f"Unknown version '{version}'. Supported: {SUPPORTED_VERSIONS}")

    params_cls, from_config_fn = PARAMS_REGISTRY[version]

    if params:
        p = params_cls.from_dict(params)
    else:
        p = from_config_fn()

    return {
        "version": version,
        "params": p.to_dict(),
    }

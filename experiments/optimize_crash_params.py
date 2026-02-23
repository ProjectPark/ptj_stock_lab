"""
Polymarket Crash Model — Optuna 파라미터 최적화
================================================
최적화 대상:
  - crash_score 가중치 6개 (합 = 1.0으로 정규화)
  - 포지션 커브 파라미터 2개 (soxl_cutoff, mstz_start)

목적함수: 훈련 기간(2024-02 ~ 2025-12) Sharpe Ratio 최대화
평가: 타겟 기간(2026-01 ~ 2026-02-17) 성과 확인

저장: data/optuna/crash_model_params.db
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.polymarket_crash_model import (
    load_poly_signals,
    load_market_data,
    TRAIN_START, TRAIN_END, TARGET_START, TARGET_END,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DB_PATH = ROOT / "data/optuna/crash_model_params.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 전역 데이터 (한 번만 로드) ─────────────────
_POLY_DF: pd.DataFrame | None = None
_MARKET_DF: pd.DataFrame | None = None
_MERGED: pd.DataFrame | None = None


def _get_merged() -> pd.DataFrame:
    global _POLY_DF, _MARKET_DF, _MERGED
    if _MERGED is None:
        _POLY_DF = load_poly_signals()
        _MARKET_DF = load_market_data()
        _MERGED = _POLY_DF.join(_MARKET_DF, how="left")
    return _MERGED.copy()


# ── 커스텀 crash_score 계산 ────────────────────

def compute_crash_score_custom(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """가중치 dict를 받아 crash_score 계산."""
    df = df.copy()

    # ① BTC 3일 하락
    btc_down = (1 - df["btc_up"].fillna(0.5))
    df["poly_btc_bear"] = btc_down.rolling(3, min_periods=1).mean()

    # ② 주간 dip 편향
    df["poly_weekly_dip_bias"] = 0.0
    mask = df["weekly_reach"].notna() & df["weekly_dip"].notna()
    df.loc[mask, "poly_weekly_dip_bias"] = (
        (df.loc[mask, "weekly_dip"] - df.loc[mask, "weekly_reach"]).clip(0, 1)
    )
    df["poly_weekly_dip_bias"] = df["poly_weekly_dip_bias"].replace(0, np.nan).ffill(limit=7).fillna(0)

    # ③ VIX 레벨
    df["vix_signal"] = df.get("VIX_norm", pd.Series(0.3, index=df.index)).fillna(0.3)

    # ④ SOXL 5일 모멘텀
    if "SOXL_ret" in df.columns:
        soxl_mom = -df["SOXL_ret"].rolling(5, min_periods=1).mean()
        df["mkt_momentum"] = soxl_mom.clip(0, 0.08) / 0.08
    else:
        df["mkt_momentum"] = 0.0

    # ⑤ VIX 급등
    if "VIX_chg" in df.columns:
        df["vix_spike"] = df["VIX_chg"].clip(0, 0.5).fillna(0) / 0.5
    else:
        df["vix_spike"] = 0.0

    # ⑥ BTC 실제 수익률
    if "BTC_ret" in df.columns:
        btc_drop = -df["BTC_ret"].rolling(2, min_periods=1).mean()
        df["btc_ret_signal"] = btc_drop.clip(0, 0.05) / 0.05
    else:
        df["btc_ret_signal"] = 0.0

    df["crash_score"] = (
        df["poly_btc_bear"]        * weights["w_btc"]
        + df["vix_signal"]         * weights["w_vix"]
        + df["mkt_momentum"]       * weights["w_mom"]
        + df["vix_spike"]          * weights["w_spike"]
        + df["btc_ret_signal"]     * weights["w_btc_ret"]
        + df["poly_weekly_dip_bias"] * weights["w_weekly"]
    )

    fed_hawk = df.get("fed_hike_prob", pd.Series(0.0, index=df.index)).fillna(0)
    df["crash_score"] = (df["crash_score"] + fed_hawk * 0.1).clip(0, 1)
    return df


# ── 포지션 결정 (커스텀 커브) ─────────────────

def assign_position_custom(score: float, soxl_cutoff: float, mstz_start: float) -> dict:
    """커스텀 파라미터로 연속 포지션 결정."""
    soxl_w = float(np.clip(1.0 - score / soxl_cutoff, 0.0, 1.0))
    mstz_w = float(np.clip((score - mstz_start) / (1.0 - mstz_start + 1e-9), 0.0, 1.0))
    tqqq_w = max(0.0, 1.0 - soxl_w - mstz_w)
    result = {}
    if soxl_w > 0.001: result["SOXL"] = soxl_w
    if tqqq_w > 0.001: result["TQQQ"] = tqqq_w
    if mstz_w > 0.001: result["MSTZ"] = mstz_w
    return result if result else {"SOXL": 1.0}


# ── 백테스트 (커스텀 파라미터) ─────────────────

def backtest_custom(
    df: pd.DataFrame,
    start: str, end: str,
    soxl_cutoff: float,
    mstz_start: float,
) -> pd.DataFrame:
    period = df.loc[start:end].copy()
    ret_map = {"SOXL": "SOXL_ret", "TQQQ": "TQQQ_ret", "MSTZ": "MSTZ_ret"}

    portfolio_ret = []
    for i, (_, row) in enumerate(period.iterrows()):
        score = 0.0 if i == 0 else float(period["crash_score"].iloc[i - 1])
        pos = assign_position_custom(score, soxl_cutoff, mstz_start)
        p_ret = sum(
            w * (row.get(ret_map[t], 0) or 0)
            for t, w in pos.items()
        )
        portfolio_ret.append(p_ret)

    period = period.copy()
    period["strategy_ret"] = portfolio_ret
    return period


# ── 목적함수 ───────────────────────────────────

def sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 10:
        return -999.0
    ann_ret = (1 + clean).prod() ** (252 / len(clean)) - 1
    vol = clean.std() * np.sqrt(252)
    return ann_ret / vol if vol > 1e-6 else -999.0


def calmar(returns: pd.Series) -> float:
    """Calmar = 연수익 / |MDD|"""
    clean = returns.dropna()
    if len(clean) < 10:
        return -999.0
    ann_ret = (1 + clean).prod() ** (252 / len(clean)) - 1
    cum = (1 + clean).cumprod()
    mdd = abs((cum / cum.cummax() - 1).min())
    return ann_ret / mdd if mdd > 1e-6 else -999.0


def objective(trial: optuna.Trial) -> float:
    # 가중치 (합 1.0 정규화)
    w_btc     = trial.suggest_float("w_btc",     0.05, 0.40)
    w_vix     = trial.suggest_float("w_vix",     0.05, 0.35)
    w_mom     = trial.suggest_float("w_mom",     0.10, 0.45)
    w_spike   = trial.suggest_float("w_spike",   0.05, 0.30)
    w_btc_ret = trial.suggest_float("w_btc_ret", 0.05, 0.30)
    w_weekly  = trial.suggest_float("w_weekly",  0.00, 0.15)

    total = w_btc + w_vix + w_mom + w_spike + w_btc_ret + w_weekly
    weights = {
        "w_btc":     w_btc / total,
        "w_vix":     w_vix / total,
        "w_mom":     w_mom / total,
        "w_spike":   w_spike / total,
        "w_btc_ret": w_btc_ret / total,
        "w_weekly":  w_weekly / total,
    }

    # 포지션 커브 파라미터
    soxl_cutoff = trial.suggest_float("soxl_cutoff", 0.25, 0.70)
    mstz_start  = trial.suggest_float("mstz_start",  0.40, 0.90)

    merged = _get_merged()
    scored = compute_crash_score_custom(merged, weights)
    result = backtest_custom(scored, TRAIN_START, TRAIN_END, soxl_cutoff, mstz_start)

    # 목적: Sharpe + Calmar 복합 (MDD도 고려)
    s = sharpe(result["strategy_ret"])
    c = calmar(result["strategy_ret"])
    return 0.6 * s + 0.4 * c


# ── 메인 ───────────────────────────────────────

def main(n_trials: int = 300) -> None:
    print(f"Optuna 최적화 시작 — {n_trials}회 시도")
    print(f"훈련 기간: {TRAIN_START} ~ {TRAIN_END}")
    print(f"DB: {DB_PATH}")

    # 데이터 사전 로드
    _get_merged()
    print("데이터 로드 완료")

    study = optuna.create_study(
        direction="maximize",
        study_name="crash_model_v1",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n최적 파라미터 (trial #{best.number}, score={best.value:.4f}):")
    for k, v in best.params.items():
        print(f"  {k:15s}: {v:.4f}")

    # 최적 파라미터로 훈련/타겟 성과 평가
    print("\n최적 파라미터 성과 검증:")
    total_w = sum(best.params[k] for k in ["w_btc","w_vix","w_mom","w_spike","w_btc_ret","w_weekly"])
    opt_weights = {k: best.params[k] / total_w
                   for k in ["w_btc","w_vix","w_mom","w_spike","w_btc_ret","w_weekly"]}
    sc = best.params["soxl_cutoff"]
    ms = best.params["mstz_start"]

    merged = _get_merged()
    scored = compute_crash_score_custom(merged, opt_weights)

    for label, start, end in [
        ("훈련", TRAIN_START, TRAIN_END),
        ("타겟", TARGET_START, TARGET_END),
    ]:
        res = backtest_custom(scored, start, end, sc, ms)
        soxl_res = backtest_custom(scored, start, end, 1.0, 1.0)  # SOXL 보유

        s_strat = sharpe(res["strategy_ret"])
        s_soxl  = sharpe(soxl_res["SOXL_ret"].fillna(0))
        total_strat = (1 + res["strategy_ret"].dropna()).prod() - 1
        total_soxl  = (1 + soxl_res["SOXL_ret"].fillna(0)).prod() - 1
        cum_strat = (1 + res["strategy_ret"]).cumprod()
        mdd_strat = (cum_strat / cum_strat.cummax() - 1).min()

        print(f"\n  [{label}]")
        print(f"    전략:  총수익={total_strat:+.1%}  샤프={s_strat:.2f}  MDD={mdd_strat:.1%}")
        print(f"    SOXL:  총수익={total_soxl:+.1%}  샤프={s_soxl:.2f}")

    # 최적 파라미터 저장
    out_path = ROOT / "data/results/backtests/optimal_crash_params.json"
    import json
    optimal = {
        "weights": opt_weights,
        "soxl_cutoff": sc,
        "mstz_start": ms,
        "best_score": best.value,
        "trial": best.number,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(optimal, f, indent=2)
    print(f"\n최적 파라미터 저장: {out_path}")

    # Top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe().sort_values("value", ascending=False).head(5)
    print(trials_df[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]].to_string(index=False))


if __name__ == "__main__":
    main(n_trials=300)

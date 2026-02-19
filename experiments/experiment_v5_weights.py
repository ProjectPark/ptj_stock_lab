#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.optimizers.optimize_v5_optuna import _get_baseline_params, _run_single_trial

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = PROJECT_ROOT / "data" / "v5_weight_experiments.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "v5_weight_experiments_report.md"

# Match the v5 Optuna search ranges so random experiments are comparable.
PARAM_SPACE = {
    "V5_PAIR_GAP_ENTRY_THRESHOLD": ("float", 1.0, 4.0, 0.2),
    "V5_DCA_MAX_COUNT": ("int", 1, 7, 1),
    "V5_MAX_PER_STOCK": ("int", 3_000, 10_000, 250),
    "V5_COIN_TRIGGER_PCT": ("float", 2.0, 7.0, 0.5),
    "V5_CONL_TRIGGER_PCT": ("float", 2.0, 7.0, 0.5),
    "V5_SPLIT_BUY_INTERVAL_MIN": ("int", 5, 30, 5),
    "V5_ENTRY_CUTOFF_HOUR": ("int", 10, 14, 1),
    "V5_ENTRY_CUTOFF_MINUTE": ("cat", [0, 30]),
    "V5_INITIAL_BUY": ("int", 1_000, 3_500, 250),
    "V5_DCA_BUY": ("int", 250, 1_500, 125),
    "V5_SIDEWAYS_MIN_SIGNALS": ("int", 2, 5, 1),
    "V5_SIDEWAYS_POLY_LOW": ("float", 0.30, 0.50, 0.05),
    "V5_SIDEWAYS_POLY_HIGH": ("float", 0.50, 0.70, 0.05),
    "V5_SIDEWAYS_GLD_THRESHOLD": ("float", 0.1, 0.8, 0.1),
    "V5_SIDEWAYS_INDEX_THRESHOLD": ("float", 0.2, 1.0, 0.1),
    "STOP_LOSS_PCT": ("float", -6.0, -1.5, 0.5),
    "STOP_LOSS_BULLISH_PCT": ("float", -12.0, -5.0, 0.5),
    "COIN_SELL_PROFIT_PCT": ("float", 1.0, 5.0, 0.5),
    "CONL_SELL_PROFIT_PCT": ("float", 1.0, 5.0, 0.5),
    "DCA_DROP_PCT": ("float", -2.0, -0.3, 0.1),
    "MAX_HOLD_HOURS": ("int", 2, 8, 1),
    "TAKE_PROFIT_PCT": ("float", 1.0, 5.0, 0.5),
    "PAIR_GAP_SELL_THRESHOLD_V2": ("float", 2.0, 10.0, 0.1),
    "PAIR_SELL_FIRST_PCT": ("float", 0.5, 1.0, 0.05),
}

METRIC_KEYS = [
    "total_return_pct",
    "mdd",
    "sharpe",
    "win_rate",
    "total_sells",
]


def _sample_float(rng: random.Random, low: float, high: float, step: float) -> float:
    steps = int(round((high - low) / step))
    idx = rng.randint(0, steps)
    return round(low + idx * step, 10)


def _sample_value(rng: random.Random, spec: tuple) -> int | float:
    kind = spec[0]
    if kind == "cat":
        return rng.choice(spec[1])

    low = spec[1]
    high = spec[2]
    step = spec[3]
    if kind == "int":
        steps = int((high - low) // step)
        return low + step * rng.randint(0, steps)
    if kind == "float":
        return _sample_float(rng, low, high, step)
    raise ValueError(f"Unsupported spec kind: {kind}")


def _sample_overrides(rng: random.Random) -> dict:
    return {key: _sample_value(rng, spec) for key, spec in PARAM_SPACE.items()}


def _fingerprint(params: dict) -> tuple:
    return tuple((k, params[k]) for k in sorted(PARAM_SPACE.keys()))


def _fmt_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}" if value >= 1_000_000 else str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _metrics_view(result: dict) -> dict:
    return {key: result.get(key) for key in METRIC_KEYS}


def _build_report(payload: dict, top_k: int) -> str:
    baseline = payload["baseline"]["result"]
    trials = payload["trials"]
    ranked = sorted(trials, key=lambda t: t["result"]["total_return_pct"], reverse=True)
    top = ranked[: min(top_k, len(ranked))]

    lines = [
        "# PTJ v5 Weight Experiment Report",
        "",
        f"- generated_at: {payload['timestamp']}",
        f"- baseline_runs: 1",
        f"- random_runs: {payload['n_trials']}",
        f"- seed: {payload['seed']}",
        f"- top_k: {top_k}",
        "",
        "## Baseline Metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]

    for key in METRIC_KEYS:
        value = baseline.get(key)
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Top Random Trials",
        "",
        "| rank | trial | total_return_pct | mdd | sharpe | win_rate | total_sells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, item in enumerate(top, start=1):
        result = item["result"]
        lines.append(
            "| "
            f"{rank} | {item['trial']} | "
            f"{result['total_return_pct']:+.2f} | "
            f"{result['mdd']:.2f} | "
            f"{result['sharpe']:.4f} | "
            f"{result['win_rate']:.1f} | "
            f"{result['total_sells']} |"
        )

    lines += [
        "",
        "## All Random Trials",
        "",
        "| trial | total_return_pct | mdd | sharpe | win_rate | total_sells |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(trials, key=lambda t: t["trial"]):
        result = item["result"]
        lines.append(
            "| "
            f"{item['trial']} | "
            f"{result['total_return_pct']:+.2f} | "
            f"{result['mdd']:.2f} | "
            f"{result['sharpe']:.4f} | "
            f"{result['win_rate']:.1f} | "
            f"{result['total_sells']} |"
        )

    if top:
        lines += [
            "",
            "## Top Trial Parameters",
            "",
        ]
    for rank, item in enumerate(top, start=1):
        lines += [
            f"### Rank {rank} - Trial {item['trial']}",
            "",
            "| parameter | value |",
            "|---|---|",
        ]
        for key in sorted(item["overrides"].keys()):
            lines.append(f"| {key} | {_fmt_number(item['overrides'][key])} |")
        lines.append("")

    return "\n".join(lines)


def run_experiments(n_trials: int, seed: int, top_k: int) -> dict:
    rng = random.Random(seed)

    baseline_params = _get_baseline_params()
    print("[v5] Running baseline...")
    baseline_start = time.time()
    baseline_result = _run_single_trial(baseline_params)
    baseline_elapsed = time.time() - baseline_start
    print(
        "[v5] Baseline done: "
        f"return={baseline_result['total_return_pct']:+.2f}% "
        f"elapsed={baseline_elapsed:.1f}s"
    )

    trials = []
    seen = {_fingerprint(baseline_params)}

    for trial_idx in range(1, n_trials + 1):
        attempts = 0
        while True:
            overrides = _sample_overrides(rng)
            fp = _fingerprint(overrides)
            attempts += 1
            if fp not in seen:
                seen.add(fp)
                break
            if attempts > 1000:
                raise RuntimeError("Could not sample a unique random parameter set.")

        params = dict(baseline_params)
        params.update(overrides)

        started = time.time()
        result = _run_single_trial(params)
        elapsed = time.time() - started

        trials.append(
            {
                "trial": trial_idx,
                "elapsed_sec": round(elapsed, 4),
                "overrides": overrides,
                "params": params,
                "result": result,
                "metrics": _metrics_view(result),
            }
        )
        print(
            f"[v5] trial {trial_idx:02d}/{n_trials:02d} "
            f"return={result['total_return_pct']:+.2f}% "
            f"mdd={result['mdd']:.2f} "
            f"sharpe={result['sharpe']:.4f}"
        )

    ranked = sorted(trials, key=lambda t: t["result"]["total_return_pct"], reverse=True)
    top_trials = [
        {
            "rank": rank,
            "trial": trial["trial"],
            "metrics": _metrics_view(trial["result"]),
        }
        for rank, trial in enumerate(ranked[: min(top_k, len(ranked))], start=1)
    ]

    payload = {
        "version": "v5",
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "seed": seed,
        "top_k": top_k,
        "metric_keys": METRIC_KEYS,
        "baseline": {
            "elapsed_sec": round(baseline_elapsed, 4),
            "params": baseline_params,
            "result": baseline_result,
            "metrics": _metrics_view(baseline_result),
        },
        "trials": trials,
        "top_trials": top_trials,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v5 weight experiments (baseline + random combinations)."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12,
        help="Number of random trials to run (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top random trials to include in the report (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 0:
        raise ValueError("--n-trials must be >= 0")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    payload = run_experiments(
        n_trials=args.n_trials,
        seed=args.seed,
        top_k=args.top_k,
    )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    report = _build_report(payload, top_k=args.top_k)
    OUTPUT_MD.write_text(report, encoding="utf-8")

    print(f"[v5] JSON saved: {OUTPUT_JSON}")
    print(f"[v5] Report saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

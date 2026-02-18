#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
  if [[ -x "${HOME}/.pyenv/versions/market/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.pyenv/versions/market/bin/python"
  fi
fi
if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
  echo "Neither python nor python3 is executable in this environment." >&2
  exit 1
fi

echo "[1/5] Checking required v4/v5 files..."
required_files=(
  "config.py"
  "backtest_v4.py"
  "backtest_v5.py"
  "signals_v4.py"
  "signals_v5.py"
  "optimize_v4_optuna.py"
  "optimize_v5_optuna.py"
  "experiment_v4_weights.py"
  "experiment_v5_weights.py"
  "scripts/check_v45_parity.sh"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 1
  fi
done
echo "OK"

echo "[2/5] Running python -m py_compile..."
"${PYTHON_BIN}" -m py_compile \
  config.py \
  backtest_v4.py \
  backtest_v5.py \
  signals_v4.py \
  signals_v5.py \
  optimize_v4_optuna.py \
  optimize_v5_optuna.py \
  experiment_v4_weights.py \
  experiment_v5_weights.py
echo "OK"

echo "[3/5] Checking required V4/V5 constants in config.py..."
if command -v rg >/dev/null 2>&1; then
  V4_CONSTS="$(rg -No 'config\.V4_[A-Z0-9_]+' backtest_v4.py signals_v4.py optimize_v4_optuna.py | sed 's/.*config\.//' | sort -u)"
  V5_CONSTS="$(rg -No 'config\.V5_[A-Z0-9_]+' backtest_v5.py signals_v5.py optimize_v5_optuna.py | sed 's/.*config\.//' | sort -u)"
else
  V4_CONSTS="$(grep -Eho 'config\.V4_[A-Z0-9_]+' backtest_v4.py signals_v4.py optimize_v4_optuna.py | sed 's/.*config\.//' | sort -u)"
  V5_CONSTS="$(grep -Eho 'config\.V5_[A-Z0-9_]+' backtest_v5.py signals_v5.py optimize_v5_optuna.py | sed 's/.*config\.//' | sort -u)"
fi

export V4_CONSTS
export V5_CONSTS
"${PYTHON_BIN}" - <<'PY'
import ast
from pathlib import Path
import os
import sys


def check(group: str, raw: str) -> list[str]:
    names = [line.strip() for line in raw.splitlines() if line.strip()]
    config_source = Path("config.py").read_text(encoding="utf-8")
    tree = ast.parse(config_source, filename="config.py")

    defined = set()

    def collect(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                collect(target)
        elif isinstance(node, ast.AnnAssign):
            collect(node.target)

    missing = [name for name in names if name not in defined]
    if missing:
        print(f"{group} missing constants: {', '.join(missing)}", file=sys.stderr)
    else:
        print(f"{group} constants present: {len(names)}")
    return missing


missing = []
missing.extend(check("V4", os.environ.get("V4_CONSTS", "")))
missing.extend(check("V5", os.environ.get("V5_CONSTS", "")))
if missing:
    sys.exit(1)
PY
echo "OK"

echo "[4/5] Checking module imports and required symbols..."
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

checks = [
    ("backtest_v4", "BacktestEngineV4"),
    ("backtest_v5", "BacktestEngineV5"),
    ("signals_v4", "generate_all_signals_v4"),
    ("signals_v5", "generate_all_signals_v5"),
    ("optimize_v4_optuna", "run_stage1"),
    ("optimize_v4_optuna", "run_stage2"),
    ("optimize_v5_optuna", "run_stage1"),
    ("optimize_v5_optuna", "run_stage2"),
]

errors = []
for module_name, symbol_name in checks:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        errors.append(f"import {module_name} failed: {exc}")
        continue
    if not hasattr(module, symbol_name):
        errors.append(f"missing symbol: {module_name}.{symbol_name}")

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    sys.exit(1)

print("All required imports and symbols are present.")
PY
echo "OK"

echo "[5/5] Running v4 document coverage gate..."
"${PYTHON_BIN}" scripts/check_v4_coverage.py
echo "OK"

echo "v4/v5 parity check passed."

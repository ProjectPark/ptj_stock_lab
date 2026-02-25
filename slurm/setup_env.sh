#!/bin/bash
# setup_env.sh — 서버에 Python venv 생성 + 패키지 설치
# data/.venv 에 설치하므로 clean-remote-code 후에도 유지됨
#
# Usage: ssh gigaflops-node54 "bash /mnt/giga/project/ptj_stock_lab/slurm/setup_env.sh"

set -e

REMOTE_DIR="/mnt/giga/project/ptj_stock_lab"
VENV_DIR="${REMOTE_DIR}/data/.venv"

echo "[setup_env] Creating venv at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"

echo "[setup_env] Activating venv ..."
source "${VENV_DIR}/bin/activate"

echo "[setup_env] Upgrading pip ..."
pip install --upgrade pip

echo "[setup_env] Installing packages ..."
pip install \
  numpy \
  pandas \
  optuna \
  matplotlib \
  scipy

echo "[setup_env] Done."
echo "[setup_env] venv: ${VENV_DIR}"
python --version
pip list --format=columns

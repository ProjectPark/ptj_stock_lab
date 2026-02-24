#!/bin/bash
#SBATCH --job-name=ptj-{{PROFILE}}
#SBATCH --account=taehyun
#SBATCH --partition={{PARTITION}}
#SBATCH --cpus-per-task={{SBATCH_CPUS}}
#SBATCH --mem={{SBATCH_MEM}}
#SBATCH --time={{SBATCH_TIME}}
#SBATCH --output={{REMOTE_DIR}}/slurm/logs/%j.out
#SBATCH --error={{REMOTE_DIR}}/slurm/logs/%j.err

set -e
cd {{RUN_DIR}}
source {{REMOTE_DIR}}/data/.venv/bin/activate

echo "[$(date)] Start: {{SCRIPT}} {{SCRIPT_ARGS}}"
python {{SCRIPT}} {{SCRIPT_ARGS}}
EXIT_CODE=$?
echo "[$(date)] Done. exit=$EXIT_CODE"
exit $EXIT_CODE

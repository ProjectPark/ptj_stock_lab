#!/bin/bash
#SBATCH --job-name=ptj-{{PROFILE}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --partition={{PARTITION}}
#SBATCH --cpus-per-task={{SBATCH_CPUS}}
#SBATCH --mem={{SBATCH_MEM}}
#SBATCH --time={{SBATCH_TIME}}
#SBATCH --output={{REMOTE_DIR}}/slurm/logs/%j.out
#SBATCH --error={{REMOTE_DIR}}/slurm/logs/%j.err

echo "[$(date)] Start: {{SCRIPT}} {{SCRIPT_ARGS}}"

srun \
  --container-image={{REMOTE_DIR}}/slurm/images/ptj_stock_lab.sqsh \
  --container-mounts={{RUN_DIR}}:/workspace,{{REMOTE_DIR}}/data:/workspace/data \
  --container-workdir=/workspace \
  python {{SCRIPT}} {{SCRIPT_ARGS}}

echo "[$(date)] Done. exit=$?"

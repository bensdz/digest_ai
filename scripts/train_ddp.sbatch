
#!/bin/bash
#SBATCH --job-name=digest-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Change these paths for your cluster.
cd ../
source .venv\Scripts\Activate.ps1
mkdir -p logs

# Use the first allocated node as the rendezvous/master node.
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"

srun --ntasks="$SLURM_NTASKS" \
     --ntasks-per-node=4 \
     --gpus-per-task=1 \
     python -u src/train_llm/train.py

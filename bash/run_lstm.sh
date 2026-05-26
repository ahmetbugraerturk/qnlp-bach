#!/bin/bash
#SBATCH --account=ai
#SBATCH --job-name=aerturk23_lstm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --output=output_classical/job_%j/output.log
#SBATCH --error=output_classical/job_%j/output.err

# ===========================================================================
# CLASSICAL LSTM TRAINING SCRIPT
# ===========================================================================

# Use the sbatch submission directory (if present) or current working dir.
# When SLURM copies the job script into /var/spool/slurm/... BASH_SOURCE points
# there. Prefer SLURM_SUBMIT_DIR which is the original submission cwd.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_SOURCE=${1:-bach}
TRAIN_SIZE=${2:-50}
VAL_SIZE=${3:-10}
TEST_SIZE=${4:-10}
DROPOUT=${5:-0.5}
SEED=${6:-42}

module load anaconda3/2025.06
module load cuda/12.1.1
module load cudnn/9.10.2
module load git/2.9.5

# Path to the Python executable (can be overridden in environment)
ENV=${ENV:-"/home/aerturk23/.conda/envs/bach_qnlp/bin/python"}

OUT="$PROJ_ROOT/output_classical/job_${SLURM_JOB_ID}"

# Create the output directory
mkdir -p "$OUT"

echo "========================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Start  : $(date)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

echo "--- Classical LSTM Script starting ---"

cd "$PROJ_ROOT" || exit 2

# To load a previously generated config file, you can use:
# --config_file "$OUT/previous_config.json" \

"$ENV" -u "$PROJ_ROOT/scripts/models/lstm.py" \
    --data_source    "$DATA_SOURCE" \
    --output_dir     "$OUT" \
    --num_train      "$TRAIN_SIZE" \
    --num_val        "$VAL_SIZE" \
    --num_test       "$TEST_SIZE" \
    --embed_dim      16 \
    --hidden_dim     32 \
    --epochs         100 \
    --learning_rate  0.01 \
    --weight_decay   1e-4 \
    --dropout        "$DROPOUT" \
    --threshold      0.5 \
    --batch_size     16 \
    --seed           "$SEED"

echo "========================================"
echo "End    : $(date)"
echo "Output : $OUT"
ls -lh "$OUT"
echo "========================================"
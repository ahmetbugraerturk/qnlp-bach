#!/bin/bash
#SBATCH --account=ai
#SBATCH --job-name=grid_manager
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=72:00:00 
#SBATCH --output=output_grid_master/job_%j/output.log
#SBATCH --error=output_grid_master/job_%j/output.err

# ==============================================================================
# UNIFIED GRID SEARCH SUBMISSION SCRIPT (WITH AUTO-WAIT / QUEUE MANAGER)
# ==============================================================================

# Limit for concurrent jobs to avoid hitting the system admin quota
# (From logs the quota appears to be ~50; we stay conservative and use 45)
MAX_JOBS=45

# Wait until there's room in the queue before submitting a job
wait_for_slot() {
    while true; do
        # Count current user's jobs
        CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
        
        if [ "$CURRENT_JOBS" -lt "$MAX_JOBS" ]; then
            # If the limit is not exceeded, break and submit the job
            break
        fi
        
        # Queue full: wait 60s and check again
        echo "[QUEUE FULL] Current jobs: $CURRENT_JOBS. Waiting 60s for free slots..."
        sleep 60
    done
}

# Define the global parameter grids
DATA_SOURCES=("bach" "sst2" "synthetic")
TRAIN_SIZES=(10 50 100 250 500) 
SEEDS=(42 100 2026)
LAYERS=(1 2)         
DROPOUTS=(0.2 0.5)   
VAL_SIZE=50
TEST_SIZE=100

echo "=========================================================="
echo "Starting Unified Grid Search Submission (Auto-Managed)"
echo "=========================================================="

# Use the sbatch submission directory (if present) or current working dir.
# This makes the grid manager submit jobs relative to where sbatch was invoked.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

for ds in "${DATA_SOURCES[@]}"; do
    for size in "${TRAIN_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # ------------------------------------------------------------------
            # 1. QNLP JOBS (Iterating over Layers)
            # ------------------------------------------------------------------
            for layer in "${LAYERS[@]}"; do
                # Check SLURM quota before submitting the job
                wait_for_slot
                
                echo "[QNLP] Submitting -> DS: $ds | Train: $size | Val: $VAL_SIZE | Test: $TEST_SIZE | Layers: $layer | Seed: $seed"
                sbatch "$PROJ_ROOT/bash/run_qnlp.sh" "$ds" "$size" "$VAL_SIZE" "$TEST_SIZE" "$layer" "$seed"
            done
            
            # ------------------------------------------------------------------
            # 2. LSTM JOBS (Iterating over Dropouts)
            # ------------------------------------------------------------------
            for drop in "${DROPOUTS[@]}"; do
                # Check SLURM quota before submitting the job
                wait_for_slot
                
                echo "[LSTM] Submitting -> DS: $ds | Train: $size | Val: $VAL_SIZE | Test: $TEST_SIZE | Dropout: $drop | Seed: $seed"
                sbatch "$PROJ_ROOT/bash/run_lstm.sh" "$ds" "$size" "$VAL_SIZE" "$TEST_SIZE" "$drop" "$seed"
            done
            
        done
    done
done

echo "=========================================================="
echo "All jobs successfully submitted to the queue!"
echo "=========================================================="
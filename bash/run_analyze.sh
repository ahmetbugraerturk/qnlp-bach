#!/bin/bash
#SBATCH --account=ai
#SBATCH --job-name=analyze_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=72:00:00 
#SBATCH --output=output_analyze/job_%j/output.log
#SBATCH --error=output_analyze/job_%j/output.err

# ==============================================================================
# GRID SEARCH ANALYSIS SCRIPT
# ==============================================================================

echo "=========================================================="
echo "Starting Grid Search Result Analysis"
echo "=========================================================="


# Use the sbatch submission directory (if present) or current working dir.
# Prefer SLURM_SUBMIT_DIR because SLURM copies the script into /var/spool/slurm
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Path to your specific python environment (can be overridden)
ENV=${ENV:-"/home/aerturk23/.conda/envs/bach_qnlp/bin/python"}

# Run the python script from repository root so relative paths resolve
cd "$PROJ_ROOT" || exit 2

set -euo pipefail

# Create a job-specific output directory under PROJ_ROOT/output_analyze
JOB_ID="${SLURM_JOB_ID:-local_$(date +%s)}"
OUT="$PROJ_ROOT/output_analyze/job_${JOB_ID}"
LOG_DIR="$OUT/logs"
OUT="$PROJ_ROOT/output_analyze/job_${JOB_ID}/results"

mkdir -p "$OUT"
mkdir -p "$LOG_DIR"

echo "[1/4] Running timing_and_scaling.py (outputs -> $OUT)"
# Run timing script with working dir OUT so timing_results.csv is placed there
(cd "$OUT" && "$ENV" -u "$PROJ_ROOT/scripts/analyze/timing_and_scaling.py") > "$LOG_DIR/timing.log" 2>&1 || {
    echo "timing_and_scaling failed: see $LOG_DIR/timing.log"; exit 1;
}
echo "  timing saved -> $OUT/timing_results.csv"

echo "[2/4] Generating diagrams (outputs -> $OUT)"
# Run diagrams generator with working dir OUT so it writes outputs under OUT/
(cd "$OUT" && "$ENV" -u "$PROJ_ROOT/scripts/analyze/generate_diagrams.py") > "$LOG_DIR/diagrams.log" 2>&1 || {
    echo "generate_diagrams failed: see $LOG_DIR/diagrams.log"; exit 1;
}
echo "  diagrams saved under -> $OUT/ (see overleaf_project_ folder)"

echo "[3/4] Running comprehensive grid analysis (outputs -> $OUT)"
# Run analyzer with working dir OUT so grid_analysis_results is created under OUT/
(cd "$OUT" && "$ENV" -u "$PROJ_ROOT/scripts/analyze/analyze_grid.py") > "$LOG_DIR/analyze.log" 2>&1 || {
    echo "analyze_grid failed: see $LOG_DIR/analyze.log"; exit 1;
}
echo "  analysis csvs -> $OUT/grid_analysis_results/"

echo "[4/4] Generating aggregated plots (outputs -> $OUT/report_figures)"
"$ENV" -u "$PROJ_ROOT/scripts/analyze/generate_plots.py" \
    --lstm_dir  "$PROJ_ROOT/output_classical" \
    --qnlp_dir  "$PROJ_ROOT/output_qnlp" \
    --out_dir   "$OUT/report_figures" > "$LOG_DIR/plots.log" 2>&1 || {
    echo "generate_plots failed: see $LOG_DIR/plots.log"; exit 1;
}
echo "  plots saved -> $OUT/report_figures"

echo "=========================================================="
echo "Analysis complete. Check the generated .png files."
echo "=========================================================="
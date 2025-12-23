#!/bin/bash
#SBATCH --job-name=refractory
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --array=0-51                  # 52 tasks (13 left * 4 right)
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Create logs directory if it doesn't exist
kinit simon.kern@ZI.LOCAL -k -t ~/simon.kern.keytab
set -euo pipefail

# Define log file paths based on SBATCH directives
LOG_OUT="/zi/home/simon.kern/slurmlogs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"
LOG_ERR="/zi/home/simon.kern/slurmlogs/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}.err"

# Time to wait for system to flush logs before sending (seconds)
WAIT_TIME=5

notify() {
  local status="$1" rc="${2:-0}"
  local base_message="${status}: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}) rc=${rc} host=${HOSTNAME}"

  # Give the system time to flush the logs, which is critical for ERR trap
  sleep "$WAIT_TIME"

  # 1. Send the main status message
  telegram-send "${base_message}" &>/dev/null || true

  # 2. Send the .out log file
  if [[ -f "$LOG_OUT" ]]; then
    telegram-send --file "$LOG_OUT" --caption "SLURM Output Log: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" &>/dev/null || true
  fi

  # 3. Send the .err log file
  if [[ -f "$LOG_ERR" ]]; then
    telegram-send --file "$LOG_ERR" --caption "SLURM Error Log: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" &>/dev/null || true
  fi
}

# Notify on any error (python, missing files, etc.)
trap 'rc=$?; notify "FAIL" "$rc"; exit "$rc"' ERR

# --- UPDATED VALUE ARRAYS ---
LEFT_VALUES=(0 1 2 3 4 5 6 7 8 9 10 15 30)
RIGHT_VALUES=(0 8 15 30)

# Get lengths
LEN_R=${#RIGHT_VALUES[@]}

# --- UPDATED INDEX CALCULATION ---
# We map the 1D task ID to a 2D matrix (Left x Right)
# Integer division by the length of the inner loop (Right) gives the Left Index
IDX_L=$(( SLURM_ARRAY_TASK_ID / LEN_R ))
# Modulo by the length of the inner loop (Right) gives the Right Index
IDX_R=$(( SLURM_ARRAY_TASK_ID % LEN_R ))

# Get the actual values
VAL_LEFT=${LEFT_VALUES[$IDX_L]}
VAL_RIGHT=${RIGHT_VALUES[$IDX_R]}

echo "Running simulation with LEFT=$VAL_LEFT and RIGHT=$VAL_RIGHT (Array ID: $SLURM_ARRAY_TASK_ID)"

# Run the python script
python tmp_check_refractory.py --left $VAL_LEFT --right $VAL_RIGHT

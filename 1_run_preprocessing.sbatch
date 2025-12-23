#!/bin/bash
#SBATCH --job-name=refractory
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --array=1-31
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00


SUBJ_ID=$SLURM_ARRAY_TASK_ID
echo "Running participant ${SUBJ_ID}"

python tmp_check_refractory.py "$SUBJ_ID"

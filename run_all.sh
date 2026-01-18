#!/bin/sh
#SBATCH -c 4               # Request 4 CPU cores
#SBATCH -t 0-48:00          # Runtime in D-HH:MM, minimum of 10 mins (this requests 48 hours)
#SBATCH --partition=gpmoo-b # Partition to submit to
#SBATCH --mem=16G           # Request 32G of memory
#SBATCH --array=1-11      # 23 simulation types
#SBATCH -o output/gcm_tests/output_nn_%j.out  # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e output/gcm_tests/progress_nn_%j.err  # File to which STDERR will be written (%j inserts jobid)

~/.conda/envs/etenv/bin/python3.13 simulations.py nonlinear $SLURM_ARRAY_TASK_ID
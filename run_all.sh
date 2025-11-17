#!/bin/sh
#SBATCH -c 4                # Request 2 CPU core
#SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins (this requests 2 hours)
#SBATCH --partition=gpmoo-b # Partition to submit to
#SBATCH --mem=32G           # Request 32G of memory
#SBATCH --array=1-3      # 23 simulation types
#SBATCH -o output/nn/output_backdoor_25_%j.out  # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e output/nn/progress_backdoor_25_%j.err  # File to which STDERR will be written (%j inserts jobid)

~/.conda/envs/etenv/bin/python3.13 simulations.py nonlinear $SLURM_ARRAY_TASK_ID

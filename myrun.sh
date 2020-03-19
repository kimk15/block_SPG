#!/bin/sh
# SBATCH --job-name=TIMED
# SBATCH --t 06:00:00
# SBATCH -D /gpfs/u/home/RLML/RLMLkmkv/block_SPG

srun -N 1 -t 360 python time_ada_ill_main.py &
srun -N 1 -t 360 python time_bras_ill_main.py &

wait
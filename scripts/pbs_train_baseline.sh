#!/bin/bash
#PBS -N ppa_human_baseline
#PBS -q ais_gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /oceanstor/home/e1553200/graph-gpt-main/exp/logs/baseline_train.log

cd /oceanstor/home/e1553200/graph-gpt-main
source /oceanstor/home/e1553200/graphgpt311/bin/activate

# Fix open files limit for cached subgraph loading
ulimit -n 65536

# Verify cached subgraphs exist
if [ ! -d "./data/OGB/ogbl_ppa_human/cached_subgraphs" ]; then
    echo "ERROR: Cached subgraphs not found. Run precompute first."
    exit 1
fi

echo "Starting baseline training at $(date)"
echo "Hostname: $(hostname)"
echo "ulimit -n: $(ulimit -n)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash examples/edge_lvl/ppa_human_baseline.sh

echo "Training finished at $(date)"

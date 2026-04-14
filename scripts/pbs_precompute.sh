#!/bin/bash
#PBS -N precompute_subgraphs
#PBS -q ais_gpu
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /oceanstor/home/e1553200/graph-gpt-main/exp/logs/precompute.log

cd /oceanstor/home/e1553200/graph-gpt-main
source /oceanstor/home/e1553200/graphgpt311/bin/activate

echo "Starting subgraph precomputation at $(date)"
echo "Hostname: $(hostname)"

python scripts/precompute_subgraphs.py \
    --data_dir ./data/OGB \
    --data_path ogbl_ppa_human \
    --depth_neighbors "[[1,14]]" \
    --neg_ratio 1 \
    --neg_epochs 6 \
    --chunk_size 10000 \
    --seed 42

echo "Precomputation finished at $(date)"
echo "Cache size:"
du -sh ./data/OGB/ogbl_ppa_human/cached_subgraphs/

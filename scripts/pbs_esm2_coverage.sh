#!/bin/bash
#PBS -N esm2_coverage
#PBS -q ais_gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /oceanstor/home/e1553200/graph-gpt-main/exp/logs/esm2_coverage.log

cd /oceanstor/home/e1553200/graph-gpt-main
source /oceanstor/home/e1553200/graphgpt311/bin/activate

echo "========================================="
echo "ESM-2 Embedding for Ensembl Sequences"
echo "Sequences: ~9203 (pre-fetched from Ensembl)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "========================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python new/3_embed_ensembl_seqs.py \
  --data_dir data/OGB/ogbl_ppa_human \
  --model_name esm2_t33_650M_UR50D \
  --layer 33 \
  --pooling mean \
  --batch_size 2

echo "========================================="
echo "End: $(date)"
echo "========================================="

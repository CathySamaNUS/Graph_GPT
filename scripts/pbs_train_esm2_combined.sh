#!/bin/bash
#PBS -N esm2_combined
#PBS -q ais_gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /oceanstor/home/e1553200/graph-gpt-main/exp/logs/esm2_combined.log

cd /oceanstor/home/e1553200/graph-gpt-main
source /oceanstor/home/e1553200/graphgpt311/bin/activate
export PYTHONUNBUFFERED=1
bash examples/edge_lvl/ppa_human_esm2_combined.sh

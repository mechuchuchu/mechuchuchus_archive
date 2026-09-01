#!/usr/bin/env bash
# 3-arm x 2-arch x 3-seed 스윕
set -e

for seed in 0 1 2; do
  for arch in resnet18 mlp; do
    for pt in none noise real; do
      echo "=============================================="
      echo "arch=$arch pretrain=$pt seed=$seed"
      echo "=============================================="
      python probe_experiment.py \
        --arch "$arch" \
        --pretrain "$pt" \
        --head linear \
        --seed "$seed" \
        --out results.jsonl
    done
  done
done

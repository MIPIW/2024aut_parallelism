#!/bin/bash

# 기본값 설정
: ${NODES:=4}  # 사용할 노드 수 (기본값: 4)

# SLURM Job 실행
salloc -N $NODES -p class1 --exclusive --gres=gpu:4 \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --physcpubind 0-31 \
  ./main "$@"

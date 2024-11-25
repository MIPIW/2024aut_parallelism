#!/bin/bash
# 변경가능
: ${NODES:=4}

# salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
# 	mpirun --bind-to none -mca btl ^openib -npernode 1 \
# 		--oversubscribe -quiet \
# 		./main $@

salloc -N 4 --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		./main $@


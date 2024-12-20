#!/bin/bash
# 변경가능
: ${NODES:=4}

# salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
# 	mpirun --bind-to none -mca btl ^openib -npernode 4 \
# 		--oversubscribe -quiet \
# 		./main $@

# salloc -N 4 --partition class1 --exclusive --gres=gpu:4   \
# 	mpirun --bind-to none -mca btl ^openib -npernode 4 \
# 		--oversubscribe -quiet \
# 		./main $@ -n 4096

# salloc -N 4 --partition class1 --exclusive --gres=gpu:4 --cpus-per-task=8 --mem=64G \
# 	mpirun --bind-to core --map-by ppr:4:node --mca pml ucx -x UCX_TLS=rc_x,sm,cuda_copy,cuda_ipc \
# 	./main $@ -n 4096


salloc -N 4 --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 4 \
	--oversubscribe -quiet \
	./main $@ -n 4096
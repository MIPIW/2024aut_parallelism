#!/bin/bash
# 변경가능
: ${NODES:=4}


salloc -N 4 --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 4 \
	--oversubscribe -quiet \
	 ./main $@ -n 16384


# TMPDIR=~ srun -N 4 --partition class1 --exclusive --gres=gpu:4   \
# 	nsys profile --stats=true --force-overwrite=true mpirun --bind-to none -mca btl ^openib -npernode 4 \
# 	--oversubscribe -quiet \
# 	 ./main $@ -n 4096


# TEMPDIR=~ salloc -N 4 --partition class1 --exclusive --gres=gpu:4 \
# 	mpirun --bind-to none -mca btl ^openib -npernode 4 \
# 	--oversubscribe -quiet \
# 	nsys profile -o profile_output -t cuda,osrt,nvtx \
# 	./main "$@" -n 4096
#define BLOCK_SIZE 32
#define C_SIZE 32

__kernel void sgemm(__global const float *A, 
                               __global const float *B, 
                               __global float *C, 
                               int M, int N, int K) {
    // Compute work-item's global row and column

    const int locRow = get_local_id(0);
    const int locCol = get_local_id(1);

    const int glbRow = get_global_id(0);
    const int glbCol = get_global_id(1);
    
    const int locBSRow = get_local_size(0);
    const int glbWSRow = get_global_size(0);

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];


    float temp[C_SIZE];
    for (int j=0; j<C_SIZE; ++j){
        temp[j] = 0.0;
    }
    
    const int NUM_TILES = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < NUM_TILES; ++t){
        

        for(int j=0; j<C_SIZE; ++j){
                 
            // 왠지 모르겠는데 2D 행렬로 하면 더 빠름
            if ((glbRow + j * glbWSRow) < M && (BLOCK_SIZE * t + locCol) < K) {
                Asub[locRow + j * locBSRow][locCol] = A[(glbRow + j * glbWSRow) * K + (BLOCK_SIZE * t + locCol)];
            } else {
                Asub[locRow + j * locBSRow][locCol] = 0.0; 
            }
        }

        for(int k=0; k < BLOCK_SIZE; ++k){

            if ((BLOCK_SIZE * t + k) < K && glbCol < N) {
                Bsub[k][locCol] = B[(BLOCK_SIZE * t + k) * N + glbCol];
            } else {
                Bsub[k][locCol] = 0.0; 
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0 ; k < BLOCK_SIZE ; ++k){
            for (int j = 0; j < C_SIZE; ++j) {
                int locIdxRowA = locRow + j * locBSRow;

                temp[j] += Asub[locIdxRowA][k] * Bsub[k][locCol];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }
    for (int j = 0; j < C_SIZE; ++j) {
        int glbIdxRow = glbRow + j * glbWSRow;

        if (glbIdxRow < M && glbCol < N) {
            C[glbIdxRow * N + glbCol] = temp[j];
        }
    }
    
}


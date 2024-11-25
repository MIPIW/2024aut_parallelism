#define BLOCK_SIZE 16
#define C_SIZE 8

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
        const int t_col_start = BLOCK_SIZE * t;
        const int t_row_start = BLOCK_SIZE * t;
        

        for(int j=0; j<C_SIZE; ++j){
            
            int locIdxRow = locRow + j * locBSRow;
            int glbIdxRow = glbRow + j * glbWSRow;
            int locIdxCol = locCol;
            int glbIdxCol = t_col_start + locCol;
            
            // 왠지 모르겠는데 2D 행렬로 하면 더 빠름
            if (glbIdxRow < M && glbIdxCol < K) {
                Asub[locIdxRow][locIdxCol] = A[glbIdxRow * K + glbIdxCol];
            } else {
                Asub[locIdxRow][locIdxCol] = 0.0; 
            }
        }

        for(int k=0; k < BLOCK_SIZE; ++k){
            int locIdxRow = k;
            int glbIdxRow = t_row_start + k;
            int locIdxCol = locCol;
            int glbIdxCol = glbCol;            

            if (glbIdxRow < K && glbIdxCol < N) {
                Bsub[locIdxRow][locIdxCol] = B[glbIdxRow * N + glbIdxCol];
            } else {
                Bsub[locIdxRow][locIdxCol] = 0.0; 
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


#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


void matmul(float *A, float *B, float *C, int M, int N, int K,
            int num_threads) {

    
	// TODO : FILL IN HERE
    omp_set_num_threads(num_threads);

    
    // 8(instruction) * 2(block_n) * 2(block_k) = 32(total cache memory size)(num_blocks = 32/num_per_line)
    int block_N = 128; 
    int block_K = 32;

    #pragma omp parallel
    {
        int j, k, ii, jj, kk;
        float temp;
        int index_b, index_c;
        #pragma omp for schedule(guided)
        for (ii = 0; ii < M; ++ii){
            
            for(kk = 0; kk < K; kk += block_K){
                for (jj = 0; jj < N; jj += block_N){

                    int max_or_block_N = (jj + block_N <= N) ? block_N : (N - jj);
                    int max_or_block_K = (kk + block_K <= K) ? block_K : (K - kk);
            
                    for(k = 0; k < max_or_block_K; ++k){
                        temp = A[ii * K + kk + k];
                        
                        index_c = ii * N + jj;
                        index_b = (kk + k) * N + jj;    

                        #pragma omp simd
                        for (j = 0; j < max_or_block_N; ++j){
                            C[index_c+j] += temp * B[index_b + j];
                        }

                        }
                            
                    }
                }
            }
        }    
}



    

    


    
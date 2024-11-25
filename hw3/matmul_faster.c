
#define _GNU_SOURCE
#include "util.h"
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
    omp_set_num_threads(num_threads);
    


    int block_M = 64; // 블록 크기
    int block_N = 64;
    int block_K = 16;

    // 행렬 곱셈 수행
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < M; ii += block_M) {
        for (int jj = 0; jj < N; jj += block_N) {
            for (int kk = 0; kk < K; kk += block_K) {
                
                // 남은 부분을 고려하여 블록의 크기 결정
                int max_i = (ii + block_M < M) ? (ii + block_M) : M;
                int max_j = (jj + block_N < N) ? (jj + block_N) : N;
                int max_k = (kk + block_K < K) ? (kk + block_K) : K;

                // 블록 내에서 행렬 곱셈 수행
                for (int i = ii; i < max_i; ++i) {
                    for (int k = kk; k < max_k; ++k) {
                        float temp = A[i * K + k]; // A 행렬의 특정 행 추출
                        for (int j = jj; j < max_j; ++j) {
                            // C[i * N + j]에 접근할 때 데이터 레이싱을 피하도록 독립적으로 계산
                            C[i * N + j] += temp * B[k * N + j]; // 행렬 곱셈 및 누적
                        }
                    }
                }
            }
        }
    }
}

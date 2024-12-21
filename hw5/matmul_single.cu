#include "matmul_single.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }


__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // Shared memory for storing tiles of A and B
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate thread row and column within the output matrix
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over tiles of A and B required for C[row, col]
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load elements into shared memory (if within bounds)
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to the output matrix (if within bounds)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// __global__ void matmul_kernel_vec4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
//     // Shared memory for tiles
//     __shared__ float4 shared_A[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float4 shared_B[BLOCK_SIZE][BLOCK_SIZE];

//     // Thread indices
//     int tx = threadIdx.x; // Local thread index in block (x)
//     int ty = threadIdx.y; // Local thread index in block (y)

//     int row = blockIdx.y * BLOCK_SIZE + ty; // Row index of the matrix C
//     int col = blockIdx.x * BLOCK_SIZE + tx; // Column index of the matrix C

//     float4 sum = make_float4(0, 0, 0, 0); // Accumulate the result as float4

//     for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
//         // Load a tile of A into shared memory
//         if (row < M && (tile * BLOCK_SIZE + tx) < K) {
//             shared_A[ty][tx] = *(reinterpret_cast<const float4*>(&A[row * K + tile * BLOCK_SIZE + tx * 4]));
//         } else {
//             shared_A[ty][tx] = make_float4(0, 0, 0, 0);
//         }

//         // Load a tile of B into shared memory
//         if (col < N && (tile * BLOCK_SIZE + ty) < K) {
//             shared_B[ty][tx] = *(reinterpret_cast<const float4*>(&B[(tile * BLOCK_SIZE + ty) * N + col]));
//         } else {
//             shared_B[ty][tx] = make_float4(0, 0, 0, 0);
//         }

//         __syncthreads();

//         // Compute the dot product for the current tile
//         for (int k = 0; k < BLOCK_SIZE; ++k) {
//             sum.x += shared_A[ty][k].x * shared_B[k][tx].x;
//             sum.y += shared_A[ty][k].y * shared_B[k][tx].y;
//             sum.z += shared_A[ty][k].z * shared_B[k][tx].z;
//             sum.w += shared_A[ty][k].w * shared_B[k][tx].w;
//         }

//         __syncthreads();
//     }

//     // Write the result to the output matrix
//     if (row < M && col < N) {
//         float* c_ptr = &C[row * N + col];
//         *reinterpret_cast<float4*>(c_ptr) = sum;
//     }
// }




// Array of device (GPU) pointers
// static float4 *a_d;
// static float4 *b_d;
// static float4 *c_d;

static float *a_d;
static float *b_d;
static float *c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  CUDA_CALL(cudaMemcpy(a_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(b_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // // // Launch kernel on every GPU
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K);

  // dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // matmul_kernel_vec4<<<blocks, threads>>>(a_d, b_d, c_d, M, N, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
  }

  CUDA_CALL(cudaDeviceSynchronize());

  // Download C matrix from GPUs
  CUDA_CALL(cudaMemcpy(C, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CUDA_CALL(cudaDeviceSynchronize());
}

void matmul_initialize(int M, int N, int K) {
  
  size_t a_size = M * (K / 4) * sizeof(float4); // M x (K / 4) float4 elements
  size_t b_size = (K / 4) * (N / 4) * sizeof(float4); // (K / 4) x (N / 4) float4 elements
  size_t c_size = M * (N / 4) * sizeof(float4); // M x (N / 4) float4 elements
  
  int num_devices;
  // Only root process do something
  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Allocate device memory 

  CUDA_CALL(cudaMalloc(&a_d, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc(&b_d, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc(&c_d, M * N * sizeof(float)));
}

void matmul_finalize() {

  // Free GPU memory
  CUDA_CALL(cudaFree(a_d));
  CUDA_CALL(cudaFree(b_d));
  CUDA_CALL(cudaFree(c_d));
}

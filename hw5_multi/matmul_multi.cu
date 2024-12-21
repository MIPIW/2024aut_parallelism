#include "matmul_multi.h"
#include "util.h"

#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_SIZE 32
#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define MAX_NUM_GPU 4
int num_devices = 0;


__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  // Shared memory for A and B tiles
  __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

  // Thread indices
  int tx = threadIdx.x;  // Thread within a block
  int ty = threadIdx.y;

  // Global indices
  int row = blockIdx.x * TILE_SIZE + tx;  // Row index of C
  int col = blockIdx.y * TILE_SIZE + ty;  // Column index of C

  // Initialize accumulator for C element
  float sum = 0.0f;

  // Loop over tiles
  for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
    // Load A and B tiles into shared memory (coalesced memory access)
    if (row < M && (tile_idx * TILE_SIZE + ty) < K) {
      shared_A[tx][ty] = A[row * K + tile_idx * TILE_SIZE + ty];
    } else {
      shared_A[tx][ty] = 0.0f;
    }

    if (col < N && (tile_idx * TILE_SIZE + tx) < K) {
      shared_B[tx][ty] = B[(tile_idx * TILE_SIZE + tx) * N + col];
    } else {
      shared_B[tx][ty] = 0.0f;
    }

    __syncthreads();  // Ensure all threads have loaded their tiles

    // Perform multiplication for the current tile
    #pragma unroll  // Loop unrolling for better performance
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += shared_A[tx][k] * shared_B[k][ty];
    }

    __syncthreads();  // Wait for all threads to complete computation
  }

  // Store the result in the output matrix
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}



// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  // Upload A and B matrix to every GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(a_d[i], A + Mbegin[i] * K,
                         (Mend[i] - Mbegin[i]) * K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(
        cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Launch kernel on every GPU
  for (int i = 0; i < num_devices; i++) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((Mend[i] - Mbegin[i])/TILE_SIZE, N/TILE_SIZE, 1);

    CUDA_CALL(cudaSetDevice(i));
    matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
  }

  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }

  // Download C matrix from GPUs
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaMemcpy(C + Mbegin[i] * N, c_d[i],
                         (Mend[i] - Mbegin[i]) * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaDeviceSynchronize());
  }
}

void matmul_initialize(int M, int N, int K) {

  CUDA_CALL(cudaGetDeviceCount(&num_devices));

  printf("Using %d devices\n", num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("GPU %d: %s\n", i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  // Setup problem size for each GPU
  for (int i = 0; i < num_devices; i++) {
    Mbegin[i] = (M / num_devices) * i;
    Mend[i] = (M / num_devices) * (i + 1);
  }
  Mend[num_devices - 1] = M;

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}

void matmul_finalize() {

  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
  }
}

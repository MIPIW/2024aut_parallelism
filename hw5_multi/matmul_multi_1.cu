#include "matmul_multi.h"
#include "util.h"
#include <pthread.h>
#include <stdio.h>
#include <cuda_runtime.h>

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
#define TILE_SIZE 16
int num_devices = 0;

typedef struct {
    int device_id;
    const float *A;
    const float *B;
    float *C;
    int M, N, K;
    int Mbegin, Mend;
} ThreadData;


// __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
//                               int K) {
//   int i = blockDim.x * blockIdx.x + threadIdx.x;
//   int j = blockDim.y * blockIdx.y + threadIdx.y;
//   if (i >= M || j >= N)
//     return;

//   C[i * N + j] = 0;
//   for (int k = 0; k < K; ++k) {
//     C[i * N + j] += A[i * K + k] * B[k * N + j];
//   }
// }


// __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
//                               int K) {
//     int row = blockIdx.x;
//     int col = blockIdx.y;

//     // Ensure we are within bounds
//     if (row >= M || col >= N) return;

//     // Compute the dot product of row from A and column from B
//     float sum = 0.0f;
//     for (int k = 0; k < K; k++) {
//         sum += A[row * K + k] * B[k * N + col];
//     }

//     // Write the computed value to the output matrix
//     C[row * N + col] = sum;
// }

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // Shared memory for storing tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate thread row and column within the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Loop over tiles of A and B required for C[row, col]
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load elements into shared memory (if within bounds)
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < TILE_SIZE; k++) {
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

// Array of device (GPU) pointers
// static float *a_d[MAX_NUM_GPU];
// static float *b_d[MAX_NUM_GPU];
// static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];

void *threaded_matmul(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    // Set GPU for this thread
    CUDA_CALL(cudaSetDevice(data->device_id));

    // Allocate memory on the GPU
    float *a_d, *b_d, *c_d;
    CUDA_CALL(cudaMalloc(&a_d, (data->Mend - data->Mbegin) * data->K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d, data->K * data->N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d, (data->Mend - data->Mbegin) * data->N * sizeof(float)));

    // Copy matrices to the GPU
    CUDA_CALL(cudaMemcpy(a_d, data->A + data->Mbegin * data->K,
                         (data->Mend - data->Mbegin) * data->K * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(b_d, data->B, data->K * data->N * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((data->N + TILE_SIZE - 1) / TILE_SIZE,
                 (data->Mend - data->Mbegin + TILE_SIZE - 1) / TILE_SIZE,
                 1);

    // Launch the kernel
    matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, data->M, data->N, data->K);

    // Synchronize GPU
    CUDA_CALL(cudaDeviceSynchronize());

    // Copy the result matrix back to host memory
    CUDA_CALL(cudaMemcpy(data->C + data->Mbegin * data->N, c_d,
                         (data->Mend - data->Mbegin) * data->N * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CALL(cudaFree(a_d));
    CUDA_CALL(cudaFree(b_d));
    CUDA_CALL(cudaFree(c_d));

    return NULL;
}

// void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

//   // Upload A and B matrix to every GPU
//   for (int i = 0; i < num_devices; i++) {
//     CUDA_CALL(cudaMemcpy(a_d[i], A + Mbegin[i] * K,
//                          (Mend[i] - Mbegin[i]) * K * sizeof(float),
//                          cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(b_d[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice));
//   }

//   // Launch kernel on every GPU
//   for (int i = 0; i < num_devices; i++) {
//     // dim3 blockDim(1, 1, 1);
//     // dim3 gridDim(Mend[i] - Mbegin[i], N, 1);

//     dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
//     dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (Mend[i] - Mbegin[i] + TILE_SIZE - 1) / TILE_SIZE, 1);

//     CUDA_CALL(cudaSetDevice(i));
//     matmul_kernel<<<gridDim, blockDim>>>(a_d[i], b_d[i], c_d[i], M, N, K);
//   }

//   for (int i = 0; i < num_devices; i++) {
//     CUDA_CALL(cudaDeviceSynchronize());
//   }

//   // Download C matrix from GPUs
//   for (int i = 0; i < num_devices; i++) {
//     CUDA_CALL(cudaMemcpy(C + Mbegin[i] * N, c_d[i],
//                          (Mend[i] - Mbegin[i]) * N * sizeof(float),
//                          cudaMemcpyDeviceToHost));
//   }

//   // DO NOT REMOVE; NEEDED FOR TIME MEASURE
//   for (int i = 0; i < num_devices; i++) {
//     CUDA_CALL(cudaDeviceSynchronize());
//   }
// }

void * temp(void *arg){
  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    pthread_t threads[MAX_NUM_GPU];
    ThreadData thread_data[MAX_NUM_GPU];

    // Create and launch threads
    for (int i = 0; i < num_devices; i++) {
        thread_data[i].device_id = i;
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].M = M;
        thread_data[i].N = N;
        thread_data[i].K = K;
        thread_data[i].Mbegin = Mbegin[i];
        thread_data[i].Mend = Mend[i];

        pthread_create(&threads[i], NULL, threaded_matmul, &thread_data[i]);
        // pthread_create(&threads[i], NULL, temp, &thread_data[i]);

    }

    // Join threads
    for (int i = 0; i < num_devices; i++) {
        pthread_join(threads[i], NULL);
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
  // for (int i = 0; i < num_devices; i++) {
  //   CUDA_CALL(cudaSetDevice(i));
  //   CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
  //   CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
  //   CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  // }
}

void matmul_finalize() {

  // Free all GPU memory
  // for (int i = 0; i < num_devices; i++) {
  //   CUDA_CALL(cudaFree(a_d[i]));
  //   CUDA_CALL(cudaFree(b_d[i]));
  //   CUDA_CALL(cudaFree(c_d[i]));
  // }
}

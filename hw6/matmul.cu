#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 32
#define NUM_GPUS 4  // 4 GPUs per node

// Global variables for memory reuse
float *d_A[NUM_GPUS], *d_B[NUM_GPUS], *d_C[NUM_GPUS];
dim3 block, grid;
cudaStream_t streams[NUM_GPUS];

// Global pinned memory pointers
float *local_A, *host_B, *host_C;

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
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

        __syncthreads();
    
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialization function: allocate memory and create streams for each GPU
void matmul_initialize(int M, int N, int K) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_gpu = M / (NUM_GPUS * 4);  // Total 4 nodes, each with 4 GPUs
    int local_rows_per_node = rows_per_gpu * NUM_GPUS;

    // Allocate pinned memory for host buffers
    cudaMallocHost((void **)&local_A, local_rows_per_node * K * sizeof(float));
    cudaMallocHost((void **)&host_B, K * N * sizeof(float));
    cudaMallocHost((void **)&host_C, local_rows_per_node * N * sizeof(float));

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);

        size_t size_A = rows_per_gpu * K * sizeof(float);
        size_t size_B = K * N * sizeof(float);
        size_t size_C = rows_per_gpu * N * sizeof(float);

        cudaMalloc((void **)&d_A[i], size_A);
        cudaMalloc((void **)&d_B[i], size_B);
        cudaMalloc((void **)&d_C[i], size_C);

        block = dim3(BLOCK_SIZE, BLOCK_SIZE);
        grid = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
}

// Matrix multiplication function using asynchronous API for multiple GPUs
void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_gpu = M / (NUM_GPUS * 4);
    int local_rows_per_node = rows_per_gpu * NUM_GPUS;

    // Broadcast B to all nodes
    MPI_Bcast((void *)B, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    memcpy(host_B, B, K * N * sizeof(float));

    // Distribute A among nodes
    MPI_Scatter(A, local_rows_per_node * K, MPI_FLOAT,
                local_A, local_rows_per_node * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Copy B to all GPUs asynchronously
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        size_t size_B = K * N * sizeof(float);
        cudaMemcpyAsync(d_B[i], host_B, size_B, cudaMemcpyHostToDevice, streams[i]);
    }

    // Divide A and perform computation on each GPU
    #pragma omp parallel for schedule(static) num_threads(NUM_GPUS)
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        size_t size_A = rows_per_gpu * K * sizeof(float);
        size_t size_C = rows_per_gpu * N * sizeof(float);

        // Copy a chunk of A to each GPU asynchronously
        cudaMemcpyAsync(d_A[i], local_A + i * rows_per_gpu * K, size_A, cudaMemcpyHostToDevice, streams[i]);

        // Launch the kernel asynchronously
        matmul_kernel<<<grid, block, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], rows_per_gpu, N, K);

        // Copy result back to host asynchronously
        cudaMemcpyAsync(host_C + i * rows_per_gpu * N, d_C[i], size_C, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all GPUs to finish
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Gather results from all nodes
    MPI_Gather(host_C, local_rows_per_node * N, MPI_FLOAT,
               rank == 0 ? C : NULL, local_rows_per_node * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// Finalization function: free memory and destroy streams
void matmul_finalize() {
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free pinned host memory
    cudaFreeHost(local_A);
    cudaFreeHost(host_B);
    cudaFreeHost(host_C);

    cudaDeviceReset();
}

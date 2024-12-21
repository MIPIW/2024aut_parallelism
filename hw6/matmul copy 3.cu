#include "matmul.h"
#include "util.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Define constants
#define MAX_GPUS 4
#define TILE_SIZE 32  // 유지: 32으로 설정
#define NSTREAM 16     // 각 GPU당 스트림 수

// Macro for error checking (CUDA)
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)
#endif

// Macro for error checking (MPI)
#define CHECK_MPI(call)                                                      \
    do {                                                                     \
        int err = call;                                                       \
        if (err != MPI_SUCCESS) {                                             \
            char err_string[MPI_MAX_ERROR_STRING];                           \
            int err_len;                                                      \
            MPI_Error_string(err, err_string, &err_len);                     \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    err_string);                                              \
            MPI_Abort(MPI_COMM_WORLD, err);                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                    \
    } while (0)

// Structure to hold per-GPU resources
typedef struct {
    float *d_A;
    float *d_B;
    float *d_C;
    cudaStream_t streams[NSTREAM]; // 16 스트림
} GPUResources;

// Global variables
static int num_gpus = 0;
static GPUResources gpu_resources[MAX_GPUS];
static float *h_B_buffer = NULL; // Host buffer for B
static bool b_broadcasted = false; // Flag to ensure B is broadcasted once

// Matrix multiplication kernel using tiling and loop unrolling
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    // Calculate the row and column index of the C matrix element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float accum = 0.0f;

    // Shared memory for A and B tiles
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load data into shared memory
        if ((tile * TILE_SIZE + threadIdx.x) < K && row < M) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((tile * TILE_SIZE + threadIdx.y) < K && col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform multiplication with loop unrolling and FMA
        #pragma unroll 32
        for (int k = 0; k < TILE_SIZE; k++) {
            accum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    C[row * N + col] = accum;
}

// Initialize GPUs, allocate memory, and create streams
void matmul_initialize(int M, int N, int K) {
    // Get the number of available GPUs
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    if (num_gpus > MAX_GPUS) {
        num_gpus = MAX_GPUS;
    }

    // Allocate host buffer for B (pinned memory for faster transfers)
    CHECK_CUDA(cudaMallocHost(&h_B_buffer, sizeof(float) * K * N));

    // Initialize each GPU
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));

        // Calculate the portion of rows this GPU will handle
        int rows_per_gpu = (M + num_gpus - 1) / num_gpus;
        int row_start = i * rows_per_gpu;
        int row_end = (i + 1) * rows_per_gpu;
        if (row_end > M) row_end = M;
        int rows = row_end - row_start;

        // Allocate device memory for A and C subsets
        CHECK_CUDA(cudaMalloc(&gpu_resources[i].d_A, sizeof(float) * rows * K));
        CHECK_CUDA(cudaMalloc(&gpu_resources[i].d_C, sizeof(float) * rows * N));

        // Allocate device memory for B (all GPUs use the same B)
        CHECK_CUDA(cudaMalloc(&gpu_resources[i].d_B, sizeof(float) * K * N));

        // Create 16 CUDA streams for each GPU
        for (int j = 0; j < NSTREAM; j++) {
            CHECK_CUDA(cudaStreamCreate(&gpu_resources[i].streams[j]));
        }
    }
}

// Perform matrix multiplication using multiple GPUs with OpenMP and CUDA streams
void matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    // Ensure that B is broadcasted only once
    #pragma omp single
    {
        if (!b_broadcasted) {
            // Obtain MPI rank
            int rank;
            CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

            if (rank == 0) {
                // Copy B to host buffer
                memcpy(h_B_buffer, B, sizeof(float) * K * N);
            } else {
                // Initialize host buffer to receive B
                memset(h_B_buffer, 0, sizeof(float) * K * N);
            }

            // Broadcast B from rank 0 to all other ranks
            CHECK_MPI(MPI_Bcast(h_B_buffer, K * N, MPI_FLOAT, 0, MPI_COMM_WORLD));

            b_broadcasted = true;
        }
    }

    // Parallelize across GPUs using OpenMP
    #pragma omp parallel for schedule(static) num_threads(num_gpus)
    for (int i = 0; i < num_gpus; i++) {
        // Set the current GPU
        CHECK_CUDA(cudaSetDevice(i));

        // Calculate the portion of rows this GPU will handle
        int rows_per_gpu = (M + num_gpus - 1) / num_gpus;
        int row_start = i * rows_per_gpu;
        int row_end = (i + 1) * rows_per_gpu;
        if (row_end > M) row_end = M;
        int rows = row_end - row_start;

        // Determine the number of rows per stream
        int rows_per_stream = rows / NSTREAM;
        int remainder = rows % NSTREAM;

        for (int j = 0; j < NSTREAM; j++) {
            // Calculate actual rows for this stream (handle remainder)
            int current_rows = rows_per_stream;
            if (j < remainder) {
                current_rows += 1;
            }

            // Calculate the starting row for this stream
            int current_row_start = row_start + j * rows_per_stream + (j < remainder ? j : remainder);

            // If there are no rows to process for this stream, skip
            if (current_rows <= 0) continue;

            // Asynchronously copy A subset to device using stream[j]
            CHECK_CUDA(cudaMemcpyAsync(gpu_resources[i].d_A + j * rows_per_stream * K,
                                       A + current_row_start * K,
                                       sizeof(float) * current_rows * K,
                                       cudaMemcpyHostToDevice,
                                       gpu_resources[i].streams[j]));

            // Asynchronously copy B to device using stream[j] (only once per GPU)
            // To prevent multiple copies of B, ensure that B is copied only once per GPU
            // This requires copying B outside the stream loop or using synchronization
            // Here, we'll copy B once before the stream loop
            if (j == 0) {
                CHECK_CUDA(cudaMemcpyAsync(gpu_resources[i].d_B,
                                           h_B_buffer,
                                           sizeof(float) * K * N,
                                           cudaMemcpyHostToDevice,
                                           gpu_resources[i].streams[j]));
            }

            // Define grid and block dimensions
            dim3 dimBlock(TILE_SIZE, TILE_SIZE);
            dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (current_rows + TILE_SIZE - 1) / TILE_SIZE);

            // Launch the kernel using stream[j]
            matmul_kernel<<<dimGrid, dimBlock, 0, gpu_resources[i].streams[j]>>>(
                gpu_resources[i].d_A + j * rows_per_stream * K,
                gpu_resources[i].d_B,
                gpu_resources[i].d_C + j * rows_per_stream * N,
                current_rows, N, K
            );

            // Check for kernel launch errors
            CHECK_CUDA(cudaGetLastError());

            // Asynchronously copy the result back to host using stream[j]
            CHECK_CUDA(cudaMemcpyAsync(C + current_row_start * N,
                                       gpu_resources[i].d_C + j * rows_per_stream * N,
                                       sizeof(float) * current_rows * N,
                                       cudaMemcpyDeviceToHost,
                                       gpu_resources[i].streams[j]));
        }

        // After stream loop, ensure that B is copied before other streams
        // This can be done by synchronizing the first stream or using CUDA events
        // For simplicity, we'll assume that B is copied in the first stream and is available for other streams
    }

    // Synchronize all streams
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < NSTREAM; j++) {
            CHECK_CUDA(cudaStreamSynchronize(gpu_resources[i].streams[j]));
        }
    }
}

// Finalize by freeing device memory and destroying streams
void matmul_finalize() {
    // Free host buffer for B
    if (h_B_buffer != NULL) {
        CHECK_CUDA(cudaFreeHost(h_B_buffer));
        h_B_buffer = NULL;
    }

    for (int i = 0; i < num_gpus; i++) {
        // Set the current GPU
        CHECK_CUDA(cudaSetDevice(i));

        // Destroy CUDA streams
        for (int j = 0; j < NSTREAM; j++) {
            CHECK_CUDA(cudaStreamDestroy(gpu_resources[i].streams[j]));
        }

        // Free device memory
        CHECK_CUDA(cudaFree(gpu_resources[i].d_A));
        CHECK_CUDA(cudaFree(gpu_resources[i].d_B));
        CHECK_CUDA(cudaFree(gpu_resources[i].d_C));
    }
}
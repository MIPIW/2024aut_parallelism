////변경가능

#include "layer.h"
// #define TILE_SIZE 16
// #define C_SIZE 16

// __global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
//     // Compute work-item's global row and column
//     const int locRow = threadIdx.y;
//     const int locCol = threadIdx.x;

//     const int glbRow = blockIdx.y * blockDim.y + threadIdx.y;
//     const int glbCol = blockIdx.x * blockDim.x + threadIdx.x;
    
//     const int locBSRow = blockDim.y;
//     const int glbWSRow = gridDim.y * blockDim.y;

//     __shared__ float Asub[TILE_SIZE][TILE_SIZE];
//     __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

//     float temp[C_SIZE];
//     for (int j = 0; j < C_SIZE; ++j) {
//         temp[j] = 0.0;
//     }
    
//     const int NUM_TILES = (K + TILE_SIZE - 1) / TILE_SIZE;

//     for (int t = 0; t < NUM_TILES; ++t) {
//         for (int j = 0; j < C_SIZE; ++j) { // example이 항상 2의 제곱수이니까 패딩을 넣을 필요 없음. 
//           Asub[locRow + j * locBSRow][locCol] = A[(glbRow + j * glbWSRow) * K + (TILE_SIZE * t + locCol)];
//         }

//         for (int k = 0; k < TILE_SIZE; ++k) {
//           Bsub[k][locCol] = B[(TILE_SIZE * t + k) * N + glbCol];
//         }
        
//         __syncthreads();

//         for (int k = 0; k < TILE_SIZE; ++k) {
//             for (int j = 0; j < C_SIZE; ++j) {
//                 int locIdxRowA = locRow + j * locBSRow;
//                 temp[j] += Asub[locIdxRowA][k] * Bsub[k][locCol];
//             }
//         }
//         __syncthreads();
//     }

//     for (int j = 0; j < C_SIZE; ++j) {
//         int glbIdxRow = glbRow + j * glbWSRow;

//         if (glbIdxRow < M && glbCol < N) {
//             C[glbIdxRow * N + glbCol] = temp[j];
//         }
//     }
// }

/* Embedding
 * @param [in1]  in: [b, s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [b, s, H]
 * 'b' is the batch size
 * 's' is the sequence length
 * 'H' is the embedding dimension
 */
void Embedding(int *in, Tensor* w, Tensor *out) {
  size_t B = out->shape[0];  // Batch size
  size_t S = out->shape[1];  // Sequence length
  size_t H = out->shape[2];  // Hidden dimension


  for (size_t k = 0; k < B; ++k) { // Iterate over batches
    for (size_t i = 0; i < S; ++i) { // Iterate over sequence length
      int vocab_idx = in[k * S + i]; // Input index for the current batch and sequence position
      for (size_t j = 0; j < H; ++j) { // Iterate over hidden dimensions
      printf("asdfasdf");
        out->buf[k * (S * H) + i * H + j] = w->buf[vocab_idx * H + j];
      }

    }
  }
}


/* Permute
 * @param [in]   in: [B, S, H]
 * @param [out] out: [B, H, S]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t b = in->shape[0]; // Batch size
  size_t s = in->shape[1]; // Sequence length
  size_t H = in->shape[2]; // Hidden dimension

  for (size_t k = 0; k < b; ++k) { // Iterate over batches
    for (size_t i = 0; i < s; ++i) { // Iterate over sequence length
      for (size_t j = 0; j < H; ++j) { // Iterate over hidden dimensions
        // Swap dimensions S (sequence length) and H (hidden dimensions) for each batch
        out->buf[k * (H * s) + j * s + i] = in->buf[k * (s * H) + i * H + j];
      }
    }
  }
}

/* Conv1D 
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
/* Batched Conv1D 
 * @param [in1]  in: [B, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [B, OC, os]
 */
/* Batched Conv1D 
 * @param [in1]  in: [B, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [B, OC, os]
 */
void Conv1D(Tensor *in, Tensor *w, Tensor *bias, Tensor *out) {
  size_t B = in->shape[0];  // Batch size
  size_t C = in->shape[1];  // Input channels
  size_t s = in->shape[2];  // Input sequence length
  size_t OC = w->shape[0];  // Output channels
  size_t K = w->shape[2];   // Kernel size

  size_t os = s - K + 1;    // Output sequence length

  for (size_t b = 0; b < B; ++b) { // Iterate over batches
    for (size_t oc = 0; oc < OC; ++oc) { // Iterate over output channels
      for (size_t j = 0; j < os; ++j) { // Iterate over output sequence length
        float val = 0.f;
        for (size_t c = 0; c < C; ++c) { // Iterate over input channels
          for (size_t k = 0; k < K; ++k) { // Iterate over kernel size
            val += in->buf[b * (C * s) + c * s + j + k] * 
                   w->buf[oc * (C * K) + c * K + k];
          }
        }
        out->buf[b * (OC * os) + oc * os + j] = val + bias->buf[oc];
      }
    }
  }
}


/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t B = inout->shape[0]; // Batch size
  size_t N = 1;              // Total number of elements per batch (excluding batch dimension)

  for (size_t i = 1; i < inout->ndim; ++i) {
    N *= inout->shape[i];
  }
  
  size_t total_elements = B * N; // Total elements across all batches

  for (size_t i = 0; i < total_elements; ++i) {
    inout->buf[i] = inout->buf[i] > 0 ? inout->buf[i] : 0;
  }
}


/* ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), 
                        cudaMemcpyHostToDevice));

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t B = in->shape[0]; // Batch size
  size_t C = in->shape[1]; // Channel size
  size_t s = in->shape[2]; // Sequence length

  for (size_t b = 0; b < B; ++b) { // Iterate over batches
    for (size_t i = 0; i < C; ++i) { // Iterate over channels
      // Initialize the max value for the current channel in the current batch
      float max_val = in->buf[b * (C * s) + i * s];
      for (size_t j = 1; j < s; ++j) { // Iterate over sequence dimension
        float current_val = in->buf[b * (C * s) + i * s + j];
        if (current_val > max_val) {
          max_val = current_val;
        }
      }
      // Store the maximum value in the output tensor
      out->buf[b * C + i] = max_val;
    }
  }
}

/* Concat
 * @param [in1] in1: [B, N1]
 * @param [in2] in2: [B, N2]
 * @param [in3] in3: [B, N3]
 * @param [in4] in4: [B, N4]
 * @param [out] out: [B, N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  size_t B = in1->shape[0]; // Batch size
  size_t N1 = in1->shape[1]; // Size of the last dimension for in1
  size_t N2 = in2->shape[1]; // Size of the last dimension for in2
  size_t N3 = in3->shape[1]; // Size of the last dimension for in3
  size_t N4 = in4->shape[1]; // Size of the last dimension for in4

  for (size_t b = 0; b < B; ++b) { // Iterate over batches
    // Copy in1
    for (size_t i = 0; i < N1; ++i) {
      out->buf[b * (N1 + N2 + N3 + N4) + i] = in1->buf[b * N1 + i];
    }
    // Copy in2
    for (size_t i = 0; i < N2; ++i) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + i] = in2->buf[b * N2 + i];
    }
    // Copy in3
    for (size_t i = 0; i < N3; ++i) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + N2 + i] = in3->buf[b * N3 + i];
    }
    // Copy in4
    for (size_t i = 0; i < N4; ++i) {
      out->buf[b * (N1 + N2 + N3 + N4) + N1 + N2 + N3 + i] = in4->buf[b * N4 + i];
    }
  }
}

/* Batched Linear 
 * @param [in1]  in: [B, N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [B, M]
 * 
 * 'B' is the batch size
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *bias, Tensor *out) {
  size_t B = in->shape[0]; // Batch size
  size_t N = in->shape[1]; // Input feature size
  size_t M = w->shape[0];  // Output feature size

  for (size_t b = 0; b < B; ++b) { // Iterate over batches
    for (size_t i = 0; i < M; ++i) { // Iterate over output features
      float val = 0.f;
      for (size_t j = 0; j < N; ++j) { // Iterate over input features
        val += in->buf[b * N + j] * w->buf[i * N + j];
      }
      out->buf[b * M + i] = val + bias->buf[i];
    }
  }
}



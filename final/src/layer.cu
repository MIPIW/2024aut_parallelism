////변경가능

#include "layer.h"
#define BLOCK_SIZE 32  // Tile size for shared memory (tune for best performance)
#define ELEMENTS_PER_THREAD 4 // Number of elements each thread will process

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
        out->buf[k * (S * H) + i * H + j] = w->buf[vocab_idx * H + j];
      }

    }
  }
}

__global__ void EmbeddingKernel(int *in, float *w, float *out, size_t B, size_t S, size_t H) {
  // Calculate the global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * S) return; // Ensure the thread index is within bounds

  // Calculate the batch and sequence indices
  size_t batch_idx = idx / S;  // Batch index
  size_t seq_idx = idx % S;    // Sequence index

  // Retrieve the vocabulary index for this input
  int vocab_idx = in[batch_idx * S + seq_idx];
  if (vocab_idx < 0 || vocab_idx >= 21635) return; // Check bounds for vocabulary size

  // Compute the offset in the output buffer
  size_t out_offset = batch_idx * (S * H) + seq_idx * H;

  // Populate the embedding for the given sequence position
  for (size_t j = 0; j < H; ++j) {
    out[out_offset + j] = w[vocab_idx * H + j];
  }
}



 // Number of elements each thread copies from the embedding

__global__ void EmbeddingKernelOptimized(int *in, float *w, float *out, size_t B, size_t S, size_t H) {
    // Batch index (blockIdx.y) and sequence index (blockIdx.x)
    size_t batch_idx = blockIdx.y;
    size_t seq_idx = blockIdx.x;

    // Offset for the vocabulary index
    int vocab_idx = in[batch_idx * S + seq_idx];
    if (vocab_idx < 0 || vocab_idx >= 21635) return; // Bounds check for vocabulary size

    // Each thread processes a chunk of `ELEMENTS_PER_THREAD` dimensions
    size_t thread_offset = threadIdx.x * ELEMENTS_PER_THREAD;

    // Ensure we don't go out of bounds
    if (thread_offset >= H) return;

    // Compute the output offset for this batch and sequence position
    size_t out_offset = batch_idx * (S * H) + seq_idx * H + thread_offset;

    // Compute the embedding weight offset
    size_t w_offset = vocab_idx * H + thread_offset;

    // Copy the embedding vector chunk
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD && (thread_offset + i) < H; ++i) {
        out[out_offset + i] = w[w_offset + i];
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

/* Permute
 * @param [in]   in: [B, S, H]
 * @param [out] out: [B, H, S]
 */
__global__ void PermuteKernel(float *in, float *out, size_t b, size_t s, size_t H) {
  // Calculate the global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of elements across all batches
  size_t total_elements = b * s * H;

  // Return if the thread index exceeds the total number of elements
  if (idx >= total_elements) return;

  // Calculate the batch index, sequence index, and hidden dimension index
  size_t batch_idx = idx / (s * H);         // Batch index
  size_t seq_hidden_idx = idx % (s * H);    // Position within the sequence and hidden dims
  size_t seq_idx = seq_hidden_idx / H;      // Sequence index
  size_t hidden_idx = seq_hidden_idx % H;   // Hidden dimension index

  // Perform the permutation (swap dimensions S and H)
  out[batch_idx * (s * H) + hidden_idx * s + seq_idx] = 
      in[batch_idx * (s * H) + seq_idx * H + hidden_idx];
}

__global__ void EmbeddingPermuteKernel(int *in, float *w, float *out, size_t B, size_t S, size_t H) {
    // Calculate the global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of elements across all batches
    size_t total_elements = B * S * H;

    // Return if the thread index exceeds the total number of elements
    if (idx >= total_elements) return;

    // Calculate the batch index, sequence index, and hidden dimension index
    size_t batch_idx = idx / (S * H);           // Batch index
    size_t seq_hidden_idx = idx % (S * H);      // Position within the sequence and hidden dims
    size_t seq_idx = seq_hidden_idx / H;        // Sequence index
    size_t hidden_idx = seq_hidden_idx % H;     // Hidden dimension index

    // Retrieve the vocabulary index for this input
    int vocab_idx = in[batch_idx * S + seq_idx];
    if (vocab_idx < 0 || vocab_idx >= 21635) return; // Check bounds for vocabulary size

    // Perform embedding lookup and permutation in one step
    out[batch_idx * (S * H) + hidden_idx * S + seq_idx] = w[vocab_idx * H + hidden_idx];
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

__global__ void Conv1DKernel(
    float *in, float *w, float *bias, float *out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) {
  // Output sequence length
  size_t os = s - K + 1;

  // Thread index: represents the position in the sequence
  int j = threadIdx.x; // Sequence position

  // Block index: represents the batch
  int b = blockIdx.x; // Batch index

  // Ensure within bounds
  if (j >= os) return;

  // Iterate over output channels
  for (size_t oc = 0; oc < OC; ++oc) {
    float val = 0.f;

    // Compute the convolution for the current output channel and sequence position
    for (size_t c = 0; c < C; ++c) {
      for (size_t k = 0; k < K; ++k) {
        val += in[b * (C * s) + c * s + j + k] * w[oc * (C * K) + c * K + k];
      }
    }

    // Add bias and write the result to the output tensor
    out[b * (OC * os) + oc * os + j] = val + bias[oc];
  }
}
  // Tile size for shared memory (tune based on hardware capabilities)

__global__ void Conv1DKernelTiled(
    const float *in, const float *w, const float *bias, float *out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) {
    
    // Output sequence length
    size_t os = s - K + 1;

    // Batch index (each block processes one batch element)
    size_t b = blockIdx.x;

    // Output channel index
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;

    // Sequence position index
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    // Ensure within bounds
    if (b >= B || oc >= OC || j >= os) return;

    // Accumulator for the convolution result
    float val = 0.0f;

    // Compute the convolution for the current output channel, sequence position, and batch
    for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < K; ++k) {
            val += in[b * (C * s) + c * s + j + k] * w[oc * (C * K) + c * K + k];
        }
    }

    val += bias[oc];
    // Add bias and store the result in the output tensor
    out[b * (OC * os) + oc * os + j] = val > 0? val : 0;
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

__global__ void ReLUKernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%llu\t", N);
  if (i < N) {
    // inout[i] = fmax(inout[i], 0.0f);
    inout[i] = inout[i] > 0 ? inout[i] : 0;

  }
}

__global__ void ReLUKernelVectorized(float *inout, size_t N) {
    size_t i = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // Process four elements per thread using vectorized operations
    float4 *vec_inout = reinterpret_cast<float4*>(inout);
    size_t num_vectors = N / 4;

    for (size_t idx = i / 4; idx < num_vectors; idx += gridDim.x * blockDim.x) {
        float4 val = vec_inout[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        vec_inout[idx] = val;
    }

    // Handle remaining elements if N is not a multiple of 4
    size_t remaining = N % 4;
    size_t remainder_start = N - remaining;
    i = remainder_start + threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N) {
        inout[i] = fmaxf(inout[i], 0.0f);
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

__global__ void GetMaxKernel(float *in, float *out, size_t B, size_t C, size_t S) {
  // Calculate the global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Total number of channels across all batches
  size_t total_channels = B * C;

  // Ensure the thread index is within bounds
  if (idx >= total_channels) return;

  // Calculate the batch index and channel index
  size_t batch_idx = idx / C;     // Batch index
  size_t channel_idx = idx % C;   // Channel index

  // Initialize the max value for the current channel in the current batch
  size_t base_idx = batch_idx * (C * S) + channel_idx * S;
  float max_val = in[base_idx];

  // Iterate over the sequence dimension to find the maximum value
  for (size_t j = 1; j < S; ++j) {
    float current_val = in[base_idx + j];
    if (current_val > max_val) {
      max_val = current_val;
    }
  }

  // Store the maximum value in the output array
  out[idx] = max_val;
}


__global__ void Conv1DReLUAndMaxPoolKernel(
    const float *in, const float *w, const float *bias, float *out, float *pool_out,
    size_t B, size_t C, size_t s, size_t OC, size_t K, size_t pool_size) {

    // Output sequence length after convolution
    size_t os = s - K + 1;

    // Batch index (each block processes one batch element)
    size_t b = blockIdx.x;

    // Output channel index
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;

    // Sequence position index
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    // Ensure within bounds
    if (b >= B || oc >= OC || j >= os) return;

    // Accumulator for the convolution result
    float val = 0.0f;

    // Compute the convolution for the current output channel, sequence position, and batch
    for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < K; ++k) {
            val += in[b * (C * s) + c * s + j + k] * w[oc * (C * K) + c * K + k];
        }
    }

    // Add bias and apply ReLU activation
    val += bias[oc];
    val = val > 0 ? val : 0;

    // Store the convolution result with ReLU in the output tensor
    size_t output_index = b * (OC * os) + oc * os + j;

    // Perform max pooling within the thread (assuming pool_size divides os evenly)
    __shared__ float shared_pool_vals[1024];  // Adjust size based on your block configuration

    // Each thread loads its value into shared memory
    shared_pool_vals[threadIdx.x] = val;
    __syncthreads();

    // Perform reduction to find the maximum value for the pooling window
    for (size_t stride = pool_size / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_pool_vals[threadIdx.x] = fmaxf(shared_pool_vals[threadIdx.x], shared_pool_vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // Write the pooled result to the output (only the first thread in each pool window writes)
    if (threadIdx.x == 0) {
        pool_out[b * OC + oc] = shared_pool_vals[0];
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

// __global__ void ConcatKernel(const float *in1, const float *in2, const float *in3, const float *in4,
//                              float *out, size_t B, size_t N1, size_t N2, size_t N3, size_t N4) {
//     // Get the global thread index
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Calculate the per-batch sizes for each input tensor
//     size_t per_batch_N1 = N1 / B;
//     size_t per_batch_N2 = N2 / B;
//     size_t per_batch_N3 = N3 / B;
//     size_t per_batch_N4 = N4 / B;

//     // Calculate the total number of elements per batch
//     size_t total_size_per_batch = per_batch_N1 + per_batch_N2 + per_batch_N3 + per_batch_N4;

//     // Total number of elements across all batches
//     size_t total_elements = B * total_size_per_batch;

//     // Ensure the thread index is within bounds
//     if (idx >= total_elements) return;

//     // Determine the batch index
//     size_t b = idx / total_size_per_batch;
//     size_t offset_within_batch = idx % total_size_per_batch;

//     // Copy data from the appropriate input tensor
//     if (offset_within_batch < per_batch_N1) {
//         out[idx] = in1[b * per_batch_N1 + offset_within_batch];
//     } else if (offset_within_batch < per_batch_N1 + per_batch_N2) {
//         out[idx] = in2[b * per_batch_N2 + (offset_within_batch - per_batch_N1)];
//     } else if (offset_within_batch < per_batch_N1 + per_batch_N2 + per_batch_N3) {
//         out[idx] = in3[b * per_batch_N3 + (offset_within_batch - per_batch_N1 - per_batch_N2)];
//     } else if (offset_within_batch < per_batch_N1 + per_batch_N2 + per_batch_N3 + per_batch_N4) {
//         out[idx] = in4[b * per_batch_N4 + (offset_within_batch - per_batch_N1 - per_batch_N2 - per_batch_N3)];
//     }
// }


__global__ void ConcatKernel(const float *in1, const float *in2, const float *in3, const float *in4,
                             float *out, size_t B, size_t N1, size_t N2, size_t N3, size_t N4) {
    // Calculate the total number of elements per batch
    size_t per_batch_N1 = N1 / B;
    size_t per_batch_N2 = N2 / B;
    size_t per_batch_N3 = N3 / B;
    size_t per_batch_N4 = N4 / B;
    size_t total_size_per_batch = per_batch_N1 + per_batch_N2 + per_batch_N3 + per_batch_N4;

    // Global thread index, adjusted to process multiple elements per thread
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    // Ensure we are within bounds
    if (idx >= B * total_size_per_batch) return;

    // Determine the batch index
    size_t b = idx / total_size_per_batch;
    size_t offset_within_batch = idx % total_size_per_batch;

    // Process multiple elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (idx + i >= B * total_size_per_batch) break;

        size_t current_offset = offset_within_batch + i;

        if (current_offset < per_batch_N1) {
            out[idx + i] = in1[b * per_batch_N1 + current_offset];
        } else if (current_offset < per_batch_N1 + per_batch_N2) {
            out[idx + i] = in2[b * per_batch_N2 + (current_offset - per_batch_N1)];
        } else if (current_offset < per_batch_N1 + per_batch_N2 + per_batch_N3) {
            out[idx + i] = in3[b * per_batch_N3 + (current_offset - per_batch_N1 - per_batch_N2)];
        } else {
            out[idx + i] = in4[b * per_batch_N4 + (current_offset - per_batch_N1 - per_batch_N2 - per_batch_N3)];
        }
    }
}

__global__ void ConcatKernelOneN(const float *in1, const float *in2, const float *in3, const float *in4,
                             float *out, size_t B, size_t N) {
    // Shared memory for inputs
    __shared__ float shared_in1[ELEMENTS_PER_THREAD * BLOCK_SIZE];
    __shared__ float shared_in2[ELEMENTS_PER_THREAD * BLOCK_SIZE];
    __shared__ float shared_in3[ELEMENTS_PER_THREAD * BLOCK_SIZE];
    __shared__ float shared_in4[ELEMENTS_PER_THREAD * BLOCK_SIZE];

    // Calculate the total number of elements per batch
    size_t total_size_per_batch = 4 * N;  // Since N1, N2, N3, N4 are equal

    // Global thread index, adjusted to process multiple elements per thread
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    // Ensure we are within bounds
    if (idx >= B * total_size_per_batch) return;

    // Determine the batch index
    size_t b = idx / total_size_per_batch;
    size_t offset_within_batch = idx % total_size_per_batch;

    // Load data into shared memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (offset_within_batch + i < N) {
            shared_in1[threadIdx.x * ELEMENTS_PER_THREAD + i] = in1[b * N + offset_within_batch + i];
            shared_in2[threadIdx.x * ELEMENTS_PER_THREAD + i] = in2[b * N + offset_within_batch + i];
            shared_in3[threadIdx.x * ELEMENTS_PER_THREAD + i] = in3[b * N + offset_within_batch + i];
            shared_in4[threadIdx.x * ELEMENTS_PER_THREAD + i] = in4[b * N + offset_within_batch + i];
        }
    }

    __syncthreads();

    // Copy data from shared memory to the output
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (offset_within_batch + i < N) {
            out[b * total_size_per_batch + offset_within_batch + i] = shared_in1[threadIdx.x * ELEMENTS_PER_THREAD + i];
            out[b * total_size_per_batch + offset_within_batch + N + i] = shared_in2[threadIdx.x * ELEMENTS_PER_THREAD + i];
            out[b * total_size_per_batch + offset_within_batch + 2 * N + i] = shared_in3[threadIdx.x * ELEMENTS_PER_THREAD + i];
            out[b * total_size_per_batch + offset_within_batch + 3 * N + i] = shared_in4[threadIdx.x * ELEMENTS_PER_THREAD + i];
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

__global__ void LinearKernel(const float *in, const float *w, const float *bias, float *out,
                             size_t B, size_t N, size_t M) {
    // Batch index (each block processes one batch element)
    size_t b = blockIdx.x;
    // Output feature index (each thread processes one output feature)
    size_t m = threadIdx.x + blockIdx.y * blockDim.x;

    if (b < B && m < M) {
        float val = 0.0f;
        // Compute the dot product for the m-th output feature
        for (size_t j = 0; j < N; ++j) {
            val += w[m * N + j] * in[b * N + j];
        }
        // Add bias and store the result in the output tensor
        out[b * M + m] = val + bias[m];
    }
}




__global__ void LinearKernelTiled(const float *in, const float *w, const float *bias, float *out,
                                  size_t B, size_t N, size_t M) {
    // Shared memory for storing tiles of input and weight matrices
    __shared__ float tileIn[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileW[BLOCK_SIZE][BLOCK_SIZE];

    // Batch index (each block processes one batch element)
    size_t b = blockIdx.z;

    // Output feature index (m) and input feature index (n) for the current thread
    size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Accumulator for the dot product
    float val = 0.0f;

    // Loop over tiles of the input and weight matrices
    for (size_t t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load a tile of the weight matrix into shared memory (if within bounds)
        if (m < M && t * BLOCK_SIZE + threadIdx.x < N) {
            tileW[threadIdx.y][threadIdx.x] = w[m * N + t * BLOCK_SIZE + threadIdx.x];
        } else {
            tileW[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of the input matrix into shared memory (if within bounds)
        if (n < N && t * BLOCK_SIZE + threadIdx.y < N) {
            tileIn[threadIdx.y][threadIdx.x] = in[b * N + t * BLOCK_SIZE + threadIdx.y];
        } else {
            tileIn[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            val += tileW[threadIdx.y][k] * tileIn[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to the output tensor (if within bounds) and add bias
    if (m < M && n < N) {
        out[b * M + m] = val + bias[m];
    }
}



__global__ void LinearKernelTiledWithRelu(const float *in, const float *w, const float *bias, float *out,
                                  size_t B, size_t N, size_t M) {
    // Shared memory for storing tiles of input and weight matrices (double buffering)
    __shared__ float tileIn[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileW[2][BLOCK_SIZE][BLOCK_SIZE];

    // Batch index (each block processes one batch element)
    size_t b = blockIdx.z;

    // Output feature index (m) and input feature index (n) for the current thread
    size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Accumulator for the dot product
    float val = 0.0f;

    // Double buffering index
    int bufIdx = 0;

    // Loop over tiles of the input and weight matrices
    for (size_t t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load a tile of the weight matrix into shared memory (if within bounds)
        if (m < M && t * BLOCK_SIZE + threadIdx.x < N) {
            tileW[bufIdx][threadIdx.y][threadIdx.x] = w[m * N + t * BLOCK_SIZE + threadIdx.x];
        } else {
            tileW[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of the input matrix into shared memory (if within bounds)
        if (n < N && t * BLOCK_SIZE + threadIdx.y < N) {
            tileIn[bufIdx][threadIdx.y][threadIdx.x] = in[b * N + t * BLOCK_SIZE + threadIdx.y];
        } else {
            tileIn[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            val += tileW[bufIdx][threadIdx.y][k] * tileIn[bufIdx][k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();

        // Toggle the buffer index for double buffering
        bufIdx = 1 - bufIdx;
    }

    val += bias[m];
    // Write the result to the output tensor (if within bounds) and add bias
    if (m < M && n < N) {
        out[b * M + m] = val > 0? val : 0;
    }
}

__global__ void LinearKernelTiled2(const float *in, const float *w, const float *bias, float *out,
                                  size_t B, size_t N, size_t M) {
    // Shared memory for storing tiles of input and weight matrices (double buffering)
    __shared__ float tileIn[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileW[2][BLOCK_SIZE][BLOCK_SIZE];

    // Batch index (each block processes one batch element)
    size_t b = blockIdx.z;

    // Output feature index (m) and input feature index (n) for the current thread
    size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Accumulator for the dot product
    float val = 0.0f;

    // Double buffering index
    int bufIdx = 0;

    // Loop over tiles of the input and weight matrices
    for (size_t t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load a tile of the weight matrix into shared memory (if within bounds)
        if (m < M && t * BLOCK_SIZE + threadIdx.x < N) {
            tileW[bufIdx][threadIdx.y][threadIdx.x] = w[m * N + t * BLOCK_SIZE + threadIdx.x];
        } else {
            tileW[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of the input matrix into shared memory (if within bounds)
        if (n < N && t * BLOCK_SIZE + threadIdx.y < N) {
            tileIn[bufIdx][threadIdx.y][threadIdx.x] = in[b * N + t * BLOCK_SIZE + threadIdx.y];
        } else {
            tileIn[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial dot product for the current tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            val += tileW[bufIdx][threadIdx.y][k] * tileIn[bufIdx][k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();

        // Toggle the buffer index for double buffering
        bufIdx = 1 - bufIdx;
    }

    // Write the result to the output tensor (if within bounds) and add bias
    if (m < M && n < N) {
        out[b * M + m] = val + bias[m];
    }
}


////변경가능

#include "layer.h"
#include <cuda_fp16.h>
#include <mma.h>

#define BLOCK_SIZE 32  // Tile size for shared memory (tune for best performance)
#define ELEMENTS_PER_THREAD 4 // Number of elements each thread will process
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda::wmma;  // For WMMA (Warp Matrix Multiply-Accumulate)
using namespace nvcuda;  // For WMMA (Warp Matrix Multiply-Accumulate)


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


// Assume block size along x is fixed to 32 for the output sequence dimension.

// This kernel uses shared memory tiling and float4 loads.
// Grid dimensions:
//   gridDim.x = B
//   gridDim.y = number of output channel blocks (ceil(OC / OC_per_block))
//   gridDim.z = number of sequence tiles (ceil((s - K + 1) / BLOCK_SIZE))
//
// blockDim = (BLOCK_SIZE, maybe an OC_per_block if multi-dimensional in Y, 1)
//
// For simplicity, we consider a single output channel per block in Y dimension below.
// Modify as needed if you want multiple output channels per block.


#define TILE_C 128      // Number of input channels processed per tile
#define TILE_OC 8       // Number of output channels processed per tile
#define TILE_J 32 
#define KERNEL_SIZE 3   // K (given as 5)
// #define THREADS_X 32    // Threads in x-dim
// #define THREADS_Y 4     // Threads in y-dim, total 128 threads/block


__global__ void Conv1DKernelTiled3(
    const float * in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) 
{
    // Output sequence length
    size_t os = s - K + 1;

    // Identify current batch index
    size_t b = blockIdx.x;
    // Output channel index
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;
    // Sequence position index
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    // Bounds check
    if (b >= B || oc >= OC || j >= os) return;

    // Accumulator
    float val = 0.0f;

    // Unroll K since K=3.
    // We'll process C in chunks of 4.
    // We'll process C in chunks of 4.
    for (size_t c4 = 0; c4 < C; c4 += 4) {
        // Instead of float4 vector loads, we do scalar loads for each element.
        // Pre-load the input values for k=0, k=1, and k=2 for these 4 channels.

        float x_k0[4], x_k1[4], x_k2[4];
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                x_k0[idx] = in[b * (C * s) + c_index * s + (j + 0)];
                x_k1[idx] = in[b * (C * s) + c_index * s + (j + 1)];
                x_k2[idx] = in[b * (C * s) + c_index * s + (j + 2)];
            } else {
                // If c_index is out of range, set to 0.0f
                x_k0[idx] = 0.0f;
                x_k1[idx] = 0.0f;
                x_k2[idx] = 0.0f;
            }
        }

        // Now perform the accumulation with the corresponding weights
        #pragma unroll
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                // Load weights for each k
                float w_k0 = w[oc * (C * K) + c_index * K + 0];
                float w_k1 = w[oc * (C * K) + c_index * K + 1];
                float w_k2 = w[oc * (C * K) + c_index * K + 2];

                float val_k0 = x_k0[idx];
                float val_k1 = x_k1[idx];
                float val_k2 = x_k2[idx];

                val += val_k0 * w_k0 + val_k1 * w_k1 + val_k2 * w_k2;
            }
        }
    }

    // Add bias and apply ReLU
    val += bias[oc];
    val = val > 0.0f ? val : 0.0f;

    // Store the result
    out[b * (OC * os) + oc * os + j] = val;

}


__global__ void Conv1DKernelTiled5(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) 
{
    // Output sequence length
    size_t os = s - K + 1; // K=5 -> os = s - 4

    size_t b = blockIdx.x;
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    if (b >= B || oc >= OC || j >= os) return;

    float val = 0.0f;

    // Process C in chunks of 4
    for (size_t c4 = 0; c4 < C; c4 += 4) {
        float x_k0[4], x_k1[4], x_k2[4], x_k3[4], x_k4[4];
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                x_k0[idx] = in[b * (C * s) + c_index * s + (j + 0)];
                x_k1[idx] = in[b * (C * s) + c_index * s + (j + 1)];
                x_k2[idx] = in[b * (C * s) + c_index * s + (j + 2)];
                x_k3[idx] = in[b * (C * s) + c_index * s + (j + 3)];
                x_k4[idx] = in[b * (C * s) + c_index * s + (j + 4)];
            } else {
                x_k0[idx] = 0.0f;
                x_k1[idx] = 0.0f;
                x_k2[idx] = 0.0f;
                x_k3[idx] = 0.0f;
                x_k4[idx] = 0.0f;
            }
        }

        #pragma unroll
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                float w_k0 = w[oc * (C * K) + c_index * K + 0];
                float w_k1 = w[oc * (C * K) + c_index * K + 1];
                float w_k2 = w[oc * (C * K) + c_index * K + 2];
                float w_k3 = w[oc * (C * K) + c_index * K + 3];
                float w_k4 = w[oc * (C * K) + c_index * K + 4];

                val += x_k0[idx] * w_k0
                     + x_k1[idx] * w_k1
                     + x_k2[idx] * w_k2
                     + x_k3[idx] * w_k3
                     + x_k4[idx] * w_k4;
            }
        }
    }

    val += bias[oc];
    val = val > 0.0f ? val : 0.0f;

    out[b * (OC * os) + oc * os + j] = val;
}

__global__ void Conv1DKernelTiled7(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) 
{
    // Output sequence length
    size_t os = s - K + 1; // K=7 -> os = s - 6

    size_t b = blockIdx.x;
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    if (b >= B || oc >= OC || j >= os) return;

    float val = 0.0f;

    // Process C in chunks of 4
    for (size_t c4 = 0; c4 < C; c4 += 4) {
        float x_k0[4], x_k1[4], x_k2[4], x_k3[4], x_k4[4], x_k5[4], x_k6[4];
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                x_k0[idx] = in[b * (C * s) + c_index * s + (j + 0)];
                x_k1[idx] = in[b * (C * s) + c_index * s + (j + 1)];
                x_k2[idx] = in[b * (C * s) + c_index * s + (j + 2)];
                x_k3[idx] = in[b * (C * s) + c_index * s + (j + 3)];
                x_k4[idx] = in[b * (C * s) + c_index * s + (j + 4)];
                x_k5[idx] = in[b * (C * s) + c_index * s + (j + 5)];
                x_k6[idx] = in[b * (C * s) + c_index * s + (j + 6)];
            } else {
                x_k0[idx] = 0.0f;
                x_k1[idx] = 0.0f;
                x_k2[idx] = 0.0f;
                x_k3[idx] = 0.0f;
                x_k4[idx] = 0.0f;
                x_k5[idx] = 0.0f;
                x_k6[idx] = 0.0f;
            }
        }

        #pragma unroll
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                float w_k0 = w[oc * (C * K) + c_index * K + 0];
                float w_k1 = w[oc * (C * K) + c_index * K + 1];
                float w_k2 = w[oc * (C * K) + c_index * K + 2];
                float w_k3 = w[oc * (C * K) + c_index * K + 3];
                float w_k4 = w[oc * (C * K) + c_index * K + 4];
                float w_k5 = w[oc * (C * K) + c_index * K + 5];
                float w_k6 = w[oc * (C * K) + c_index * K + 6];

                val += x_k0[idx] * w_k0
                     + x_k1[idx] * w_k1
                     + x_k2[idx] * w_k2
                     + x_k3[idx] * w_k3
                     + x_k4[idx] * w_k4
                     + x_k5[idx] * w_k5
                     + x_k6[idx] * w_k6;
            }
        }
    }

    val += bias[oc];
    val = val > 0.0f ? val : 0.0f;

    out[b * (OC * os) + oc * os + j] = val;
}


__global__ void Conv1DKernelTiled9(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) 
{
    // Output sequence length
    size_t os = s - K + 1; // K=9 -> os = s - 8

    size_t b = blockIdx.x;
    size_t oc = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.z * blockDim.x + threadIdx.x;

    if (b >= B || oc >= OC || j >= os) return;

    float val = 0.0f;

    // Process C in chunks of 4
    for (size_t c4 = 0; c4 < C; c4 += 4) {
        float x_k0[4], x_k1[4], x_k2[4], x_k3[4], x_k4[4], x_k5[4], x_k6[4], x_k7[4], x_k8[4];
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                x_k0[idx] = in[b * (C * s) + c_index * s + (j + 0)];
                x_k1[idx] = in[b * (C * s) + c_index * s + (j + 1)];
                x_k2[idx] = in[b * (C * s) + c_index * s + (j + 2)];
                x_k3[idx] = in[b * (C * s) + c_index * s + (j + 3)];
                x_k4[idx] = in[b * (C * s) + c_index * s + (j + 4)];
                x_k5[idx] = in[b * (C * s) + c_index * s + (j + 5)];
                x_k6[idx] = in[b * (C * s) + c_index * s + (j + 6)];
                x_k7[idx] = in[b * (C * s) + c_index * s + (j + 7)];
                x_k8[idx] = in[b * (C * s) + c_index * s + (j + 8)];
            } else {
                x_k0[idx] = 0.0f;
                x_k1[idx] = 0.0f;
                x_k2[idx] = 0.0f;
                x_k3[idx] = 0.0f;
                x_k4[idx] = 0.0f;
                x_k5[idx] = 0.0f;
                x_k6[idx] = 0.0f;
                x_k7[idx] = 0.0f;
                x_k8[idx] = 0.0f;
            }
        }

        #pragma unroll
        for (int idx = 0; idx < 4; idx++) {
            size_t c_index = c4 + idx;
            if (c_index < C) {
                float w_k0 = w[oc * (C * K) + c_index * K + 0];
                float w_k1 = w[oc * (C * K) + c_index * K + 1];
                float w_k2 = w[oc * (C * K) + c_index * K + 2];
                float w_k3 = w[oc * (C * K) + c_index * K + 3];
                float w_k4 = w[oc * (C * K) + c_index * K + 4];
                float w_k5 = w[oc * (C * K) + c_index * K + 5];
                float w_k6 = w[oc * (C * K) + c_index * K + 6];
                float w_k7 = w[oc * (C * K) + c_index * K + 7];
                float w_k8 = w[oc * (C * K) + c_index * K + 8];

                val += x_k0[idx] * w_k0
                     + x_k1[idx] * w_k1
                     + x_k2[idx] * w_k2
                     + x_k3[idx] * w_k3
                     + x_k4[idx] * w_k4
                     + x_k5[idx] * w_k5
                     + x_k6[idx] * w_k6
                     + x_k7[idx] * w_k7
                     + x_k8[idx] * w_k8;
            }
        }
    }

    val += bias[oc];
    val = val > 0.0f ? val : 0.0f;

    out[b * (OC * os) + oc * os + j] = val;
}

// #define alpha 1024.0f

// using wmma::fragment;
// using wmma::load_matrix_sync;
// using wmma::store_matrix_sync;
// using wmma::fill_fragment;
// using wmma::mma_sync;
// using wmma::matrix_a;
// using wmma::matrix_b;
// using wmma::accumulator;

// #include <mma.h>
// using namespace nvcuda;

// __global__ void Conv1DKernelTiled9(
//     const float * __restrict__ in,
//     const float * __restrict__ w,
//     const float * __restrict__ bias,
//     float * __restrict__ out,
//     size_t B, size_t C, size_t s, size_t OC, size_t K)
// {
//     // For K=9, output sequence length = s - 8
//     size_t os = (K <= s) ? (s - K + 1) : 0; 
//     // If K > s, no valid output positions, os = 0

//     size_t b  = blockIdx.x;
//     size_t oc = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t j  = blockIdx.z * blockDim.x + threadIdx.x;

//     // Out-of-bound checks
//     if (b >= B || oc >= OC || j >= os) return;


//     fragment<matrix_a, 16,16,16, half, wmma::row_major> a_frag;
//     fragment<matrix_b, 16,16,16, half, wmma::col_major> b_frag;
//     fragment<accumulator,16,16,16,float> c_frag;

//     fill_fragment(c_frag, 0.0f);

//     size_t total_length = C * K;
//     // We'll process the (C*K) dimension in chunks of 16
//     // If total_length is not a multiple of 16, we'll zero out extra parts

//     // Similarly, if oc does not fall on a 16-tile boundary or exceeds OC, we handle that by zeroing out invalid entries.

//     int oc_block = (int)((oc / 16) * 16); // Start of the 16-wide block of output channels
//     // We'll handle partial blocks if oc or oc_block + col >= OC

//     // Loop over the K dimension in steps of 16
//     for (size_t k_base = 0; k_base < ((total_length + 15) / 16) * 16; k_base += 16) {
//         half A_mat[16];      // 1x16 tile for A
//         half B_mat[16*16];   // 16x16 tile for B

//         // Load A:
//         // For each col in [0..15], determine ck = k_base+col
//         // If ck >= total_length, set to zero
//         // Otherwise load in[b, c_idx, j+k_idx] if in range
//         for (int col = 0; col < 16; col++) {
//             size_t ck = k_base + col;
//             half val_h;
//             if (ck < total_length) {
//                 size_t c_idx = ck / K;
//                 size_t k_idx = ck % K;
//                 float in_val = 0.0f;
//                 if ((c_idx < C) && ((j + k_idx) < s)) {
//                     in_val = in[b * (C * s) + c_idx * s + (j + k_idx)];
//                 }
//                 val_h = __float2half_rn(in_val * alpha);
//             } else {
//                 // Out of range in K dimension
//                 val_h = __float2half_rn(0.0f);
//             }
//             A_mat[col] = val_h;
//         }

//         load_matrix_sync(a_frag, A_mat, 16);

//         // Load B:
//         // For B, rows correspond to ck in [k_base..k_base+15],
//         // columns correspond to [oc_block..oc_block+15].
//         for (int row = 0; row < 16; row++) {
//             size_t ck = k_base + row;
//             for (int col = 0; col < 16; col++) {
//                 half val_h;
//                 size_t oc_col = oc_block + col;
//                 if (ck < total_length && (oc_col < OC)) {
//                     size_t c_idx = ck / K;
//                     size_t k_idx = ck % K;
//                     float w_val = 0.0f;
//                     if (c_idx < C && k_idx < K && oc_col < OC) {
//                         w_val = w[oc_col*(C*K) + c_idx*K + k_idx];
//                     }
//                     val_h = __float2half_rn(w_val * alpha);
//                 } else {
//                     // Outside valid OC or CK range
//                     val_h = __float2half_rn(0.0f);
//                 }
//                 B_mat[row*16 + col] = val_h;
//             }
//         }

//         load_matrix_sync(b_frag, B_mat, 16);

//         // Perform matrix multiply accumulate
//         mma_sync(c_frag, a_frag, b_frag, c_frag);
//     }

//     // After accumulation:
//     // The result is scaled by alpha*alpha
//     float scale_back = 1.0f / (alpha * alpha);
//     float result = c_frag.x[0] * scale_back;

//     // Add bias if in range
//     float bias_val = 0.0f;
//     if (oc < OC) {
//         bias_val = bias[oc];
//     }
//     result += bias_val;

//     // Apply ReLU
//     result = (result > 0.0f) ? result : 0.0f;

//     // Store result if valid
//     if (b < B && oc < OC && j < os) {
//         out[b * (OC * os) + oc * os + j] = result;
//     }
// }



// __global__ void Conv1DKernelTiled3(
//     const float *in, const float *w, const float *bias, float *out,
//     size_t B, size_t C, size_t s, size_t OC, size_t K) {
    
//     // Output sequence length
//     size_t os = s - K + 1;

//     // Batch index (each block processes one batch element)
//     size_t b = blockIdx.x;

//     // Output channel index
//     size_t oc = blockIdx.y * blockDim.y + threadIdx.y;

//     // Sequence position index (with tiling applied)
//     size_t j_start = blockIdx.z * BLOCK_SIZE;
//     size_t j_local = threadIdx.x;
//     size_t j = j_start + j_local;

//     // Ensure within bounds
//     if (b >= B || oc >= OC || j >= os) return;

//     // Shared memory for input tile
//     __shared__ float in_tile[BLOCK_SIZE + 3-1];  // BLOCK_SIZE + K - 1 (to handle boundary cases)
    
//     // Accumulator for the convolution result
//     float val = 0.0f;

//     // Load the necessary input data into shared memory
//     for (size_t c = 0; c < C; ++c) {
//         // Load input data into shared memory
//         if (j + threadIdx.x < s) {
//             in_tile[threadIdx.x] = in[b * (C * s) + c * s + j + threadIdx.x];
//         }
//         __syncthreads();  // Ensure all threads have loaded their data

//         // Perform the convolution for this output channel, sequence position, and batch
//         for (size_t k = 0; k < K; ++k) {
//             if (j + k < os) {
//                 val += in_tile[j_local + k] * w[oc * (C * K) + c * K + k];
//             }
//         }
//         __syncthreads();  // Synchronize before loading the next channel
//     }

//     val += bias[oc];
//     // Apply ReLU activation and store the result in the output tensor
//     out[b * (OC * os) + oc * os + j] = val > 0 ? val : 0;
// }



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


// __global__ void LinearKernelTiledWithRelu(const float *in, const float *w, const float *bias, float *out,
//                                           size_t B, size_t N, size_t M) {
//     __shared__ float tileIn[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float tileW[BLOCK_SIZE][BLOCK_SIZE];

//     // Batch index
//     size_t b = blockIdx.z;

//     // Output and input feature indices
//     size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//     size_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x;

//     // Accumulator for dot product
//     float val = 0.0f;

//     // Loop over tiles
//     for (size_t t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
//         // Load input tile (coalesced read)
//         tileIn[threadIdx.y][threadIdx.x] = in[b * N + (t * BLOCK_SIZE + threadIdx.y)];

//     // Load weight tile (coalesced read)
//         tileW[threadIdx.y][threadIdx.x] = w[m * N + (t * BLOCK_SIZE + threadIdx.x)];

//         __syncthreads();

//         // Unroll the loop for better performance
//         #pragma unroll
//         for (int k = 0; k < BLOCK_SIZE; k++) {
//             val += tileW[threadIdx.y][k] * tileIn[k][threadIdx.x];
//         }

//         __syncthreads();
//     }

//     // Apply ReLU and write the output (coalesced write)
//     if (m < M && n < N) {
//         out[b * M + m] = fmaxf(val + bias[m], 0.0f);
//     }
// }

__global__ void LinearKernelTiledWithRelu(const float * __restrict__ in, 
                                   const float * __restrict__ w, 
                                   const float * __restrict__ bias, 
                                   float * __restrict__ out,
                                   size_t B, size_t N, size_t M) 
{
    // Each block processes one batch element (z-dimension)
    size_t b = blockIdx.z;

    // m: output feature index
    size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // We only have one block in x-dim, since we loop over N tiles inside the kernel
    // threadIdx.x indexes a chunk of 4 floats along N dimension
    // nBase is the starting N index for this thread in float terms
    size_t nBase = threadIdx.x * 4;

    // Shared memory for tiles
    __shared__ float4 tileIn[2][BLOCK_SIZE][BLOCK_SIZE/4]; 
    __shared__ float4 tileW[2][BLOCK_SIZE][BLOCK_SIZE/4];

    float val = 0.0f;
    int bufIdx = 0;

    // Number of N-tiles
    size_t numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t t = 0; t < numTiles; t++) {
        // Compute the N offset for this tile
        size_t nOffset = t * BLOCK_SIZE;
        
        // Load W tile:
        float4 wVal = make_float4(0.f, 0.f, 0.f, 0.f);
        size_t wN = nOffset + nBase; // wN is the global N index for w
        if (m < M && (wN + 3) < N) {
            wVal = *(const float4*)(&w[m * N + wN]);
        }
        tileW[bufIdx][threadIdx.y][threadIdx.x] = wVal;

        // Load In tile:
        // For 'in', we don't have M dimension. We just have B and N.
        // Each thread will load one float4 from the input vector for the given batch element `b`.
        float4 iVal = make_float4(0.f, 0.f, 0.f, 0.f);
        if (b < B && (wN + 3) < N) {
            iVal = *(const float4*)(&in[b * N + wN]);
        }

        // Since in doesn't depend on m, but we have a tile shaped [BLOCK_SIZE][BLOCK_SIZE/4],
        // we just replicate the same iVal for each threadIdx.y. This is redundant but simple.
        tileIn[bufIdx][threadIdx.y][threadIdx.x] = iVal;

        __syncthreads();

        // Dot product over the N chunk:
        // We have BLOCK_SIZE floats in N dimension, grouped in float4, so BLOCK_SIZE/4 chunks.
        for (int k = 0; k < BLOCK_SIZE/4; k++) {
            float4 wVec = tileW[bufIdx][threadIdx.y][k];
            float4 iVec = tileIn[bufIdx][threadIdx.y][k];
            val += wVec.x * iVec.x + wVec.y * iVec.y + wVec.z * iVec.z + wVec.w * iVec.w;
        }

        __syncthreads();
        bufIdx = 1 - bufIdx;
    }

    // Write the result if in range
    // Only threadIdx.x == 0 corresponds to writing the result since we accumulate over all N 
    if (b < B && m < M && threadIdx.x == 0) {
        out[b * M + m] = val + bias[m] > 0? val + bias[m]: 0;
    }
}


__global__ void LinearKernelTiled2(const float * __restrict__ in, 
                                   const float * __restrict__ w, 
                                   const float * __restrict__ bias, 
                                   float * __restrict__ out,
                                   size_t B, size_t N, size_t M) 
{
    // Each block processes one batch element (z-dimension)
    size_t b = blockIdx.z;

    // m: output feature index
    size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // We only have one block in x-dim, since we loop over N tiles inside the kernel
    // threadIdx.x indexes a chunk of 4 floats along N dimension
    // nBase is the starting N index for this thread in float terms
    size_t nBase = threadIdx.x * 4;

    // Shared memory for tiles
    __shared__ float4 tileIn[2][BLOCK_SIZE][BLOCK_SIZE/4]; 
    __shared__ float4 tileW[2][BLOCK_SIZE][BLOCK_SIZE/4];

    float val = 0.0f;
    int bufIdx = 0;

    // Number of N-tiles
    size_t numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t t = 0; t < numTiles; t++) {
        // Compute the N offset for this tile
        size_t nOffset = t * BLOCK_SIZE;
        
        // Load W tile:
        float4 wVal = make_float4(0.f, 0.f, 0.f, 0.f);
        size_t wN = nOffset + nBase; // wN is the global N index for w
        if (m < M && (wN + 3) < N) {
            wVal = *(const float4*)(&w[m * N + wN]);
        }
        tileW[bufIdx][threadIdx.y][threadIdx.x] = wVal;

        // Load In tile:
        // For 'in', we don't have M dimension. We just have B and N.
        // Each thread will load one float4 from the input vector for the given batch element `b`.
        float4 iVal = make_float4(0.f, 0.f, 0.f, 0.f);
        if (b < B && (wN + 3) < N) {
            iVal = *(const float4*)(&in[b * N + wN]);
        }

        // Since in doesn't depend on m, but we have a tile shaped [BLOCK_SIZE][BLOCK_SIZE/4],
        // we just replicate the same iVal for each threadIdx.y. This is redundant but simple.
        tileIn[bufIdx][threadIdx.y][threadIdx.x] = iVal;

        __syncthreads();

        // Dot product over the N chunk:
        // We have BLOCK_SIZE floats in N dimension, grouped in float4, so BLOCK_SIZE/4 chunks.
        for (int k = 0; k < BLOCK_SIZE/4; k++) {
            float4 wVec = tileW[bufIdx][threadIdx.y][k];
            float4 iVec = tileIn[bufIdx][threadIdx.y][k];
            val += wVec.x * iVec.x + wVec.y * iVec.y + wVec.z * iVec.z + wVec.w * iVec.w;
        }

        __syncthreads();
        bufIdx = 1 - bufIdx;
    }

    // Write the result if in range
    // Only threadIdx.x == 0 corresponds to writing the result since we accumulate over all N 
    if (b < B && m < M && threadIdx.x == 0) {
        out[b * M + m] = val + bias[m];
    }
}



// __global__ void LinearKernelTiled2(const float *in, const float *w, const float *bias, float *out,
//                                   size_t B, size_t N, size_t M) {
//     // Shared memory for storing tiles of input and weight matrices (double buffering)
//     __shared__ float tileIn[2][BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float tileW[2][BLOCK_SIZE][BLOCK_SIZE];

//     // Batch index (each block processes one batch element)
//     size_t b = blockIdx.z;

//     // Output feature index (m) and input feature index (n) for the current thread
//     size_t m = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//     size_t n = blockIdx.x * BLOCK_SIZE + threadIdx.x;

//     // Accumulator for the dot product
//     float val = 0.0f;

//     // Double buffering index
//     int bufIdx = 0;

//     // Loop over tiles of the input and weight matrices
//     for (size_t t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
//         // Load a tile of the weight matrix into shared memory (if within bounds)
//         if (m < M && t * BLOCK_SIZE + threadIdx.x < N) {
//             tileW[bufIdx][threadIdx.y][threadIdx.x] = w[m * N + t * BLOCK_SIZE + threadIdx.x];
//         } else {
//             tileW[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
//         }

//         // Load a tile of the input matrix into shared memory (if within bounds)
//         if (n < N && t * BLOCK_SIZE + threadIdx.y < N) {
//             tileIn[bufIdx][threadIdx.y][threadIdx.x] = in[b * N + t * BLOCK_SIZE + threadIdx.y];
//         } else {
//             tileIn[bufIdx][threadIdx.y][threadIdx.x] = 0.0f;
//         }

//         // Synchronize to ensure all threads have loaded their tiles
//         __syncthreads();

//         // Compute partial dot product for the current tile
//         for (int k = 0; k < BLOCK_SIZE; k++) {
//             val += tileW[bufIdx][threadIdx.y][k] * tileIn[bufIdx][k][threadIdx.x];
//         }

//         // Synchronize before loading the next tile
//         __syncthreads();

//         // Toggle the buffer index for double buffering
//         bufIdx = 1 - bufIdx;
//     }

//     // Write the result to the output tensor (if within bounds) and add bias
//     if (m < M && n < N) {
//         out[b * M + m] = val + bias[m];
//     }
// }


#define SCALE 1.0f  // Example scaling factor; tune this based on your data

__global__ void TensorCoreLinearReluKernel(float *in, float *w, float *bias, float *out, size_t B, size_t N, size_t M) {
    int batch = blockIdx.z;
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 4;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 4;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // Shared memory for storing converted half-precision tiles
    __shared__ half in_half[WMMA_M * WMMA_K];
    __shared__ half w_half[WMMA_K * WMMA_N];
    __shared__ float bias_float[WMMA_N];

    // Accumulator for FP32 precision
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Convert bias to FP32
    if (threadIdx.x < WMMA_N && threadIdx.y == 0) {
        bias_float[threadIdx.x] = bias[warpN * WMMA_N + threadIdx.x];
    }

    __syncthreads();

    // Loop over tiles of the input and weight matrices
    for (int k = 0; k < (N + WMMA_K - 1) / WMMA_K; ++k) {
        int input_idx = batch * N * WMMA_M + warpM * WMMA_M * N + k * WMMA_K;
        int weight_idx = k * WMMA_K * M + warpN * WMMA_N;

        if (input_idx < B * N * WMMA_M && weight_idx < N * M) {
            // Convert a tile of the input to half-precision with scaling
            for (int i = 0; i < WMMA_M; i++) {
                for (int j = 0; j < WMMA_K; j++) {
                    int idx = i * N + j;
                    in_half[i * WMMA_K + j] = __float2half(in[input_idx + idx] * SCALE);
                }
            }

            // Convert a tile of the weights to half-precision
            for (int i = 0; i < WMMA_K; i++) {
                for (int j = 0; j < WMMA_N; j++) {
                    int idx = i * M + j;
                    w_half[i * WMMA_N + j] = __float2half(w[weight_idx + idx] * SCALE);
                }
            }

            __syncthreads();

            // Load half-precision tiles into Tensor Core fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, in_half, WMMA_K);
            wmma::load_matrix_sync(b_frag, w_half, WMMA_N);

            // Perform matrix multiplication using Tensor Cores
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    // Add bias and apply ReLU with rescaling
    for (int i = 0; i < acc.num_elements; ++i) {
        float result = acc.x[i] / (SCALE * SCALE);
        result += bias_float[i % WMMA_N];
        acc.x[i] = result > 0 ? result : 0;  // ReLU
    }

    // Store the result in the output (FP32)
    int output_idx = batch * M * WMMA_M + warpM * WMMA_M * M + warpN * WMMA_N;
    if (output_idx < B * M * WMMA_M) {
        wmma::store_matrix_sync(out + output_idx, acc, M, wmma::mem_row_major);
    }
}

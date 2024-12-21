////변경가능
#pragma once

#include "tensor.h"
#define LAYER_H
#include <cuda_fp16.h>


/* Operations (layers) */
void Embedding(int *in, Tensor *w, Tensor *out);
void Permute(Tensor *in, Tensor *out);
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Example of using CUDA kernel */
void ReLU_CUDA(Tensor *inout);

__global__ void EmbeddingKernel(int *in, float *w, float *out, size_t B, size_t S, size_t H);
__global__ void EmbeddingKernelOptimized(int *in, float *w, float *out, size_t B, size_t S, size_t H);
__global__ void PermuteKernel(float *in, float *out, size_t b, size_t s, size_t h);
__global__ void EmbeddingPermuteKernel(int *in, float *w, float *out, size_t B, size_t S, size_t H);
__global__ void Conv1DKernel(float *in, float *w, float *bias, float *out,
                             size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void Conv1DKernelTiled(
    const float *in, const float *w, const float *bias, float *out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void Conv1DReLUAndMaxPoolKernel(
    const float *in, const float *w, const float *bias, float *out, float *pool_out,
    size_t B, size_t C, size_t s, size_t OC, size_t K, size_t pool_size);
__global__ void Conv1DKernelFloat4Tiled(
    const float * __restrict__ in, 
    const float * __restrict__ w, 
    const float * __restrict__ bias, 
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void ReLU_Kernel(float *in, size_t num_elem);
__global__ void GetMaxKernel(float *in, float *out, size_t B, size_t C, size_t S);
__global__ void ReLUKernel(float *inout, size_t N);
__global__ void ReLUKernelVectorized(float *inout, size_t N);
__global__ void ConcatKernel(const float *in1, const float *in2, const float *in3, const float *in4,
                             float *out, size_t B, size_t N1, size_t N2, size_t N3, size_t N4);
__global__ void Conv1DKernelTiled3(
    const float * in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K) ;
__global__ void Conv1DKernelTiled5(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void Conv1DKernelTiled7(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void Conv1DKernelTiled9(
    const float *in,
    const float * __restrict__ w,
    const float * __restrict__ bias,
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void Conv1DKernelFloat4Tiled(
    const float * __restrict__ in, 
    const float * __restrict__ w, 
    const float * __restrict__ bias, 
    float * __restrict__ out,
    size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void ConcatKernelOneN(const float *in1, const float *in2, const float *in3, const float *in4,
                             float *out, size_t B, size_t N);
__global__ void LinearKernel(const float *in, const float *w, const float *bias, float *out,
                             size_t B, size_t N, size_t M);
__global__ void LinearKernelTiled(const float *in, const float *w, const float *bias, float *out,
                                  size_t B, size_t N, size_t M);
__global__ void LinearKernelTiledWithRelu(const float * __restrict__ in, 
                                   const float * __restrict__ w, 
                                   const float * __restrict__ bias, 
                                   float * __restrict__ out,
                                   size_t B, size_t N, size_t M);
__global__ void LinearKernelTiled2(const float * __restrict__ in, 
                                   const float * __restrict__ w, 
                                   const float * __restrict__ bias, 
                                   float * __restrict__ out,
                                   size_t B, size_t N, size_t M) ;                             
__global__ void TensorCoreLinearReluKernel(float *in, float *w, float *bias, float *out, size_t B, size_t N, size_t M);
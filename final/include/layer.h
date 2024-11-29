////변경가능
#pragma once

#include "tensor.h"


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
__global__ void PermuteKernel(float *in, float *out, size_t b, size_t s, size_t h);
__global__ void Conv1DKernel(float *in, float *w, float *bias, float *out,
                             size_t B, size_t C, size_t s, size_t OC, size_t K);
__global__ void ReLU_Kernel(float *inout, size_t N);

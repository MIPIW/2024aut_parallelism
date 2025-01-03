////변경가능
#pragma once

#include <vector>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using std::vector;

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


void convertFloatToHalf(float *input, half *output, size_t size);
void convertHalfToFloat(half *input, float *output, size_t size);


/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4] = {0,0,0,0};
  float *buf = nullptr;

  Tensor() = default;  // Default constructor
  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();


  size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;


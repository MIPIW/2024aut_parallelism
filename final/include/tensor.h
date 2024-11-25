////변경가능
#pragma once

#include <vector>
#include <cstdio>

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


/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4] = {0,0,0,0};
  float *buf = nullptr;

  Tensor() = default;
  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  void assign(const vector<size_t> &shape_){
    ndim = shape_.size();
    for(size_t i = 0; i < ndim ; ++i){
      shape[i] = shape_[i];
    }
    buf = nullptr;
  };

  size_t getNumParams(){
    size_t elems = 1;
    for (size_t i = 0; i<ndim; ++i){
      elems *= shape[i];
    };
    return elems;
  };
};

typedef Tensor Parameter;
typedef Tensor Activation;
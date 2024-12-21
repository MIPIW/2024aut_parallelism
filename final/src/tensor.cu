////변경가능

#include "model.h"
#include "tensor.h"
#define SCALE 0.01f  // Example scaling factor; tune this based on your data

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  buf = (float *) malloc(N_ * sizeof(float));
  memcpy(buf, buf_, N_ * sizeof(float));
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

__global__ void ConvertFloatToHalfKernel(float *input, half *output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Kernel to convert half array to float array
__global__ void ConvertHalfToFloatKernel(half *input, float *output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

void convertFloatToHalf(float *input, half *output, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    ConvertFloatToHalfKernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

void convertHalfToFloat(half *input, float *output, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    ConvertHalfToFloatKernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}
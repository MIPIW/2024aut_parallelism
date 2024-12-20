////변경가능

#include <mpi.h>
#include <cstdio>
// #include <tensor.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"
#include <pthread.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// uneditable
#define HIDDEN_DIM 4096
#define INPUT_CHANNEL HIDDEN_DIM
#define OUTPUT_CHANNEL 1024
#define SEQ_LEN 16
#define M0 2048
#define M1 1024
#define M2 512
#define M3 2 
#define BLOCK_SIZE_TC 16
#define NUM_GPUS_PER_NODE 4
// editable
#define BATCH_SIZE 32 // 최소 node(4) * batch개의 sentiment sample을 넣어야 함
#define ELEMENTS_PER_THREAD 4  // Number of elements each thread will process
#define BLOCK_SIZE 32
#define NUM_STREAMS 5

#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t err = call;                           \
        if (err != cudaSuccess) {                         \
            fprintf(stderr, "CUDA Error: %s\n",           \
                    cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */

typedef struct {
    int gpu_id;
    int *gpu_inputs;
    float *gpu_outputs;
    float *emb_wg;
    float *conv0_wg, *conv0_bg, *conv1_wg, *conv1_bg, *conv2_wg, *conv2_bg, *conv3_wg, *conv3_bg;
    float *linear0_wg, *linear0_bg, *linear1_wg, *linear1_bg, *linear2_wg, *linear2_bg, *linear3_wg, *linear3_bg;
    half *linear3_wgh, *linear3_bgh;

    float *emb_ag, *permute_ag;
    float *conv0_ag, *conv1_ag, *conv2_ag, *conv3_ag;
    float *concat_ag;
    float *pool0_ag, *pool1_ag, *pool2_ag, *pool3_ag;
    float *linear0_ag, *linear1_ag, *linear2_ag, *linear3_ag;
    half *linear2_agh, *linear3_agh;

    cudaStream_t stream[NUM_STREAMS];
} GPUContext;

typedef struct {
    int embeddingBlockDim, embeddingGridDim;
    int permuteBlockDim, permuteGridDim;

    dim3 convBlockDim, convGridDim0, convGridDim1, convGridDim2, convGridDim3;
    int getMaxBlockDim1, getMaxGridDim1;

    int reluBlockDim1, reluGridDim1, reluBlockDim2, reluGridDim2;
    int reluBlockDim3, reluGridDim3, reluBlockDim4, reluGridDim4;
    
    int concatBlockDim, concatGridDim;

    dim3 grid2D0, block1D0, grid2D1, block1D1;
    dim3 grid2D2, block1D2, grid2D3, block1D3;

    int reluBlockDimLin1, reluGridDimLin1;
    int reluBlockDimLin2, reluGridDimLin2;
    int reluBlockDimLin3, reluGridDimLin3;
} GridBlockDims;

GridBlockDims dims;
GPUContext contexts[NUM_STREAMS];
extern int * inputs;


void initializeGridAndBlockDimensions(GridBlockDims *dims) {

    // Embedding dimensions
    // dims->embeddingBlockDim = SEQ_LEN;
    // dims->embeddingGridDim = BATCH_SIZE;

    dims->embeddingBlockDim = BLOCK_SIZE;
    dims ->embeddingGridDim = (BATCH_SIZE * SEQ_LEN * HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    


    // Permute dimensions
    int permuteCount = BATCH_SIZE * SEQ_LEN * HIDDEN_DIM;
    dims->permuteBlockDim = BLOCK_SIZE;
    dims->permuteGridDim = (permuteCount + dims->permuteBlockDim - 1) / dims->permuteBlockDim;

    // Convolution dimensions
    dims->convBlockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dims->convGridDim0 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 3 + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // dims->convGridDim0 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims->convGridDim1 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 5 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims->convGridDim2 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 7 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims->convGridDim3 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 9 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Max pooling dimensions
    dims->getMaxBlockDim1 = BLOCK_SIZE;
    dims->getMaxGridDim1 = (BATCH_SIZE * OUTPUT_CHANNEL + dims->getMaxBlockDim1 - 1) / dims->getMaxBlockDim1;

    // ReLU dimensions for convolutional layers
    dims->reluBlockDim1 = BLOCK_SIZE;
    dims->reluGridDim1 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 2) + dims->reluBlockDim1 - 1) / dims->reluBlockDim1;
    dims->reluBlockDim2 = BLOCK_SIZE;
    dims->reluGridDim2 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 4) + dims->reluBlockDim2 - 1) / dims->reluBlockDim2;
    dims->reluBlockDim3 = BLOCK_SIZE;
    dims->reluGridDim3 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 6) + dims->reluBlockDim3 - 1) / dims->reluBlockDim3;
    dims->reluBlockDim4 = BLOCK_SIZE;
    dims->reluGridDim4 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 8) + dims->reluBlockDim4 - 1) / dims->reluBlockDim4;

    // Concat dimensions
    int concatCount = BATCH_SIZE * OUTPUT_CHANNEL * 4;
    dims->concatBlockDim = BLOCK_SIZE;
    dims->concatGridDim = (concatCount + ELEMENTS_PER_THREAD * dims->concatBlockDim - 1) / (ELEMENTS_PER_THREAD * dims->concatBlockDim);

    // Linear dimensions
    
    // dims->grid2D0 = dim3((HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, (M0 + BLOCK_SIZE -1) / BLOCK_SIZE, BATCH_SIZE);
    // dims->block1D0 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    dims->grid2D0 = dim3((HIDDEN_DIM + BLOCK_SIZE_TC-1) / BLOCK_SIZE_TC, (M0 + BLOCK_SIZE_TC-1) / BLOCK_SIZE_TC, BATCH_SIZE);
    dims->block1D0 = dim3(BLOCK_SIZE, BLOCK_SIZE_TC/4);

    dims->grid2D1 = dim3((M0 + BLOCK_SIZE - 1) / BLOCK_SIZE, (M1 + BLOCK_SIZE-1) / BLOCK_SIZE, BATCH_SIZE);
    dims->block1D1 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    dims->grid2D2 = dim3((M1 + BLOCK_SIZE-1) / BLOCK_SIZE, (M2 + BLOCK_SIZE-1) / BLOCK_SIZE, BATCH_SIZE);
    dims->block1D2 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    // dims->grid2D3 = dim3((M2 + BLOCK_SIZE_TC-1) / BLOCK_SIZE_TC, (M3 + BLOCK_SIZE_TC-1) / BLOCK_SIZE_TC, BATCH_SIZE);
    // dims->block1D3 = dim3(BLOCK_SIZE, BLOCK_SIZE_TC/4);
    dims->grid2D3 = dim3((M2 + BLOCK_SIZE-1) / BLOCK_SIZE, (M3 + BLOCK_SIZE -1) / BLOCK_SIZE, BATCH_SIZE);
    dims->block1D3 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    // ReLU dimensions for linear layers
    dims->reluBlockDimLin1 = BLOCK_SIZE;
    dims->reluGridDimLin1 = (BATCH_SIZE * M0 + dims->reluBlockDimLin1 - 1) / dims->reluBlockDimLin1;

    dims->reluBlockDimLin2 = BLOCK_SIZE;
    dims->reluGridDimLin2 = (BATCH_SIZE * M1 + dims->reluBlockDimLin2 - 1) / dims->reluBlockDimLin2;

    dims->reluBlockDimLin3 = BLOCK_SIZE;
    dims->reluGridDimLin3 = (BATCH_SIZE * M2 + dims->reluBlockDimLin3 - 1) / dims->reluBlockDimLin3;

}

Parameter *emb_w;
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;
Parameter *linear3_w, *linear3_b;


void initGPUContextsParameters(GPUContext *contexts);
void initGPUContextsActivation(GPUContext *contexts);


void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  emb_w = new Parameter({21635, 4096}, param + pos);
  pos += 21635 * 4096; 

  conv0_w = new Parameter({1024, 4096, 3}, param + pos);
  pos += 1024 * 4096 * 3; 
  conv0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv1_w = new Parameter({1024, 4096, 5}, param + pos);
  pos += 1024 * 4096 * 5; 
  conv1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv2_w = new Parameter({1024, 4096, 7}, param + pos);
  pos += 1024 * 4096 * 7;
  conv2_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv3_w = new Parameter({1024, 4096, 9}, param + pos);
  pos += 1024 * 4096 * 9;
  conv3_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear0_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  linear0_b = new Parameter({2048}, param + pos);
  pos += 2048;

  linear1_w = new Parameter({1024, 2048}, param + pos);
  pos += 1024 * 2048;
  linear1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear2_w = new Parameter({512, 1024}, param + pos);
  pos += 512 * 1024;
  linear2_b = new Parameter({512}, param + pos);
  pos += 512;

  linear3_w = new Parameter({2, 512}, param + pos);
  pos += 2 * 512;
  linear3_b = new Parameter({2}, param + pos);
  pos += 2;

  initializeGridAndBlockDimensions(&dims);
  initGPUContextsParameters(contexts);

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  delete emb_w;
  delete conv0_w;
  delete conv0_b;
  delete conv1_w;
  delete conv1_b;
  delete conv2_w;
  delete conv2_b;
  delete conv3_w;
  delete conv3_b;
  delete linear0_w;
  delete linear0_b;
  delete linear1_w;
  delete linear1_b;
  delete linear2_w;
  delete linear2_b;
  delete linear3_w;
  delete linear3_b;
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *emb_a;
Activation *permute_a;
Activation *conv0_a, *relu0_a, *pool0_a;
Activation *conv1_a, *relu1_a, *pool1_a;
Activation *conv2_a, *relu2_a, *pool2_a;
Activation *conv3_a, *relu3_a, *pool3_a;
Activation *concat_a;
Activation *linear0_a, *linear1_a, *linear2_a, *linear3_a;

void alloc_activations() {
  emb_a = new Activation({BATCH_SIZE, SEQ_LEN, 4096});
  permute_a = new Activation({BATCH_SIZE, 4096, SEQ_LEN});
  conv0_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 2});
  pool0_a = new Activation({BATCH_SIZE, 1024});
  conv1_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 4});
  pool1_a = new Activation({BATCH_SIZE, 1024});
  conv2_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 6});
  pool2_a = new Activation({BATCH_SIZE, 1024});
  conv3_a = new Activation({BATCH_SIZE, 1024, SEQ_LEN - 8});
  pool3_a = new Activation({BATCH_SIZE, 1024});
  concat_a = new Activation({BATCH_SIZE, 4096});
  linear0_a = new Activation({BATCH_SIZE, 2048});
  linear1_a = new Activation({BATCH_SIZE, 1024});
  linear2_a = new Activation({BATCH_SIZE, 512});
  linear3_a = new Activation({BATCH_SIZE, 2});

  initGPUContextsActivation(contexts);
  
}


void initGPUContextsParameters(GPUContext *contexts){


    for (int i = 0; i < NUM_GPUS_PER_NODE; ++i) {
      contexts[i].gpu_id = i;
      CUDA_CHECK(cudaSetDevice(i));
      // Create multiple streams for concurrent memory operations
      // Create multiple streams for concurrent memory operations
      for (int j = 0; j < NUM_STREAMS; ++j) {
          CUDA_CHECK(cudaStreamCreate(&contexts[i].stream[j]));
      }

      // CUDA_CHECK(cudaStreamCreate(&contexts[i].stream));
      int stream_idx = 0;

      CUDA_CHECK(cudaMalloc(&contexts[i].emb_wg, emb_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].emb_wg, emb_w->buf, emb_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv0_wg, conv0_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv0_bg, conv0_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv0_wg, conv0_w->buf, conv0_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv0_bg, conv0_b->buf, conv0_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv1_wg, conv1_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv1_bg, conv1_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv1_wg, conv1_w->buf, conv1_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv1_bg, conv1_b->buf, conv1_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv2_wg, conv2_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv2_bg, conv2_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv2_wg, conv2_w->buf, conv2_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv2_bg, conv2_b->buf, conv2_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv3_wg, conv3_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].conv3_bg, conv3_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv3_wg, conv3_w->buf, conv3_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].conv3_bg, conv3_b->buf, conv3_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear0_wg, linear0_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear0_bg, linear0_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear0_wg, linear0_w->buf, linear0_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear0_bg, linear0_b->buf, linear0_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear1_wg, linear1_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear1_bg, linear1_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear1_wg, linear1_w->buf, linear1_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear1_bg, linear1_b->buf, linear1_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear2_wg, linear2_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear2_bg, linear2_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear2_wg, linear2_w->buf, linear2_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear2_bg, linear2_b->buf, linear2_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear3_wg, linear3_w->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear3_bg, linear3_b->num_elem() * sizeof(float)));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear3_wg, linear3_w->buf, linear3_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMemcpyAsync(contexts[i].linear3_bg, linear3_b->buf, linear3_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice, contexts[i].stream[stream_idx++ % NUM_STREAMS]));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear3_wgh, linear3_w->num_elem() * sizeof(half)));
      CUDA_CHECK(cudaMalloc(&contexts[i].linear3_bgh, linear3_b->num_elem() * sizeof(half)));
      convertFloatToHalf(contexts[i].linear3_wg, contexts[i].linear3_wgh, linear3_w->num_elem());
      convertFloatToHalf(contexts[i].linear3_bg, contexts[i].linear3_bgh, linear3_b->num_elem());

      
      // Synchronize all streams for this GPU
      for (int j = 0; j < NUM_STREAMS; ++j) {
        CUDA_CHECK(cudaStreamSynchronize(contexts[i].stream[j]));
      }
  }
}


void initGPUContextsActivation(GPUContext *contexts) {

    for (int i = 0; i < NUM_GPUS_PER_NODE; ++i) {
        contexts[i].gpu_id = i;
        CUDA_CHECK(cudaSetDevice(i));
        // Create multiple streams for concurrent memory operations
        // Create multiple streams for concurrent memory operations
        // CUDA_CHECK(cudaStreamCreate(&contexts[i].stream));
        int stream_idx = 0;

        CUDA_CHECK(cudaMallocAsync(&contexts[i].gpu_inputs, BATCH_SIZE * SEQ_LEN * sizeof(int), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].gpu_outputs, BATCH_SIZE * M3 * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].emb_ag, emb_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].permute_ag, permute_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].conv0_ag, conv0_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].conv1_ag, conv1_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].conv2_ag, conv2_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].conv3_ag, conv3_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].pool0_ag, pool0_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].pool1_ag, pool1_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].pool2_ag, pool2_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].pool3_ag, pool3_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].concat_ag, concat_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear0_ag, linear0_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear1_ag, linear1_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear2_ag, linear2_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear2_agh, linear3_a->num_elem() * sizeof(half), contexts[i].stream[stream_idx++ & NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear3_ag, linear3_a->num_elem() * sizeof(float), contexts[i].stream[stream_idx++ % NUM_STREAMS]));
        CUDA_CHECK(cudaMallocAsync(&contexts[i].linear3_agh, linear3_a->num_elem() * sizeof(half), contexts[i].stream[stream_idx++ & NUM_STREAMS]));

        // Synchronize all streams for this GPU
        for (int j = 0; j < NUM_STREAMS; ++j) {
          CUDA_CHECK(cudaStreamSynchronize(contexts[i].stream[j]));
        }
    }
}

void free_activations() {
  // delete emb_a;
  // delete permute_a;
  delete conv0_a;
  delete pool0_a;
  delete conv1_a;
  delete pool1_a;
  delete conv2_a;
  delete pool2_a;
  delete conv3_a;
  delete pool3_a;
  delete concat_a;
  delete linear0_a;
  delete linear1_a;
  delete linear2_a;
  delete linear3_a;
}

// const int NUM_GPUES_PER_NODE = 4;
// const int NUM_THREADS_PER_NODE = 4;


void freeGPUContexts(GPUContext *contexts) {
    for (int i = 0; i < NUM_GPUS_PER_NODE; ++i) {
        CUDA_CHECK(cudaSetDevice(contexts[i].gpu_id));

        CUDA_CHECK(cudaFree(contexts[i].emb_wg));
        CUDA_CHECK(cudaFree(contexts[i].conv0_wg));
        CUDA_CHECK(cudaFree(contexts[i].conv0_bg));
        CUDA_CHECK(cudaFree(contexts[i].conv0_ag));
        CUDA_CHECK(cudaFree(contexts[i].conv1_wg));
        CUDA_CHECK(cudaFree(contexts[i].conv1_bg));
        CUDA_CHECK(cudaFree(contexts[i].conv1_ag));
        CUDA_CHECK(cudaFree(contexts[i].conv2_wg));
        CUDA_CHECK(cudaFree(contexts[i].conv2_bg));
        CUDA_CHECK(cudaFree(contexts[i].conv2_ag));
        CUDA_CHECK(cudaFree(contexts[i].conv3_wg));
        CUDA_CHECK(cudaFree(contexts[i].conv3_bg));
        CUDA_CHECK(cudaFree(contexts[i].conv3_ag));
        CUDA_CHECK(cudaFree(contexts[i].linear0_wg));
        CUDA_CHECK(cudaFree(contexts[i].linear0_bg));
        CUDA_CHECK(cudaFree(contexts[i].linear1_wg));
        CUDA_CHECK(cudaFree(contexts[i].linear1_bg));
        CUDA_CHECK(cudaFree(contexts[i].linear2_wg));
        CUDA_CHECK(cudaFree(contexts[i].linear2_bg));
        CUDA_CHECK(cudaFree(contexts[i].linear3_wg));
        CUDA_CHECK(cudaFree(contexts[i].linear3_bg));


        CUDA_CHECK(cudaFree(contexts[i].gpu_inputs));
        CUDA_CHECK(cudaFree(contexts[i].gpu_outputs));
        CUDA_CHECK(cudaFree(contexts[i].emb_ag));
        CUDA_CHECK(cudaFree(contexts[i].permute_ag));
        CUDA_CHECK(cudaFree(contexts[i].pool0_ag));
        CUDA_CHECK(cudaFree(contexts[i].pool1_ag));
        CUDA_CHECK(cudaFree(contexts[i].pool2_ag));
        CUDA_CHECK(cudaFree(contexts[i].pool3_ag));
        CUDA_CHECK(cudaFree(contexts[i].concat_ag));
        CUDA_CHECK(cudaFree(contexts[i].linear0_ag));
        CUDA_CHECK(cudaFree(contexts[i].linear1_ag));
        CUDA_CHECK(cudaFree(contexts[i].linear2_ag));
        CUDA_CHECK(cudaFree(contexts[i].linear3_ag));
        
      for (int j=0; j< NUM_STREAMS; ++j){
        if (contexts[i].stream[j] != nullptr){
          CUDA_CHECK(cudaStreamDestroy(contexts[i].stream[j]));
        }
      }
    }
}



/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // 총 16개, 노드당 4개, 배치당 2개, 총 2 배치

  size_t samples_per_node = n_samples / mpi_size;
  int *local_inputs = (int *)malloc(samples_per_node * SEQ_LEN * sizeof(int));
  float *local_outputs = (float *)malloc(samples_per_node * M3 * sizeof(float));

  int mpi_status = MPI_Scatter(inputs, samples_per_node * SEQ_LEN, MPI_INT, local_inputs,
                                samples_per_node * SEQ_LEN, MPI_INT, 0, MPI_COMM_WORLD);

  size_t num_batches = samples_per_node / (BATCH_SIZE * NUM_GPUS_PER_NODE);

  // Launch processing on each GPU in parallel
  #pragma omp parallel for num_threads(NUM_GPUS_PER_NODE)
  for (int gpu_id = 0; gpu_id < NUM_GPUS_PER_NODE; ++gpu_id) {
      cudaSetDevice(contexts[gpu_id].gpu_id);

      for (size_t cur_batch = 0; cur_batch < num_batches; ++cur_batch) {
          int *batchInput = local_inputs + (cur_batch * NUM_GPUS_PER_NODE + gpu_id) * BATCH_SIZE * SEQ_LEN;
          float *batchOutput = local_outputs + (cur_batch * NUM_GPUS_PER_NODE + gpu_id) * BATCH_SIZE * M3;

        // Asynchronous copy input data to GPU
        CUDA_CHECK(cudaMemcpyAsync(contexts[gpu_id].gpu_inputs, batchInput,
                        BATCH_SIZE * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice, contexts[gpu_id].stream[0]));

        
        // Embedding layer
        EmbeddingPermuteKernel<<<dims.embeddingGridDim, dims.embeddingBlockDim, 0, contexts[gpu_id].stream[0]>>>(
          contexts[gpu_id].gpu_inputs, contexts[gpu_id].emb_wg, contexts[gpu_id].permute_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);

        CUDA_CHECK(cudaStreamSynchronize(contexts[gpu_id].stream[0]));

        // EmbeddingKernel<<<dims.embeddingGridDim, dims.embeddingBlockDim, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].gpu_inputs, contexts[gpu_id].emb_wg, contexts[gpu_id].emb_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);

        // // Permute layer
        // PermuteKernel<<<dims.permuteGridDim, dims.permuteBlockDim, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].emb_ag, contexts[gpu_id].permute_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);

        // Convolutional and pooling layers
        // Conv2 (Kernel Size 3)
        // Conv1DReLUAndMaxPoolKernel<<<dims.convGridDim0, dims.convBlockDim, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].permute_ag, contexts[gpu_id].conv0_wg, contexts[gpu_id].conv0_bg, contexts[gpu_id].conv0_ag, contexts[gpu_id].pool0_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 3, SEQ_LEN-2);
        
        Conv1DKernelTiled<<<dims.convGridDim0, dims.convBlockDim, 0, contexts[gpu_id].stream[0]>>>(
            contexts[gpu_id].permute_ag, contexts[gpu_id].conv0_wg, contexts[gpu_id].conv0_bg, contexts[gpu_id].conv0_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 3);
        // ReLUKernel<<<dims.reluGridDim1, dims.reluBlockDim1, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].conv0_ag, conv0_a->num_elem());
        GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1, 0, contexts[gpu_id].stream[0]>>>(
            contexts[gpu_id].conv0_ag, contexts[gpu_id].pool0_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN - 2);
        // Conv2 (Kernel Size 5)
        Conv1DKernelTiled<<<dims.convGridDim1, dims.convBlockDim, 0, contexts[gpu_id].stream[1]>>>(
          contexts[gpu_id].permute_ag, contexts[gpu_id].conv1_wg, contexts[gpu_id].conv1_bg, contexts[gpu_id].conv1_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 5);
        // ReLUKernel<<<dims.reluGridDim2, dims.reluBlockDim2, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].conv1_ag, conv1_a->num_elem());
        GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1, 0, contexts[gpu_id].stream[1]>>>(
            contexts[gpu_id].conv1_ag, contexts[gpu_id].pool1_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN - 4);
        // Conv2 (Kernel Size 7)
        Conv1DKernelTiled<<<dims.convGridDim2, dims.convBlockDim, 0, contexts[gpu_id].stream[2]>>>(
            contexts[gpu_id].permute_ag, contexts[gpu_id].conv2_wg, contexts[gpu_id].conv2_bg, contexts[gpu_id].conv2_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 7);
        // ReLUKernel<<<dims.reluGridDim3, dims.reluBlockDim3, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].conv2_ag, conv2_a->num_elem());
        GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1, 0, contexts[gpu_id].stream[2]>>>(
            contexts[gpu_id].conv2_ag, contexts[gpu_id].pool2_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN - 6);
        // Conv3 (Kernel Size 9)
        Conv1DKernelTiled<<<dims.convGridDim3, dims.convBlockDim, 0, contexts[gpu_id].stream[3]>>>(
            contexts[gpu_id].permute_ag, contexts[gpu_id].conv3_wg, contexts[gpu_id].conv3_bg, contexts[gpu_id].conv3_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 9);
        // ReLUKernel<<<dims.reluGridDim4, dims.reluBlockDim4, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].conv3_ag, conv3_a->num_elem());
        GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1, 0, contexts[gpu_id].stream[3]>>>(
            contexts[gpu_id].conv3_ag, contexts[gpu_id].pool3_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN - 8);

        for (int i = 0; i < 4; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(contexts[gpu_id].stream[i]));
          }
        // for (int j = 0; j < NUM_STREAMS; ++j) {
        //   CUDA_CHECK(cudaStreamSynchronize(contexts[gpu_id].stream[j]));
        // }

        // ConcatKernel<<<dims.concatGridDim, dims.concatBlockDim, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].pool0_ag, contexts[gpu_id].pool1_ag, contexts[gpu_id].pool2_ag, contexts[gpu_id].pool3_ag,
        //     contexts[gpu_id].concat_ag, BATCH_SIZE, pool0_a->num_elem(), pool1_a->num_elem(), pool2_a->num_elem(), pool3_a->num_elem());

        ConcatKernelOneN<<<dims.concatGridDim, dims.concatBlockDim, 0, contexts[gpu_id].stream[0]>>>(
            contexts[gpu_id].pool0_ag, contexts[gpu_id].pool1_ag, contexts[gpu_id].pool2_ag, contexts[gpu_id].pool3_ag,
            contexts[gpu_id].concat_ag, BATCH_SIZE, pool3_a->num_elem() / BATCH_SIZE);

        // Linear layers 1
        // LinearKernelTiledWithRelu<<<dims.grid2D0, dims.block1D0, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].concat_ag, contexts[gpu_id].linear0_wg, contexts[gpu_id].linear0_bg, contexts[gpu_id].linear0_ag, BATCH_SIZE, HIDDEN_DIM, M0);
        TensorCoreLinearReluKernel<<<dims.grid2D0, dims.block1D0, 0, contexts[gpu_id].stream[0]>>>(
          contexts[gpu_id].concat_ag, contexts[gpu_id].linear0_wg, contexts[gpu_id].linear0_bg, contexts[gpu_id].linear0_ag, BATCH_SIZE, HIDDEN_DIM, M0);
        
        // ReLUKernel<<<dims.reluGridDimLin1, dims.reluBlockDimLin1, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].linear0_ag, linear0_a->num_elem());
        // Linear layers 2
        LinearKernelTiledWithRelu<<<dims.grid2D1, dims.block1D1, 0, contexts[gpu_id].stream[0]>>>(
            contexts[gpu_id].linear0_ag, contexts[gpu_id].linear1_wg, contexts[gpu_id].linear1_bg, contexts[gpu_id].linear1_ag, BATCH_SIZE, M0, M1);
        // ReLUKernel<<<dims.reluGridDimLin2, dims.reluBlockDimLin2, 0, contexts[gpu_id].stream[0]>>>(
        //     contexts[gpu_id].linear1_ag, linear1_a->num_elem());
        // Linear layers 3
        LinearKernelTiledWithRelu<<<dims.grid2D2, dims.block1D2, 0, contexts[gpu_id].stream[0]>>>(
          contexts[gpu_id].linear1_ag, contexts[gpu_id].linear2_wg, contexts[gpu_id].linear2_bg, contexts[gpu_id].linear2_ag, BATCH_SIZE, M1, M2);
        // ReLUKernel<<<dims.reluGridDimLin3, dims.reluBlockDimLin3, 0, contexts[gpu_id].stream[0]>>>(
        //   contexts[gpu_id].linear2_ag, linear2_a->num_elem());
        // Linear layers 4

        // convertFloatToHalf(contexts[gpu_id].linear2_ag, contexts[gpu_id].linear2_agh, linear2_a->num_elem());
        // TensorCoreLinearReluKernel<<<dims.grid2D3, dims.block1D3, 0, contexts[gpu_id].stream[0]>>>(
        //   contexts[gpu_id].linear2_ag, contexts[gpu_id].linear3_wg, contexts[gpu_id].linear3_bg, contexts[gpu_id].linear3_ag, BATCH_SIZE, M2, M3);
        
        LinearKernelTiled2<<<dims.grid2D3, dims.block1D3, 0, contexts[gpu_id].stream[0]>>>(
          contexts[gpu_id].linear2_ag, contexts[gpu_id].linear3_wg, contexts[gpu_id].linear3_bg, contexts[gpu_id].linear3_ag, BATCH_SIZE, M2, M3);

        // Copy the final output back to host asynchronously
        cudaStreamSynchronize(contexts[gpu_id].stream[0]);
        cudaMemcpyAsync(batchOutput, contexts[gpu_id].linear3_ag, BATCH_SIZE * M3 * sizeof(float),
                        cudaMemcpyDeviceToHost, contexts[gpu_id].stream[4]);
    }

    // Synchronize all GPUs to ensure completion
    
  }
  for (int gpu_id = 0; gpu_id < NUM_GPUS_PER_NODE; ++gpu_id) {
          cudaSetDevice(contexts[gpu_id].gpu_id);
          cudaStreamSynchronize(contexts[gpu_id].stream[1]);
      }

  // // Gather outputs from all nodes
  MPI_Gather(local_outputs, samples_per_node * N_CLASSES, MPI_FLOAT, outputs,
            samples_per_node * N_CLASSES, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (mpi_status != MPI_SUCCESS) {
      fprintf(stderr, "MPI_Gather failed\n");
      MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  // // Free local buffers

  freeGPUContexts(contexts);


}

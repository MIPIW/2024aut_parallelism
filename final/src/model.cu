////변경가능

#include <mpi.h>
#include <cstdio>
#include <tensor.h>
#include "layer.h"
#include "model.h"
#include <pthread.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#define BATCH_SIZE 2 // 최소 node(4) * batch개의 sentiment sample을 넣어야 함
#define HIDDEN_DIM 4096
#define INPUT_CHANNEL HIDDEN_DIM
#define OUTPUT_CHANNEL 1024
#define M0 2048
#define M1 1024
#define M2 512
#define M3 2 
#define ELEMENTS_PER_THREAD 4  // Number of elements each thread will process
#define BLOCK_SIZE 32
      
/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w;
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;
Parameter *linear3_w, *linear3_b;

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

typedef struct {
  int gpu_id;
  float *A;
  float *B;
  float *C;
  int M, N, K;
} ThreadData;

struct GridBlockDims {
    int embeddingBlockDim;
    int embeddingGridDim;
    int permuteBlockDim;
    int permuteGridDim;

    dim3 convBlockDim;
    dim3 convGridDim0;
    dim3 convGridDim1;
    dim3 convGridDim2;
    dim3 convGridDim3;

    int getMaxBlockDim1;
    int getMaxGridDim1;

    int reluBlockDim1;
    int reluGridDim1;
    int reluBlockDim2;
    int reluGridDim2;
    int reluBlockDim3;
    int reluGridDim3;
    int reluBlockDim4;
    int reluGridDim4;

    int concatBlockDim;
    int concatGridDim;

    dim3 grid2D0;
    dim3 block1D0;
    dim3 grid2D1;
    dim3 block1D1;
    dim3 grid2D2;
    dim3 block1D2;
    dim3 grid2D3;
    dim3 block1D3;

    int reluBlockDimLin1;
    int reluGridDimLin1;
    int reluBlockDimLin2;
    int reluGridDimLin2;
    int reluBlockDimLin3;
    int reluGridDimLin3;
};

GridBlockDims initializeGridAndBlockDimensions() {
    GridBlockDims dims;

    // Embedding dimensions
    dims.embeddingBlockDim = SEQ_LEN;
    dims.embeddingGridDim = BATCH_SIZE;

    // Permute dimensions
    int permuteCount = BATCH_SIZE * SEQ_LEN * HIDDEN_DIM;
    dims.permuteBlockDim = BLOCK_SIZE;
    dims.permuteGridDim = (permuteCount + dims.permuteBlockDim - 1) / dims.permuteBlockDim;

    // Convolution dimensions
    dims.convBlockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dims.convGridDim0 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims.convGridDim1 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 5 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims.convGridDim2 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 7 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dims.convGridDim3 = dim3(BATCH_SIZE, (OUTPUT_CHANNEL + BLOCK_SIZE - 1) / BLOCK_SIZE, (SEQ_LEN - 9 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Max pooling dimensions
    dims.getMaxBlockDim1 = BLOCK_SIZE;
    dims.getMaxGridDim1 = (BATCH_SIZE * OUTPUT_CHANNEL + dims.getMaxBlockDim1 - 1) / dims.getMaxBlockDim1;

    // ReLU dimensions for convolutional layers
    dims.reluBlockDim1 = BLOCK_SIZE;
    dims.reluGridDim1 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 2) + dims.reluBlockDim1 - 1) / dims.reluBlockDim1;
    dims.reluBlockDim2 = BLOCK_SIZE;
    dims.reluGridDim2 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 4) + dims.reluBlockDim2 - 1) / dims.reluBlockDim2;
    dims.reluBlockDim3 = BLOCK_SIZE;
    dims.reluGridDim3 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 6) + dims.reluBlockDim3 - 1) / dims.reluBlockDim3;
    dims.reluBlockDim4 = BLOCK_SIZE;
    dims.reluGridDim4 = (BATCH_SIZE * OUTPUT_CHANNEL * (SEQ_LEN - 8) + dims.reluBlockDim4 - 1) / dims.reluBlockDim4;

    // Concat dimensions
    int concatCount = BATCH_SIZE * OUTPUT_CHANNEL * 4;
    dims.concatBlockDim = BLOCK_SIZE;
    dims.concatGridDim = (concatCount + ELEMENTS_PER_THREAD * dims.concatBlockDim - 1) / (ELEMENTS_PER_THREAD * dims.concatBlockDim);

    // Linear dimensions
    dims.grid2D0 = dim3((HIDDEN_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE, (M0 + BLOCK_SIZE -1) / BLOCK_SIZE, BATCH_SIZE);
    dims.block1D0 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    dims.grid2D1 = dim3((M0 + BLOCK_SIZE - 1) / BLOCK_SIZE, (M1 + BLOCK_SIZE-1) / BLOCK_SIZE, BATCH_SIZE);
    dims.block1D1 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    dims.grid2D2 = dim3((M1 + BLOCK_SIZE-1) / BLOCK_SIZE, (M2 + BLOCK_SIZE-1) / BLOCK_SIZE, BATCH_SIZE);
    dims.block1D2 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    dims.grid2D3 = dim3((M2 + BLOCK_SIZE-1) / BLOCK_SIZE, (M3 + BLOCK_SIZE-1) / BLOCK_SIZE, BATCH_SIZE);
    dims.block1D3 = dim3(BLOCK_SIZE, BLOCK_SIZE);

    // ReLU dimensions for linear layers
    dims.reluBlockDimLin1 = BLOCK_SIZE;
    dims.reluGridDimLin1 = (BATCH_SIZE * M0 + dims.reluBlockDimLin1 - 1) / dims.reluBlockDimLin1;

    dims.reluBlockDimLin2 = BLOCK_SIZE;
    dims.reluGridDimLin2 = (BATCH_SIZE * M1 + dims.reluBlockDimLin2 - 1) / dims.reluBlockDimLin2;

    dims.reluBlockDimLin3 = BLOCK_SIZE;
    dims.reluGridDimLin3 = (BATCH_SIZE * M2 + dims.reluBlockDimLin3 - 1) / dims.reluBlockDimLin3;

    return dims;
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


  // Calculate the number of samples per node
  
  // 총 16개, 노드당 4개, 배치당 2개, 총 2 배치
  size_t samples_per_node = n_samples / mpi_size; 

  int *local_inputs = (int *)malloc(samples_per_node * SEQ_LEN * sizeof(int));
  float *local_outputs = (float *)malloc(samples_per_node * SEQ_LEN * sizeof(float));  
  if (!local_outputs) {
      fprintf(stderr, "Failed to allocate memory for local_outputs\n");
      exit(EXIT_FAILURE);
  }

  // non-weight
  int * gpu_mem_inputs = nullptr;
  float * gpu_mem_outputs = nullptr;
  cudaMalloc(&gpu_mem_inputs, BATCH_SIZE * SEQ_LEN * sizeof(int));
  cudaMalloc(&gpu_mem_outputs, BATCH_SIZE * N_CLASSES * sizeof(float));

  float *emb_ag, *permute_ag;
  cudaMalloc(&emb_ag, emb_a->num_elem() * sizeof(float));
  cudaMalloc(&permute_ag, permute_a->num_elem() * sizeof(float));
  float *pool0_ag, *pool1_ag, *pool2_ag, *pool3_ag;
  cudaMalloc(&pool0_ag, pool0_a->num_elem() * sizeof(float));
  cudaMalloc(&pool1_ag, pool1_a->num_elem() * sizeof(float));
  cudaMalloc(&pool2_ag, pool2_a->num_elem() * sizeof(float));
  cudaMalloc(&pool3_ag, pool3_a->num_elem() * sizeof(float));
  float *concat_ag;
  cudaMalloc(&concat_ag, concat_a->num_elem() * sizeof(float));
  float *linear0_ag, *linear1_ag, *linear2_ag, *linear3_ag;
  cudaMalloc(&linear0_ag, linear0_a->num_elem() * sizeof(float));
  cudaMalloc(&linear1_ag, linear1_a->num_elem() * sizeof(float));
  cudaMalloc(&linear2_ag, linear2_a->num_elem() * sizeof(float));
  cudaMalloc(&linear3_ag, linear3_a->num_elem() * sizeof(float));


  // // tensor buf 다 nullptr로 초기화하는 걸로 바꾸기
  // cudaMalloc(&emb_a->buf, emb_a->num_elem() * sizeof(float));
  // cudaMalloc(&permute_a->buf, permute_a->num_elem() * sizeof(float));

  // weights, shouold be cop
  float * emb_wg = nullptr;
  cudaMalloc(&emb_wg, emb_w->num_elem() * sizeof(float));
  cudaMemcpy(emb_wg, emb_w->buf, emb_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  float *conv0_wg, *conv0_bg, *conv0_ag = nullptr;
  cudaMalloc(&conv0_wg, conv0_w->num_elem() * sizeof(float));
  cudaMalloc(&conv0_bg, conv0_b->num_elem() * sizeof(float));
  cudaMalloc(&conv0_ag, conv0_a->num_elem() * sizeof(float));
  cudaMemcpy(conv0_wg, conv0_w->buf, conv0_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv0_bg, conv0_b->buf, conv0_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv0_ag, conv0_a->buf, conv0_a->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  float *conv1_wg, *conv1_bg, *conv1_ag = nullptr;
  cudaMalloc(&conv1_wg, conv1_w->num_elem() * sizeof(float));
  cudaMalloc(&conv1_bg, conv1_b->num_elem() * sizeof(float));
  cudaMalloc(&conv1_ag, conv1_a->num_elem() * sizeof(float));
  cudaMemcpy(conv1_wg, conv1_w->buf, conv1_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv1_bg, conv1_b->buf, conv1_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv1_ag, conv1_a->buf, conv1_a->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  float *conv2_wg, *conv2_bg, *conv2_ag = nullptr;
  cudaMalloc(&conv2_wg, conv2_w->num_elem() * sizeof(float));
  cudaMalloc(&conv2_bg, conv2_b->num_elem() * sizeof(float));
  cudaMalloc(&conv2_ag, conv2_a->num_elem() * sizeof(float));
  cudaMemcpy(conv2_wg, conv2_w->buf, conv2_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv2_bg, conv2_b->buf, conv2_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv2_ag, conv2_a->buf, conv2_a->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  float *conv3_wg, *conv3_bg, *conv3_ag = nullptr;
  cudaMalloc(&conv3_wg, conv3_w->num_elem() * sizeof(float));
  cudaMalloc(&conv3_bg, conv3_b->num_elem() * sizeof(float));
  cudaMalloc(&conv3_ag, conv3_a->num_elem() * sizeof(float));
  cudaMemcpy(conv3_wg, conv3_w->buf, conv3_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv3_bg, conv3_b->buf, conv3_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(conv3_ag, conv3_a->buf, conv3_a->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  float *linear0_wg, *linear0_bg;
  float *linear1_wg, *linear1_bg;
  float *linear2_wg, *linear2_bg;
  float *linear3_wg, *linear3_bg;
  cudaMalloc(&linear0_wg, linear0_w->num_elem() * sizeof(float));
  cudaMalloc(&linear1_wg, linear1_w->num_elem() * sizeof(float));
  cudaMalloc(&linear2_wg, linear2_w->num_elem() * sizeof(float));
  cudaMalloc(&linear3_wg, linear3_w->num_elem() * sizeof(float));
  cudaMalloc(&linear0_bg, linear0_b->num_elem() * sizeof(float));
  cudaMalloc(&linear1_bg, linear1_b->num_elem() * sizeof(float));
  cudaMalloc(&linear2_bg, linear2_b->num_elem() * sizeof(float));
  cudaMalloc(&linear3_bg, linear3_b->num_elem() * sizeof(float));
  cudaMemcpy(linear0_wg, linear0_w->buf, linear0_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear1_wg, linear1_w->buf, linear1_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear2_wg, linear2_w->buf, linear2_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear3_wg, linear3_w->buf, linear3_w->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear0_bg, linear0_b->buf, linear0_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear1_bg, linear1_b->buf, linear1_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear2_bg, linear2_b->buf, linear2_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(linear3_bg, linear3_b->buf, linear3_b->num_elem() * sizeof(float), cudaMemcpyHostToDevice);
  
  

  // 노드에 n개 인풋 퍼뜨림(32 -> 8)
  int mpi_status = MPI_Scatter(inputs, samples_per_node * SEQ_LEN, MPI_INT, local_inputs,
              samples_per_node * SEQ_LEN, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_status != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Scatter failed\n");
    exit(EXIT_FAILURE);
  }

  GridBlockDims dims = initializeGridAndBlockDimensions();

  // 퍼뜨린 값을 4개 device에 퍼뜨리지 않고 일단은
  // 배치사이즈 2개씩 한 번에 처리함 
  size_t num_batches = samples_per_node / BATCH_SIZE;
  for (size_t cur_batch = 0; cur_batch < num_batches; ++cur_batch){
    int * batchInput = local_inputs + cur_batch * BATCH_SIZE * SEQ_LEN;
      cudaMemcpy(gpu_mem_inputs, batchInput,
      BATCH_SIZE * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice);

      // [B * 16] -> [B * 16 * 4096]
      EmbeddingKernel<<<dims.embeddingGridDim, dims.embeddingBlockDim>>>(gpu_mem_inputs, emb_wg, emb_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);
      // [B * 16 * 4096] -> [B * 4096 * 16]
      PermuteKernel<<<dims.permuteGridDim, dims.permuteBlockDim>>>(emb_ag, permute_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);
      
      // [B * 4096 * 16] -> [B * 1024 * 14]
      Conv1DKernelTiled<<<dims.convGridDim0, dims.convBlockDim>>>(permute_ag, conv0_wg, conv0_bg, conv0_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 3);
      ReLUKernel<<<dims.reluGridDim1,dims.reluBlockDim1>>>(conv0_ag, conv0_a->num_elem());
      // [B * 1024 * 14] -> [B * 1024]
      GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1>>>(conv0_ag, pool0_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN-2);

      // [B * 4096] -> [B * 1024 * 12]
      Conv1DKernelTiled<<<dims.convGridDim1, dims.convBlockDim>>>(permute_ag, conv1_wg, conv1_bg, conv1_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 5);
      ReLUKernel<<<dims.reluGridDim2,dims.reluBlockDim2>>>(conv1_ag, conv1_a->num_elem());
      // [B * 1024 * 12] -> [B * 1024]
      GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1>>>(conv1_ag, pool1_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN-4);
      // [B * 4096] -> [B * 1024 * 10]
      Conv1DKernelTiled<<<dims.convGridDim2, dims.convBlockDim>>>(permute_ag, conv2_wg, conv2_bg, conv2_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 7);
      ReLUKernel<<<dims.reluGridDim3,dims.reluBlockDim3>>>(conv2_ag, conv2_a->num_elem());
      // [B * 1024 * 10] -> [B * 1024]
      GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1>>>(conv2_ag, pool2_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN-6);

      // [B * 4096] -> [B * 1024 * 8]
      Conv1DKernelTiled<<<dims.convGridDim3, dims.convBlockDim>>>(permute_ag, conv3_wg, conv3_bg, conv3_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 9);
      ReLUKernel<<<dims.reluGridDim4,dims.reluBlockDim4>>>(conv3_ag, conv3_a->num_elem());
      // [B * 1024 * 8] -> [B * 1024]
      GetMaxKernel<<<dims.getMaxGridDim1, dims.getMaxBlockDim1>>>(conv3_ag, pool3_ag, BATCH_SIZE, OUTPUT_CHANNEL, SEQ_LEN-8);

      // [B * 1024 * 4] -> [B * 4096]
      ConcatKernel<<<dims.concatGridDim, dims.concatBlockDim>>>(pool0_ag, pool1_ag, pool2_ag, pool3_ag, concat_ag, BATCH_SIZE, pool0_a->num_elem(), pool1_a->num_elem(),pool2_a->num_elem(),pool3_a->num_elem());
      // [B * 4096] -> [B * 2048]
      LinearKernelTiled<<<dims.grid2D0, dims.block1D0>>>(concat_ag, linear0_wg, linear0_bg, linear0_ag, BATCH_SIZE, HIDDEN_DIM, M0);
      ReLUKernel<<<dims.reluGridDimLin1, dims.reluBlockDimLin1>>>(linear0_ag, linear0_a->num_elem());
      // [B * 2048] -> [B * 1024]
      LinearKernelTiled<<<dims.grid2D1, dims.block1D1>>>(linear0_ag, linear1_wg, linear1_bg, linear1_ag, BATCH_SIZE, M0, M1);
      ReLUKernel<<<dims.reluGridDimLin2, dims.reluBlockDimLin2>>>(linear1_ag, linear1_a->num_elem());
      // [B * 1024] -> [B * 512]
      LinearKernelTiled<<<dims.grid2D2, dims.block1D2>>>(linear1_ag, linear2_wg, linear2_bg, linear2_ag, BATCH_SIZE, M1, M2);
      ReLUKernel<<<dims.reluGridDimLin3, dims.reluBlockDimLin3>>>(linear2_ag, linear2_a->num_elem());
      // [B * 512] -> [B * 2]
      LinearKernelTiled<<<dims.grid2D3, dims.block1D3>>>(linear2_ag, linear3_wg, linear3_bg, linear3_ag, BATCH_SIZE, M2, M3);
      cudaDeviceSynchronize();

      // cudaMemcpy(linear2_a->buf, linear2_ag,
      //   linear2_a->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(linear3_a->buf, linear3_ag,
        linear3_a->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);
      // Linear(linear0_a, linear1_w, linear1_b, linear1_a);
      // ReLU(linear1_a);

      /* in [1024] -> out [512] */
      // Linear(linear1_a, linear2_w, linear2_b, linear2_a);
      // ReLU(linear2_a);

      /* in [512] -> out [2] */
      // Linear(linear2_a, linear3_w, linear3_b, linear3_a);

      memcpy(local_outputs + cur_batch * BATCH_SIZE * 2, linear3_a->buf, BATCH_SIZE * 2 * sizeof(float));
    // cudaMemcpy(local_outputs + cur_batch * BATCH_SIZE * 2, gpu_mem_outputs,
    //            BATCH_SIZE * N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
  
  }

  // // Gather outputs from all nodes
  MPI_Gather(local_outputs, samples_per_node * N_CLASSES, MPI_FLOAT, outputs,
            samples_per_node * N_CLASSES, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (mpi_status != MPI_SUCCESS) {
      fprintf(stderr, "MPI_Gather failed\n");
      MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  // // Free local buffers

  cudaFree(gpu_mem_outputs);
  cudaFree(gpu_mem_inputs);
  cudaFree(emb_a->buf);
  cudaFree(permute_a->buf);
  free(local_inputs);
  free(local_outputs);


}

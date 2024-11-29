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
#define BATCH_SIZE 2
#define HIDDEN_DIM 4096
#define INPUT_CHANNEL HIDDEN_DIM
#define OUTPUT_CHANNEL 1024

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

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // 총 16개, 노드당 4개, 배치당 2개, 총 2 배치
  size_t samples_per_node = n_samples / mpi_size; 

  // Use cudaMallocHost for pinned memory
  //여기가 아닌것같은데 
  // int * local_inputs = nullptr;
  // int * local_outputs = nullptr;
  // cudaMallocHost(&local_inputs, samples_per_node * SEQ_LEN * sizeof(int));
  // cudaMallocHost(&local_outputs, samples_per_node * N_CLASSES * sizeof(float));

  int *local_inputs = (int *)malloc(samples_per_node * SEQ_LEN * sizeof(int));
  float *local_outputs = (float *)malloc(samples_per_node * SEQ_LEN * sizeof(float));  
  if (!local_outputs) {
      fprintf(stderr, "Failed to allocate memory for local_outputs\n");
      exit(EXIT_FAILURE);
  }

  int * gpu_mem_inputs = nullptr;
  float * gpu_mem_outputs = nullptr;
  cudaMalloc(&gpu_mem_inputs, BATCH_SIZE * SEQ_LEN * sizeof(int));
  cudaMalloc(&gpu_mem_outputs, BATCH_SIZE * N_CLASSES * sizeof(float));

  float *emb_ag, *permute_ag;
  cudaMalloc(&emb_ag, emb_a->num_elem() * sizeof(float));
  cudaMalloc(&permute_ag, permute_a->num_elem() * sizeof(float));

  // // tensor buf 다 nullptr로 초기화하는 걸로 바꾸기
  // cudaMalloc(&emb_a->buf, emb_a->num_elem() * sizeof(float));
  // cudaMalloc(&permute_a->buf, permute_a->num_elem() * sizeof(float));


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


  // Scatter input data to all nodes
  int mpi_status = MPI_Scatter(inputs, samples_per_node * SEQ_LEN, MPI_INT, local_inputs,
              samples_per_node * SEQ_LEN, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_status != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Scatter failed\n");
    exit(EXIT_FAILURE);
  }

  size_t num_batches = samples_per_node / BATCH_SIZE;
  for (size_t cur_batch = 0; cur_batch < num_batches; ++cur_batch){
    int * batchInput = local_inputs + cur_batch * BATCH_SIZE * SEQ_LEN;
      /* in [SEQ_LEN] -> out [SEQ_LEN, 4096] */
      cudaMemcpy(gpu_mem_inputs, batchInput,
      BATCH_SIZE * SEQ_LEN * sizeof(int), cudaMemcpyHostToDevice);


      int blockDim = SEQ_LEN; // sequence length
      int gridDim = BATCH_SIZE;
      EmbeddingKernel<<<gridDim, blockDim>>>(gpu_mem_inputs, emb_wg, emb_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);
    //   /* in [SEQ_LEN, 4096] -> out [4096, SEQ_LEN] */
      cudaDeviceSynchronize();

      cudaMemcpy(emb_a->buf, emb_ag,
        permute_a->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);

      Permute(
      PermuteKernel<<<gridDim, blockDim>>>(emb_ag, permute_ag, BATCH_SIZE, SEQ_LEN, HIDDEN_DIM);
      cudaDeviceSynchronize();

      // cudaMemcpy(permute_a->buf, permute_ag,
      //   permute_a->num_elem() * sizeof(float), cudaMemcpyDeviceToHost);
      
    // //   /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - 2] */
    //   Conv1DKernel<<<gridDim, blockDim>>>(permute_ag, conv0_wg, conv0_bg, conv0_ag, BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN, OUTPUT_CHANNEL, 3);
    //   ReLU_Kernel<<<gridDim, blockDim>>>(conv0_ag, conv0_a->num_elem());
    //   cudaDeviceSynchronize();
      
      Conv1D(permute_a, conv0_w, conv0_b, conv0_a);
      ReLU(conv0_a);

      /* in [1024, SEQ_LEN - 2] -> out [1024] */
      GetMax(conv0_a, pool0_a);

      /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - 4] */
      Conv1D(permute_a, conv1_w, conv1_b, conv1_a);
      ReLU(conv1_a);

      /* in [1024, SEQ_LEN - 4] -> out [1024] */
      GetMax(conv1_a, pool1_a);

      /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - 6] */
      Conv1D(permute_a, conv2_w, conv2_b, conv2_a);
      ReLU(conv2_a);

      /* in [1024, SEQ_LEN - 6] -> out [1024] */
      GetMax(conv2_a, pool2_a);

      /* in [4096, SEQ_LEN] -> out [1024, SEQ_LEN - 8] */
      Conv1D(permute_a, conv3_w, conv3_b, conv3_a);
      ReLU(conv3_a);

      /* in [1024, SEQ_LEN - 8] -> out [1024] */
      GetMax(conv3_a, pool3_a);

      /* in [1024] +
            [1024] +
            [1024] +
            [1024] -> out [1024 * 4] */
      Concat(pool0_a, pool1_a, pool2_a, pool3_a, concat_a);

      /* in [1024 * 4] -> out [2048] */
      Linear(concat_a, linear0_w, linear0_b, linear0_a);
      ReLU(linear0_a);

      /* in [2048] -> out [1024] */
      Linear(linear0_a, linear1_w, linear1_b, linear1_a);
      ReLU(linear1_a);

      /* in [1024] -> out [512] */
      Linear(linear1_a, linear2_w, linear2_b, linear2_a);
      ReLU(linear2_a);

      /* in [512] -> out [2] */
      Linear(linear2_a, linear3_w, linear3_b, linear3_a);


    // memcpy(local_outputs + cur_batch * BATCH_SIZE * 2, linear3_a->buf, BATCH_SIZE * 2 * sizeof(float));
    cudaMemcpy(local_outputs + cur_batch * BATCH_SIZE * 2, gpu_mem_outputs,
               BATCH_SIZE * N_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
  
  }

  // // Gather outputs from all nodes
  MPI_Gather(local_outputs, samples_per_node * N_CLASSES, MPI_FLOAT, outputs,
            samples_per_node * N_CLASSES, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // // Free local buffers

  cudaFree(gpu_mem_outputs);
  cudaFree(gpu_mem_inputs);
  cudaFree(emb_a->buf);
  cudaFree(permute_a->buf);
  free(local_inputs);
  free(local_outputs);


}

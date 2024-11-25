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
  delete emb_a;
  delete permute_a;
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

const int NUM_GPUES_PER_NODE = 4;
const int NUM_THREADS_PER_NODE = 4;

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

  // Calculate the number of samples per node
  // samples must be # of power of 2
  size_t samples_per_node = n_samples / mpi_size;
  // size_t start_idx = mpi_rank * samples_per_node;
  // size_t end_idx = (mpi_rank + 1) * samples_per_node;
  // Allocate local buffers for input and output using cudaMallocHost
  int *local_inputs = nullptr;
  float *local_outputs = nullptr;

  // Use cudaMallocHost for pinned memory
  //여기가 아닌것같은데 
  // cudaMallocHost(&local_inputs, samples_per_node * SEQ_LEN * sizeof(int));
  // cudaMallocHost(&local_outputs, samples_per_node * N_CLASSES * sizeof(float));

  // Scatter input data to all nodes
  MPI_Scatter(inputs, samples_per_node * SEQ_LEN, MPI_INT, local_inputs,
              samples_per_node * SEQ_LEN, MPI_INT, 0, MPI_COMM_WORLD);


  // initializing tensor in each node.
  Tensor batch_emb_a, batch_permute_a, batch_conv0_a, batch_pool0_a;
  Tensor batch_conv1_a, batch_pool1_a, batch_conv2_a, batch_pool2_a;
  Tensor batch_conv3_a, batch_pool3_a, batch_concat_a;
  Tensor batch_linear0_a, batch_linear1_a, batch_linear2_a, batch_linear3_a;

  size_t num_batches = samples_per_node / BATCH_SIZE;
  for (size_t cur_batch = 0; cur_batch < num_batches; ++cur_batch){
    int * batchInput = local_inputs + cur_batch * BATCH_SIZE * SEQ_LEN;


    // Perform embedding for the batch
    Embedding(batchInput, emb_w, &batch_emb_a);

    // Permute
    Permute(&batch_emb_a, &batch_permute_a);

    // Apply convolutional and pooling operations
    Conv1D(&batch_permute_a, conv0_w, conv0_b, &batch_conv0_a);
    ReLU(&batch_conv0_a);
    GetMax(&batch_conv0_a, &batch_pool0_a);

    Conv1D(&batch_permute_a, conv1_w, conv1_b, &batch_conv1_a);
    ReLU(&batch_conv1_a);
    GetMax(&batch_conv1_a, &batch_pool1_a);

    Conv1D(&batch_permute_a, conv2_w, conv2_b, &batch_conv2_a);
    ReLU(&batch_conv2_a);
    GetMax(&batch_conv2_a, &batch_pool2_a);

    Conv1D(&batch_permute_a, conv3_w, conv3_b, &batch_conv3_a);
    ReLU(&batch_conv3_a);
    GetMax(&batch_conv3_a, &batch_pool3_a);

    // Concatenate pooled features
    Concat(&batch_pool0_a, &batch_pool1_a, &batch_pool2_a, &batch_pool3_a, &batch_concat_a);

    // Fully connected layers
    Linear(&batch_concat_a, linear0_w, linear0_b, &batch_linear0_a);
    ReLU(&batch_linear0_a);

    Linear(&batch_linear0_a, linear1_w, linear1_b, &batch_linear1_a);
    ReLU(&batch_linear1_a);

    Linear(&batch_linear1_a, linear2_w, linear2_b, &batch_linear2_a);
    ReLU(&batch_linear2_a);

    Linear(&batch_linear2_a, linear3_w, linear3_b, &batch_linear3_a);

    // Copy the computation result to the local outputs
    memcpy(local_outputs + cur_batch * BATCH_SIZE * 2, batch_linear3_a.buf, BATCH_SIZE * 2 * sizeof(float));

    // Free intermediate memory for this batch
    free(batch_emb_a.buf);
    free(batch_permute_a.buf);
    free(batch_pool0_a.buf);
    free(batch_pool1_a.buf);
    free(batch_pool2_a.buf);
    free(batch_pool3_a.buf);
    free(batch_concat_a.buf);
    free(batch_linear0_a.buf);
    free(batch_linear1_a.buf);
    free(batch_linear2_a.buf);
    free(batch_linear3_a.buf);
    
  }

  // Gather outputs from all nodes
  MPI_Gather(local_outputs, samples_per_node * 2, MPI_FLOAT, outputs,
            samples_per_node * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Free local buffers
  // cudaFreeHost(local_inputs);
  // cudaFreeHost(local_outputs);
  }
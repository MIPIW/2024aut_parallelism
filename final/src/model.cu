////변경가능

#include <mpi.h>

#include <cstdio>
#include <tensor.h>
#include "layer.h"
#include "model.h"


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
  emb_a = new Activation({SEQ_LEN, 4096});
  permute_a = new Activation({4096, SEQ_LEN});
  conv0_a = new Activation({1024, SEQ_LEN - 2});
  pool0_a = new Activation({1024});
  conv1_a = new Activation({1024, SEQ_LEN - 4});
  pool1_a = new Activation({1024});
  conv2_a = new Activation({1024, SEQ_LEN - 6});
  pool2_a = new Activation({1024});
  conv3_a = new Activation({1024, SEQ_LEN - 8});
  pool3_a = new Activation({1024});
  concat_a = new Activation({4096});
  linear0_a = new Activation({2048});
  linear1_a = new Activation({1024});
  linear2_a = new Activation({512});
  linear3_a = new Activation({2});
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

const size_t BATCH_SIZE = 16;

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Calculate the number of samples per node
  size_t samples_per_node = (n_samples + mpi_size - 1) / mpi_size; // Round up
  size_t start_idx = mpi_rank * samples_per_node;
  size_t end_idx = fmin(start_idx + samples_per_node, n_samples);

  // Allocate local buffers for input and output using cudaMallocHost
  size_t local_n_samples = end_idx - start_idx;
  int *local_inputs = nullptr;
  float *local_outputs = nullptr;

  // Use cudaMallocHost for pinned memory
  cudaMallocHost(&local_inputs, local_n_samples * SEQ_LEN * sizeof(int));
  cudaMallocHost(&local_outputs, local_n_samples * N_CLASSES * sizeof(float));

  // Scatter input data to all nodes
  MPI_Scatter(inputs, samples_per_node * SEQ_LEN, MPI_INT, local_inputs,
              local_n_samples * SEQ_LEN, MPI_INT, 0, MPI_COMM_WORLD);

  // Compute sentiment for the assigned subset


  // Compute sentiment for the assigned subset in batches
  for (size_t batch_start = 0; batch_start < local_n_samples; batch_start += BATCH_SIZE) {
    size_t batch_end = fmin(batch_start + BATCH_SIZE, local_n_samples);
    size_t current_batch_size = batch_end - batch_start;

    // Allocate memory for batched inputs and outputs
    int *batch_inputs = local_inputs + batch_start * SEQ_LEN;
    Tensor batch_emb_a, batch_permute_a, batch_conv0_a, batch_pool0_a;
    Tensor batch_conv1_a, batch_pool1_a, batch_conv2_a, batch_pool2_a;
    Tensor batch_conv3_a, batch_pool3_a, batch_concat_a;
    Tensor batch_linear0_a, batch_linear1_a, batch_linear2_a, batch_linear3_a;

    // Initialize shapes for intermediate tensors
    batch_emb_a.shape[0] = current_batch_size * SEQ_LEN;
    batch_emb_a.shape[1] = 4096;
    batch_emb_a.buf = (float *)malloc(current_batch_size * SEQ_LEN * 4096 * sizeof(float));

    batch_permute_a.shape[0] = current_batch_size * 4096;
    batch_permute_a.shape[1] = SEQ_LEN;
    batch_permute_a.buf = (float *)malloc(current_batch_size * 4096 * SEQ_LEN * sizeof(float));

    batch_pool0_a.buf = (float *)malloc(current_batch_size * 1024 * sizeof(float));
    batch_pool1_a.buf = (float *)malloc(current_batch_size * 1024 * sizeof(float));
    batch_pool2_a.buf = (float *)malloc(current_batch_size * 1024 * sizeof(float));
    batch_pool3_a.buf = (float *)malloc(current_batch_size * 1024 * sizeof(float));
    batch_concat_a.buf = (float *)malloc(current_batch_size * 1024 * 4 * sizeof(float));
    batch_linear0_a.buf = (float *)malloc(current_batch_size * 2048 * sizeof(float));
    batch_linear1_a.buf = (float *)malloc(current_batch_size * 1024 * sizeof(float));
    batch_linear2_a.buf = (float *)malloc(current_batch_size * 512 * sizeof(float));
    batch_linear3_a.buf = (float *)malloc(current_batch_size * 2 * sizeof(float));

    // Perform embedding for the batch
    Embedding(batch_inputs, emb_w, &batch_emb_a);

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
    memcpy(local_outputs + batch_start * 2, batch_linear3_a.buf, current_batch_size * 2 * sizeof(float));

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
  MPI_Gather(local_outputs, local_n_samples * 2, MPI_FLOAT, outputs,
            samples_per_node * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Free local buffers
  cudaFreeHost(local_inputs);
  cudaFreeHost(local_outputs);
  }
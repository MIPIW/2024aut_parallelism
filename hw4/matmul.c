#define _GNU_SOURCE
#include "matmul.h"
#include "util.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);              \
    exit(EXIT_FAILURE);                                                        \
  }

static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;
static cl_mem a_d, b_d, c_d;

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

    int BLOCK_SIZE = 32;
    int REG_SIZE = 32;
      // A, B 데이터를 GPU로 복사
    err = clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, M * K * sizeof(float), A, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, K * N * sizeof(float), B, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 커널 인자 설정
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &M);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);
    CHECK_ERROR(err);

    // 커널 실행
    
    
    // cl_ulong local_mem_size;
    // clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    // printf("Available local memory: %lu bytes\n", local_mem_size);

    // size_t max_work_group_size;
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    // printf("Maximum workgroup size: %zu\n", max_work_group_size);

    // size_t max_work_item_sizes[3];
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
    // printf("Max work-item sizes: %zu, %zu, %zu\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);


    // Calculate global work size by rounding up to the nearest multiple of the local work size


    // size_t global_work_size[2] = {M, N};  
    // size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE};  

    // size_t local_work_size[2] = {BLOCK_SIZE, BLOCK_SIZE / VEC_WIDTH};  
    // size_t global_work_size[2] = {
    //     ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE,
    //     ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / VEC_WIDTH
    // };

    size_t local_work_size[2] = {BLOCK_SIZE / REG_SIZE, BLOCK_SIZE };  
    size_t global_work_size[2] = {
        ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE / REG_SIZE,
        ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE 
    };
    // printf("%zu, %zu, %zu, %zu\n", global_work_size[0], global_work_size[1], local_work_size[0], local_work_size[1]);

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 결과 읽기
    err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);
    CHECK_ERROR(err);
}

static void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *)malloc(source_size + 1);
  size_t ntotal = 0;
  while (ntotal < source_size) {
    int nread = fread(source_code, sizeof(char), source_size, file);
    ntotal += nread;
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                      NULL, &log_size));
    char *log = (char *)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

void matmul_initialize(int M, int N, int K) {
  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);
  print_platform_info(platform);

  // Get OpenCL device (only 1)
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "sgemm", &err);
  CHECK_ERROR(err);

  // Create GPU buffers
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL,
                       &err);
  CHECK_ERROR(err);
}

void matmul_finalize() {
  // GPU 메모리 객체 해제
    err = clReleaseMemObject(a_d);
    CHECK_ERROR(err);
    err = clReleaseMemObject(b_d);
    CHECK_ERROR(err);
    err = clReleaseMemObject(c_d);
    CHECK_ERROR(err);

    // 커널과 프로그램 객체 해제
    err = clReleaseKernel(kernel);
    CHECK_ERROR(err);
    err = clReleaseProgram(program);
    CHECK_ERROR(err);

    // 명령 큐와 컨텍스트 해제
    err = clReleaseCommandQueue(queue);
    CHECK_ERROR(err);
    err = clReleaseContext(context);
    CHECK_ERROR(err);
}

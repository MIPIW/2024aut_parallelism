#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <immintrin.h>


struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;
  const float *A = (*input).A;
  const float *B = (*input).B;
  float *C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  int num_threads = (*input).num_threads;
  int rank = (*input).rank;

  // timer_start(0);


  // float a[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  //                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  // float b[16] = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
  //                 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  // float result[16];

  // // Load data into AVX-512 registers
  // __m512 vec_a = _mm512_loadu_ps(a);  // Load 16 floats into vec_a
  // __m512 vec_b = _mm512_loadu_ps(b);  // Load 16 floats into vec_b

  // // Perform element-wise multiplication
  // __m512 vec_result = _mm512_mul_ps(vec_a, vec_b);  // vec_result = vec_a * vec_b

  // // Store the result back into the array
  // _mm512_storeu_ps(result, vec_result);


  // printf("%ld\n", sizeof(B));
  // write all content in l1 cache and invalidate 
  // __builtin_ia32_wbinvd();
  // __builtin_ia32_wbnoinvd();

  // benchmark
  // 1. baseline: 12s (row splitting adopted)(cut off)
  // 2. making result_temp(1 -> 2): 15s (cut off)
  // 3. introduce transpose matrix of B(outside of 'matmul_kernel')(1 -> 3): 13s -> cut off
  // 4. change index calculation to 후치증가(3 -> 4): 10s(cut off)
  // 5. use register variable rather than directly allocate to matrix C(4 ->5): 9s(cut off)

  // 6. 2번 수정(matrix A's value reading을 최소화)(1 -> 6): 9.5s
  // 7. use register variable in Assigning value of matrix A(6 -> 7): 9.2s
  // 8. do not calculate indexing variable every time(7->8): 9.0s
  // 9. do not use result_temp(8 -> 9): 9.7s
  // 10. change index variables with register variables(8 -> 10): 8.5s
  // 11. 순서를 바꿔봤는데 안 됨: 10.6s
  // 11.5. 연산 개수 최적화(마지막 스레드가 덜 하더라도 진보적(?)으로 잡기): 안해봄
  
  // 12. ikj to kij(9 -> 12): 8.9s
  // 13. use result_temp(10 + 12 -> 13): 16s
  // 14. cycle 고려하여 instruction 병렬로 세 개 load(12 -> 14): 7.5s BEST YET
  
  // 15. C의 값을 연산 후에 로드하지 않고 연산 전에 로드함(14 -> 15): 8.0s
  // 16. 병렬로 세 개 하던거 두 개로 줄임(15 -> 16): 7.9s
  // 17. cycle 고려하여 로드와 계산 마개조(16 -> 17): 8.0s
  // 18. indexing variables to register variables(14 -> 18): 6.4s

  // 19. 연산 개수 최적화(보수적으로 잡고 마지막 스레드가 조금 더 하기)(12 + 18 -> 19): 8.0s
  // 20. cycle 고려하여 instruction 병렬로 세 개 load(19 -> 20): 6.5s
  // 21. cycle 고려하여 instruction 병렬로 네 개 load(20 -> 21): 6.5s 

  // 22. 하 연산 세 개 끼워넣는 거 못할 것 같아..다시 하나로(20 -> 22): 7.1s
  // 23. kij 이용하면서, matrix A에 대한 접근을 adjecent하게(22 -> 23): ? 좀 더 늦긴 했음. 
  // 24. A도 한 번에 나란히 읽을 수 있게(22 -> 24): 7.7s
  // 25. 자질구레할 거 다 빼기(24 -> 25): 7.0s

  // 26. B에 대한 parallelism(25 -> 26): 23초? 미쳤나 이게
  // 27. cycle 고려하여 instruction 병렬로 세 개 load(25 -> 27): 6.2s
  // 28. index 계산에 매번 새로 곱하기 말고 캐시 활용해서 더하기(27 -> 28): 6.8s

  // 29. 캐시를 고려하여 j를 쪼개보자(28 -> 29): 7.2s
  // 30. j를 쪼갠 거에 A를 transpose해서 연산해보자(29 -> 30): 8.8s
  // 31. A to A_temp 

  // TODO: FILL IN HERE
  // compare available thread and the size of matrix는 validation 코드 보니까 필요 없을듯

  // float result_temp[N];
  // for (int i = 0; i < N; i++){
  //   result_temp[i] = 0.0;
  // }
  // float B_vector_temp[N];
  // for (int i = 0; i < N; i++){
  //   B_vector_temp[i] = 0.0;
  // }

  // float result_temp2[M*N];
  // for (int i = 0; i < M*N; i++){
  //   result_temp2[i] = 0.0;
  // }
  
  register float temp;
  
  // register int c_loc = 0;
  // register int a_loc = 0;
  // register int b_loc = 0;
  
  int rows_per_thread = M / num_threads; // 마지막 thread는 하나 더 해야 함
  int start_rows = rank * rows_per_thread;
  int end_rows = rank == num_threads - 1 ? (M) : (start_rows + rows_per_thread);
  // printf("%d, %d, %d, %d, %d\n", N, cols_per_thread, rank, start_cols, end_cols);
  
  //   for (int k = 0; k < K; ++k) {
    //     for (int j = 0; j < N; ++j) {
    //       // result_temp[j] += A[i * K + k] * B[k * N + j];
    //       // C[i * N + j] += A[i * K + k] * B[k * N + j];
    //       C[i * N + j] += A[i * K + k] * B[k * N + j];
    //     }
    //     // for (int j = 0; j < N; ++j) {
    //     //   C[i * N + j] = result_temp[i];
    //     //   result_temp[i] = 0.0;
    //     // }
    //     // C[i * N + j] = temp;
    //     // temp = 0.0;
    //   }
    // }
    // for(int i = start_rows; i < min_val; ++i){
    //   for (int j = 0; j < N; ++j) {
    //     c_loc = i * N + j;
    //     a_loc = i * K;
    //     b_loc = j * K;
    //     for (int k = 0; k < K; ++k) {
    //       temp += A[a_loc++] * B[b_loc++];
    //     }
    //     C[c_loc] = temp;
    //     temp = 0.0;

    //   }
    // }
  // register float temp_b = 0.0;
  // register float temp_b2 = 0.0;
  // register float temp_b3 = 0.0;
  // register float temp_c = 0.0;
  // register float temp_c2 = 0.0;
  // register float temp_c3 = 0.0;
  // register int index_a = 0;
  // register int index_b = 0;
  // register int index_c = 0;

  // 29
  // register short val2 = 16; // 4의 배수여야 해
  // register short j_to = (N + 1 - val2);
  // register short j_from = (N / val2) * val2;

  //   for (int k = 0; k < K; ++k) {
  //     a_loc = start_rows * K + k;
  //     b_loc = k * N;
  //     c_loc = start_rows * N; // for문 밖으로 빼면 안 되네 왜지 malloc(): corrupted top size
      
  //     for(int j = 0 ; j < j_to ; j+=val2){ // main instruction이 4개씩 4번 도니까 16
  //       index_a = a_loc;
  //       index_c = c_loc;

  //       for(int i = start_rows; i < end_rows; ++i){
  //           index_b = b_loc;
            
  //           temp = A[index_a];

  //           for(int l = 0 ; l < val2; l += val){
  //             // 4개니까 val = 4
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //           }

  //           index_a += K; // 다음줄
  //           index_c += (N - val2); // 다음줄
  //           // index_c += N; ++ 연산 때문에 하지 않아도 됨
  //         }
        
  //       b_loc += val2;
  //       c_loc += val2;  
  //     }

      
  //     // B 행렬 칼럼 중 마지막 남은 거
  //     index_a = a_loc;
  //     index_c = c_loc;
  //     for(int i = start_rows; i < end_rows; ++i){     
  //         index_b = b_loc;
          
  //         temp = A[index_a];

  //         for(int j = j_from ; j < N; ++j){
  //           C[index_c++] += temp * B[index_b++];
  //         }

  //         index_a += K; // 다음줄
  //         index_c += j_from; // 다음줄
          
  //       }

  //   }

  // 30
  // register short val2 = 16; // 4의 배수여야 해
  // register short j_to = (N + 1 - val2);
  // register short j_from = (N / val2) * val2;

  //   for (int k = 0; k < K; ++k) {
  //     a_loc = k * M + start_rows;
  //     b_loc = k * N;
  //     c_loc = start_rows * N; // for문 밖으로 빼면 안 되네 왜지 malloc(): corrupted top size
      
  //     for(int j = 0 ; j < j_to ; j+=val2){ // main instruction이 4개씩 4번 도니까 16
  //       index_a = a_loc;
  //       index_c = c_loc;

  //       for(int i = start_rows; i < end_rows; ++i){
  //           index_b = b_loc;
            
  //           temp = A[index_a++];

  //           for(int l = 0 ; l < val2; l += val){
  //             // 4개니까 val = 4
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //             C[index_c++] += temp * B[index_b++];
  //           }

  //           index_c += (N - val2); // 다음줄
  //           // index_c += N; ++ 연산 때문에 하지 않아도 됨
  //         }
        
  //       b_loc += val2;
  //       c_loc += val2;  
  //     }

      
  //     // B 행렬 칼럼 중 마지막 남은 거
  //     index_a = a_loc;
  //     index_c = c_loc;
  //     for(int i = start_rows; i < end_rows; ++i){     
  //         index_b = b_loc;
          
  //         temp = A[index_a++];

  //         for(int j = j_from ; j < N; ++j){
  //           C[index_c++] += temp * B[index_b++];
  //         }

  //         index_c += j_from; // 다음줄
          
  //       }

  //   }
  

  // 31 
  // register char val2 = 32; // 4의 배수여야 해
  // register short j_to = (N + 1 - val2);
  // register short j_from = (N / val2) * val2;

  // register char val3 = 16;
  // register short i_to;
  // register short i_from;

    // for (int k = 0; k < K; ++k) {
    //   a_loc = k * M + start_rows;
    //   b_loc = k * N;
    //   c_loc = start_rows * N; // for문 밖으로 빼면 안 되네 왜지 malloc(): corrupted top size

            
    //   for(int j = 0 ; j < j_to ; j+=val2){ // main instruction이 4개씩 4번 도니까 16
    //     index_a = a_loc;
    //     index_c = c_loc;

    //     for(int i = start_rows; i < end_rows; ++i){
    //         index_b = b_loc;
            
    //         temp = A[index_a++];

    //         for(int l = 0 ; l < val2; l += val){
    //           // 4개니까 val = 4
    //           C[index_c++] += temp * B[index_b++];
    //           // __builtin_ia32_clflush(&B[index_b++]);
    //           C[index_c++] += temp * B[index_b++];
    //           C[index_c++] += temp * B[index_b++];
    //           C[index_c++] += temp * B[index_b++];
    //         }
            
    //         index_c += (N - val2); // 다음줄
    //         // index_c += N; ++ 연산 때문에 하지 않아도 됨
    //       }
        
    //     b_loc += val2;
    //     c_loc += val2;  


    //   }

      
    //   // B 행렬 칼럼 중 마지막 남은 거
    //   index_a = a_loc;
    //   index_c = c_loc;
    //   for(int i = start_rows; i < end_rows; ++i){     
    //       index_b = b_loc;
          
    //       temp = A[index_a++];

    //       for(int j = j_from ; j < N; ++j){
    //         C[index_c++] += temp * B[index_b++];
    //       }

    //       index_c += j_from; // 다음줄
          
    //     }

    // }

    // printf("\n%ld------\n", sizeof(C) / sizeof(float));

    // a_loc, b_loc, c_loc, index_a, index_b, index_c, c_loc_default, j_to, j_from
    
    // int * a_loc = &end_rows + 1;
    // // a_loc = a_loc + rank * 28;
    // int * b_loc = a_loc + 1; // = a_loc + 8;
    // int * c_loc = a_loc + 2;
    // int * index_a = a_loc + 3;
    // int * index_b = a_loc + 4;
    // int * index_c = a_loc + 5;
    // int * j_to = a_loc + 6;
    // int * j_from = a_loc + 7;

    // (*a_loc) = 0;
    // (*b_loc) = 0;

    

    // *c_loc = start_rows * N;

    // a_loc, b_loc, c_loc, index_a, index_b, index_c, c_loc_default, j_to, j_from

    // // 32 use set of variables 
    // int variables[8] = {0,0,0,0,0,0,0,0};
    // printf("---%p, %p, %p, %p---\n", B, C,A, variables);
    // // best
    // // register char j_to = (N + 1 - val);
    // // register char j_from = (N / val) * val;
    // variables[6] = (N + 1 - val); 
    // variables[7] = (N / val) * val; // j_from
    // // *j_to = (N + 1 - val);
    // // *j_from = (N / val) * val;

    // // printf("%d, %d, %d", N, *j_to, *j_from);

    // variables[0] = start_rows;  // a_loc
    // variables[1] = 0; // b_loc
    // variables[2] = start_rows * N; // c_loc_default
    // // *a_loc = start_rows;
    // // *b_loc = 0;
    // // *c_loc = start_rows * N;

    // for (int k = 0; k < K; ++k) {
    //   // *index_a = *a_loc;
    //   // *index_c = *c_loc;
    //   variables[3] = variables[0]; // index_a
    //   variables[5] = variables[2]; // index_c
    //   // *index_a = *a_loc;
    //   // *index_c = *c_loc;

        
    //     for(int i = start_rows; i < end_rows; ++i){      
    //         variables[4] = variables[1];
    //         // *index_b = *b_loc;
    //         // index_b = b_loc;
    //         // temp = A[a_loc++];
    //         temp = A[variables[3]++];
    //         // temp = A[(*index_a)++];

    //         // a_loc += K;    

    //         // for(int j = 0 ; j < j_to; j += val){
    //         for(int j = 0 ; j < variables[6]; j += val){
              
    //         // for(int j = 0 ; j < *j_to; j += val){

    //           C[variables[5]++] += temp * B[variables[4]++];
    //           C[variables[5]++] += temp * B[variables[4]++];
    //           C[variables[5]++] += temp * B[variables[4]++];
    //           C[variables[5]++] += temp * B[variables[4]++];
    //           // C[c_loc++] += temp * B[index_b++];
    //           // C[c_loc++] += temp * B[index_b++];
    //           // C[c_loc++] += temp * B[index_b++];
    //           // C[c_loc++] += temp * B[index_b++];
    //           // C[(*index_c)++] += temp * B[(*index_b)++];
    //           // C[(*index_c)++] += temp * B[(*index_b)++];
    //           // C[(*index_c)++] += temp * B[(*index_b)++];
    //           // C[(*index_c)++] += temp * B[(*index_b)++];
              
    //         }

    //         // for(int j = j_from ; j < N ; j++){
    //         for(int j = variables[7] ; j < N ; j++){
    //         // for(int j = *j_from ; j < N ; j++){
    //           // printf("--%d, %d--", *j_from, N);
            
    //           C[variables[5]++] += temp * B[variables[4]++];
    //           // C[(*index_c)++] += temp * B[(*index_b)++];
              
    //           // printf("%d", 4);

    //         }
    //       }       
      
    //   variables[0] += M;
    //   variables[1] += N;
    //   // *a_loc += M;
    //   // *c_loc += N;      
    //   // c_loc += N;
    // }

    // 33 마개조

  //   register char val = 8; // 한 번에 수행하는 연산의 개수(operation line number)
  //   const float * index_a, * index_b;
  //   float * index_c;

  //   float temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10, temp11, temp12 = 0.0;
    
  //   int variables[15] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  //   variables[11] = 8; // BLOCK_K
  //   variables[12] = K / variables[11]; // lines_per_block
  //   printf("lines per block %d\n", variables[12]);
  //   // variables[13] = 8; // lines per block
  //   // variables[14] = 0; // 

  // for (int block_k = 0; block_k < variables[11]; block_k++){
  //   // register char j_to = (N + 1 - val);
  //   // register char j_from = (N / val) * val;
  //   // variables[6] = (K + 1 - val); 
  //   // variables[7] = (K / val) * val; // j_from
    
  //   variables[9] = block_k * variables[12]; // starting lines     
  //   variables[8] =  (block_k == (variables[11] - 1)) ? K : block_k * variables[12] + variables[12]; // end lines

  //   variables[10] = (variables[8] - variables[9]); // num_processing blocks
  //   variables[6] = variables[8] + 1 - val; // j_to per block
  //   variables[7] = variables[9] + (variables[10] / val) * val; // j_from per block

  //   // variables[9] = block_k * lines_per_block; // starting lines     
  //   // variables[8] = block_k * lines_per_block + lines_per_block; // end lines
  
  //   // variables[10] = lines_per_block; // num_processing blocks
  //   // variables[6] = variables[8] + 1 - val; // j_to per block
  //   // variables[7] = variables[9] + variables[10]; // j_from per block


  //   variables[0] = start_rows * K + variables[9];  // a_loc
  //   variables[1] = variables[9]; // b_loc
  //   variables[2] = start_rows * N; // c_loc_default


  
  //   for (int j = 0; j < N; ++j) {
  //     variables[3] = variables[0]; // index_a
  //     variables[5] = variables[2]; // index_c
      
  //     // index_a = &A[variables[0]]; // index_a
  //     // index_c = &C[variables[2]]; // index_c

  //       for(int i = start_rows; i < end_rows; ++i){  
  //           A[variables[3]]; // 미리 캐싱 -> deperform
  //           C[variables[5]]; // 미리 캐싱 -> deperform

  //           variables[4] = variables[1]; // index_b
  //           // index_b = &B[variables[1]]; // index_b

            
  //           for(int k = variables[9] ; k < variables[6]; k += val){
  //             C[variables[5]];
  //             // temp1 += *(index_a++) * *(index_b++);
  //             // temp2 += *(index_a++) * *(index_b++);
  //             // temp3 += *(index_a++) * *(index_b++);
  //             // temp4 += *(index_a++) * *(index_b++);
              
  //             // temp7 += *(index_a++) * *(index_b++);
  //             // temp8 += *(index_a++) * *(index_b++);
  //             // temp9 += *(index_a++) * *(index_b++);
  //             // temp10 += *(index_a++) * *(index_b++);
  //             temp1 += A[variables[3]++] * B[variables[4]++]; // 시간 제일 많이 걸려 어떻게 해결하지
  //             temp2 += A[variables[3]++] * B[variables[4]++];
  //             temp3 += A[variables[3]++] * B[variables[4]++];
  //             temp4 += A[variables[3]++] * B[variables[4]++];

  //             temp7 += A[variables[3]++] * B[variables[4]++];
  //             temp8 += A[variables[3]++] * B[variables[4]++];
  //             temp9 += A[variables[3]++] * B[variables[4]++];
  //             temp10 += A[variables[3]++] * B[variables[4]++];
                
  //           }
            
  //           temp5 = temp1 + temp2;
  //           temp6 = temp3 + temp4;
  //           temp11 = temp7 + temp8;
  //           temp12 = temp9 + temp10;
  //           C[variables[5]] += temp5 + temp6 + temp11 + temp12;

  //           // for(int j = j_from ; j < N ; j++){
  //           for(int k = variables[7] ; k < variables[8] ; k++){
  //           // for(int j = *j_from ; j < N ; j++){
  //             // printf("--%d, %d--", *j_from, N);
  //             C[variables[5]] += A[variables[3]++] * B[variables[4]++];
  //             // *index_c += *(index_a++) * *(index_b++);

  //             // A[variables[3]++] * B[variables[4]++];
  //             // C[(*index_c)++] += temp * B[(*index_b)++];
  //             // printf("%d", 4);
  //           }
            
  //           // *index_c += temp5 + temp6 + temp11 + temp12;

            

  //           variables[5] += N;
  //           // index_c += N;
  //           variables[3] += K - variables[10];
  //           // index_a += (K - variables[10]);

  //           temp1 = temp2 = temp3 = temp4 = temp5 = temp6 = temp7 = temp8 = temp9 = temp10 = temp11 = temp12 = 0.0;
  //         } 

           

  //     variables[1] += K;
  //     variables[2] ++;
  //     // *a_loc += M;
  //     // *c_loc += N;      
  //     // c_loc += N;
  //   }
  // }

 

    // block 개수가 아닌 batch 수로 내 이전 코드를 똑같이 구현해 보기
    // char val = 8; // 한 번에 수행하는 연산의 개수(operation line number)
    // char batches = 32; // 블록 당 배치 수(8 이상의 2의 제곱수)
    // char num_blocks = K / batches; // 블록 수. K가 2의 제곱수가 아니라면 잉여 블록이 있음. 
    // for(int block = 0; block < num_blocks; block++){
    //   for(int j = 0; j < N; j++){
    //     for(int i = start_rows; i  < end_rows; i++){
    //       for(int k = );
    //     }
    //   }
      
    // }

  //   char val = 8; // 한 번에 수행하는 연산의 개수(operation line number)

  //   const float * index_a, * index_b;
  //   float * index_c;

  //   float temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10, temp11, temp12 = 0.0;
    
  //   int variables[15] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  //   variables[11] = 8; // BLOCK_K
  //   variables[12] = K / variables[11]; // lines_per_block
  //   printf("lines per block %d\n", variables[12]);
  //   // variables[13] = 8; // lines per block
  //   // variables[14] = 0; // 

  // for (int block_k = 0; block_k < variables[11]; block_k++){
  //   // register char j_to = (N + 1 - val);
  //   // register char j_from = (N / val) * val;
  //   // variables[6] = (K + 1 - val); 
  //   // variables[7] = (K / val) * val; // j_from
    
  //   variables[9] = block_k * variables[12]; // starting lines     
  //   variables[8] =  (block_k == (variables[11] - 1)) ? K : block_k * variables[12] + variables[12]; // end lines

  //   variables[10] = (variables[8] - variables[9]); // num_processing blocks
  //   variables[6] = variables[8] + 1 - val; // j_to per block
  //   variables[7] = variables[9] + (variables[10] / val) * val; // j_from per block

  //   // variables[9] = block_k * lines_per_block; // starting lines     
  //   // variables[8] = block_k * lines_per_block + lines_per_block; // end lines
  
  //   // variables[10] = lines_per_block; // num_processing blocks
  //   // variables[6] = variables[8] + 1 - val; // j_to per block
  //   // variables[7] = variables[9] + variables[10]; // j_from per block


  //   variables[0] = start_rows * K + variables[9];  // a_loc
  //   variables[1] = variables[9]; // b_loc
  //   variables[2] = start_rows * N; // c_loc_default


  
  //   for (int j = 0; j < N; ++j) {
  //     variables[3] = variables[0]; // index_a
  //     variables[5] = variables[2]; // index_c
      
  //     // index_a = &A[variables[0]]; // index_a
  //     // index_c = &C[variables[2]]; // index_c

  //       for(int i = start_rows; i < end_rows; ++i){  
  //           A[variables[3]]; // 미리 캐싱 -> deperform
  //           C[variables[5]]; // 미리 캐싱 -> deperform

  //           variables[4] = variables[1]; // index_b
  //           // index_b = &B[variables[1]]; // index_b

            
  //           for(int k = variables[9] ; k < variables[6]; k += val){
  //             C[variables[5]];
  //             // temp1 += *(index_a++) * *(index_b++);
  //             // temp2 += *(index_a++) * *(index_b++);
  //             // temp3 += *(index_a++) * *(index_b++);
  //             // temp4 += *(index_a++) * *(index_b++);
              
  //             // temp7 += *(index_a++) * *(index_b++);
  //             // temp8 += *(index_a++) * *(index_b++);
  //             // temp9 += *(index_a++) * *(index_b++);
  //             // temp10 += *(index_a++) * *(index_b++);
  //             temp1 += A[variables[3]++] * B[variables[4]++]; // 시간 제일 많이 걸려 어떻게 해결하지
  //             temp2 += A[variables[3]++] * B[variables[4]++];
  //             temp3 += A[variables[3]++] * B[variables[4]++];
  //             temp4 += A[variables[3]++] * B[variables[4]++];

  //             temp7 += A[variables[3]++] * B[variables[4]++];
  //             temp8 += A[variables[3]++] * B[variables[4]++];
  //             temp9 += A[variables[3]++] * B[variables[4]++];
  //             temp10 += A[variables[3]++] * B[variables[4]++];
                
  //           }
            
  //           temp5 = temp1 + temp2;
  //           temp6 = temp3 + temp4;
  //           temp11 = temp7 + temp8;
  //           temp12 = temp9 + temp10;
  //           C[variables[5]] += temp5 + temp6 + temp11 + temp12;

  //           // for(int j = j_from ; j < N ; j++){
  //           for(int k = variables[7] ; k < variables[8] ; k++){
  //           // for(int j = *j_from ; j < N ; j++){
  //             // printf("--%d, %d--", *j_from, N);
  //             C[variables[5]] += A[variables[3]++] * B[variables[4]++];
  //             // *index_c += *(index_a++) * *(index_b++);

  //             // A[variables[3]++] * B[variables[4]++];
  //             // C[(*index_c)++] += temp * B[(*index_b)++];
  //             // printf("%d", 4);
  //           }
            
  //           // *index_c += temp5 + temp6 + temp11 + temp12;

            

  //           variables[5] += N;
  //           // index_c += N;
  //           variables[3] += K - variables[10];
  //           // index_a += (K - variables[10]);

  //           temp1 = temp2 = temp3 = temp4 = temp5 = temp6 = temp7 = temp8 = temp9 = temp10 = temp11 = temp12 = 0.0;
  //         } 

           

  //     variables[1] += K;
  //     variables[2] ++;
  //     // *a_loc += M;
  //     // *c_loc += N;      
  //     // c_loc += N;
  //   }
  // }


    // char val = 8; // 한 번에 수행하는 연산의 개수(operation line number)

    // // const float * index_a, * index_b;
    // // float * index_c;
    // int index_a, index_b, index_c = 0;
    // const float * point_a, * point_b; 
    // float * point_c;
    // float temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10, temp11, temp12 = 0.0;

    // //우석이형 말대로 하려면 그냥 반대로 뒤집으면 됨
    // int num_block = 2;
    // int num_batches = K / num_block; // for run_performance, it's 2^14 / num_blocks

    // for (int block = 0; block < num_block; ++block){
    //   for(int j = 0 ; j < N ; ++j){
    //     for(int i = start_rows; i < end_rows; ++i){

    //         index_a = i * K + num_batches * block;
    //         index_b = j * K + num_batches * block;
    //         index_c = i * N + j;

    //         // point_a = &A[index_a];
    //         // point_b = &B[index_b];
    //         // point_c = &C[index_c];

    //         for(int k = 0; k < (num_batches + 1 - val) ; k+=val ){ // cols가 8보다 작으면 어떡해. 일단 킵고잉
    //           // temp1 += *point_a       * *point_b;
    //           // temp2 += *(point_a + 1) * *(point_b + 1);
    //           // temp3 += *(point_a + 2) * *(point_b + 2);
    //           // temp4 += *(point_a + 3) * *(point_b + 3);
              
    //           // temp5 += *(point_a + 4) * *(point_b + 4);
    //           // temp6 += *(point_a + 5) * *(point_b + 5);
    //           // temp7 += *(point_a + 6) * *(point_b + 6);
    //           // temp8 += *(point_a + 7) * *(point_b + 7);

    //           // point_a += 8;
    //           // point_b += 8;
    //           // temp1 += A[index_a++] * B[index_b++];
    //           // temp2 += A[index_a++] * B[index_b++];
    //           // temp3 += A[index_a++] * B[index_b++];
    //           // temp4 += A[index_a++] * B[index_b++];
              
    //           // temp5 += A[index_a++] * B[index_b++];
    //           // temp6 += A[index_a++] * B[index_b++];
    //           // temp7 += A[index_a++] * B[index_b++];
    //           // temp8 += A[index_a++] * B[index_b++];

    //           temp1 += A[index_a] * B[index_b];
    //           temp2 += A[index_a + 1] * B[index_b + 1];
    //           temp3 += A[index_a + 2] * B[index_b + 2];
    //           temp4 += A[index_a + 3] * B[index_b + 3];
              
    //           temp5 += A[index_a + 4] * B[index_b + 4];
    //           temp6 += A[index_a + 5] * B[index_b + 5];
    //           temp7 += A[index_a + 6] * B[index_b + 6];
    //           temp8 += A[index_a + 7] * B[index_b + 7];

    //           index_a += 8;
    //           index_b += 8;
    //         }
            
    //         temp9 = temp1 + temp2;
    //         temp10 = temp3 + temp4;
    //         temp11 = temp5 + temp6;
    //         temp12 = temp7 + temp8;
    //         // C[index_c] += temp9 + temp10 + temp11 + temp12;
    //         C[index_c] += temp9 + temp10 + temp11 + temp12;

 
    //         temp1 = temp2 = temp3 = temp4 = temp5 = temp6 = temp7 = temp8 = temp9 = temp10 = temp11 = temp12 = 0.0;
    //       }
    //   }
    // }    

    // // block 남은 거 계산
    // for(int j = 0 ; j < N ; ++j){
    //   for(int i = start_rows; i < end_rows; ++i){

    //     index_a = i * K + num_batches * num_block;
    //     index_b = j * K + num_batches * num_block;
    //     index_c = i * N + j;

    //     point_a = &A[index_a];
    //     point_b = &B[index_b];
    //     point_c = &C[index_c];
        
    //     for(int k = num_batches * num_block; k < K ; ++k){
    //       C[index_c] += A[index_a++] * B[index_b++];
    //       // C[index_c] += *(point_a++) * *(point_b++);
    //     }
        
    //   }
    // } 
  
   char val = 8; // 한 번에 수행하는 연산의 개수(operation line number)

    // const float * index_a, * index_b;
    // float * index_c;
    int index_a, index_b, index_c = 0;

    //우석이형 말대로 하려면 그냥 반대로 뒤집으면 됨
    int num_block = 2;
    int num_batches = K / num_block; // for run_performance, it's 2^14 / num_blocks
    for(int i = start_rows; i < end_rows; ++i){
      for (int block = 0; block < num_block; ++block){
          
          index_a = i * K + num_batches * block;
          index_b = N * (num_batches * block);

          for(int k = 0 ; k < num_batches ; ++k){
            temp = A[index_a++];

            index_c = i * N;
            for(int j = 0 ; j < (N + 1 - val); j += val ){ 

              C[index_c] += temp * B[index_b];
              C[index_c + 1] += temp * B[index_b + 1];
              C[index_c + 2] += temp * B[index_b + 2];
              C[index_c + 3] += temp * B[index_b + 3];
              
              C[index_c + 4] += temp * B[index_b + 4];
              C[index_c + 5] += temp * B[index_b + 5];
              C[index_c + 6] += temp * B[index_b + 6];
              C[index_c + 7] += temp * B[index_b + 7];

              index_b += 8;
              index_c += 8;

            }

            for(int j = (N / val) * val ; j < N; ++j ){ 
              // index_c = i * N + j;
              C[index_c++] += temp * B[index_b++];

            }
            // temp9 = temp1 + temp2;
            // temp10 = temp3 + temp4;
            // temp11 = temp5 + temp6;
            // temp12 = temp7 + temp8;
            // // C[index_c] += temp9 + temp10 + temp11 + temp12;
            // C[index_c] += temp9 + temp10 + temp11 + temp12;
          }
      }
    }    

    // block 남은 거 계산    
    for(int i = start_rows; i < end_rows; ++i){  
      for(int k = 0 ; k < K - num_batches * num_block ; ++k){
        index_a = i * K + num_batches * num_block + k;  
        temp = A[index_a];

        index_b = N * (num_batches * num_block + k);
        
            for(int j = 0 ; j < (N + 1 - val); j += val ){ // cols가 8보다 작으면 어떡해. 일단 킵고잉
              index_c = i * N + j;

              C[index_c] += temp * B[index_b];
              C[index_c + 1] += temp * B[index_b + 1];
              C[index_c + 2] += temp * B[index_b + 2];
              C[index_c + 3] += temp * B[index_b + 3];
              
              C[index_c + 4] += temp * B[index_b + 4];
              C[index_c + 5] += temp * B[index_b + 5];
              C[index_c + 6] += temp * B[index_b + 6];
              C[index_c + 7] += temp * B[index_b + 7];

              index_b += 8;
              index_c += 8;

            }

            for(int j = (N / val) * val ; j < N; ++j){ // cols가 8보다 작으면 어떡해. 일단 킵고잉
              index_c = i * N + j;
              C[index_c++] += temp * B[index_b++];

            }
      }
    }
    
   
        
          // C[index_c++] += temp * B[index_b++];
          // C[index_c] +=  temp * B[index_b];


          // temp_b = temp * B[index_b++];
          // temp_c = C[c_loc + j]; 

          // temp_b2 = temp * B[index_b];
          // C[c_loc + j] = temp_c + temp_b;
          // temp_c2 = C[c_loc + j + 1]; 
          // C[c_loc + j+1] = temp_c2 + temp_b2;
          
          // temp_b3 = temp * B[b_loc + j + 2];
          // temp_c3 = C[c_loc + j + 2]; 
          // C[c_loc + j+2] = temp_c3 + temp_b3;
        

        // for (int j = 0; j < N; ++j) {
        //     C[c_loc + j] += result_temp[j];
        //     result_temp[j] = 0.0;
        // }
      
    

    // // 23
    // int interval_a = end_rows - start_rows;

    // for (int k = 0; k < K; ++k) {
    //   b_loc = k * N;

    //   for(int i = start_rows; i < end_rows; ++i){
        
    //     a_loc = k * M + start_rows;
    //     c_loc = i * N;

    //     temp = A[a_loc++];

    //     // # range out of index error 고쳐야 함
    //     index_b = b_loc;
                  
    //     for (int j = 0; j < N; ++j) {
          
    //       C[c_loc++] += temp * B[index_b++];
    //     }
    //   }
    // }

    // 25
    // int interval_a = end_rows - start_rows;

    // for (int k = 0; k < K; ++k) {
    //   b_loc = k * N;

    //   for(int i = 0; i < M; ++i){
        
    //     a_loc = i * K;
    //     c_loc = i * N;

    //     temp = A[a_loc + k];

    //     // # range out of index error 고쳐야 함
                  
    //     for (int j = start_cols; j < end_cols; ++j) {
    //       index_b = b_loc + j;
    //       index_c = c_loc + j;

    //       C[index_c] += temp * B[index_b];
    //     }
    //   }
    // }

    // // 11
    // for (int k = 0; k < K; ++k){
    //   b_loc = k * N;
        
    //   for(int j = 0; j < N; j++){
    //     B_vector_temp[j] = B[b_loc++];
    //   }

    //   for(int i = start_rows; i < min_val; ++i){
    //     a_loc = i * K + k;
    //     temp = A[a_loc];

    //     for(int j = 0; j < N; j++){
    //       C[i*N + j] += temp * B_vector_temp[j];
    //     }


    //   }  
    // }
   
  
    // double elapsed_time = timer_stop(0);
    // printf("time in convert matrix: %f sec\n", elapsed_time);

  // printf("Done of rank %d thread, job with %d \t %d rows in %f sec\n", rank, start_rows, end_rows, elapsed_time);
  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }


  // printf("%s\t", "converting");



  // float *A_temp = (float *)malloc(M * K * sizeof(float));

  // for (int i = 0; i < M; i++){
  //   for (int j = 0; j < K; j++){
  //     A_temp[j * M + i] = A[i * K + j];
  //   }
  // }

  // printf("%s\n", "running");







  // float *B_temp = (float *)malloc(K * N * sizeof(float));

  // for (int i = 0; i < K; i++){
  //   for (int j = 0; j < N; j++){
  //     B_temp[j * K + i] = B[i * N + j];
  //   }
  // }



  

  // #define max(x, y) (x) > (y) ? (x) : (y)

  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    // args[t].A = A_temp, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    // args[t].A = A, args[t].B = B_temp, args[t].C = C, args[t].M = M, args[t].N = N,

    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
    if (err) {
      printf("pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }



  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      printf("pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  

  // free(B_temp);
}

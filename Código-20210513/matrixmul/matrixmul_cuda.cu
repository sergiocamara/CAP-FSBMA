#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
/*CUDA square matirx multiplication */
#define MIN(I, J) (I) < (J) ? (I) : (J)

uint64_t rtdsc() {
  uint32_t high, low;

  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  return ((uint64_t)(high) << 32 | (low));
}

__global__ void multiply(int N, int *CC, int *AA, int *BB) {

#define TILE_WIDTH 32
#define A(I, J) AA[I * N + J]
#define B(I, J) BB[I * N + J]
#define C(I, J) CC[I * N + J]

  __shared__ int A_SharedMem[TILE_WIDTH][TILE_WIDTH];
  __shared__ int B_SharedMem[TILE_WIDTH][TILE_WIDTH];

  int accumulator = 0;
  int i, j;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row <N && col < N){
    for (int k = 0; k < N; ++k)
	     C(row,col)+=A(row,k)*B(k,col);

  }
}

void initialize(int N, int *AA, int *BB) {
  int i, j;

#define A(I, J) AA[I * N + J]
#define B(I, J) BB[I * N + J]

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      A(i, j) = random() % 4;
      B(i, j) = random() % 4;
    }
  }
}

int check(int N, int *CC, int *AA, int *BB) {
  int i, j, k;
  int sum;

#define A(I, J) AA[I * N + J]
#define B(I, J) BB[I * N + J]
#define C(I, J) CC[I * N + J]

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      sum = 0;

      for (k = 0; k < N; ++k)
        sum = sum + A(i, k) * B(k, j);

      if (sum != C(i, j)){
	      printf("%d %d sum=%d C(i,j)=%d C(25,25)=%d \n",i,j,sum, C(i,j),C(25,25));
        return false;
      }

      }
    }

  return true;
}

int main(int argc, char *argv[]) {
  int N;
  int *A;
  int *B;
  int *C;

  int *A_DevPtr;
  int *B_DevPtr;
  int *C_DevPtr;

  uint64_t start, end;
  const char *result_str;

  srand(getpid());

  if (argc < 2) {
    fprintf(stderr, "USAGE: %s <N>", argv[0]);
    exit(1);
  }

  N = atoi(argv[1]);
  A = (int *)malloc(sizeof(int) * N * N);
  B = (int *)malloc(sizeof(int) * N * N);
  C = (int *)malloc(sizeof(int) * N * N);

if(  cudaMalloc(&A_DevPtr, sizeof(int) * N * N)!=cudaSuccess) {
	printf("Alloc Error A\n");return -1;
}
  if( cudaMalloc(&B_DevPtr, sizeof(int) * N * N)!=cudaSuccess){
	printf("Alloc Error B\n");return -1;
  cudaFree(A_DevPtr);
	return -1;
  }
  if(cudaMalloc(&C_DevPtr, sizeof(int) * N * N)!=cudaSuccess){
	printf("Alloc Error C\n");
  cudaFree(B_DevPtr);
  cudaFree(A_DevPtr);
	return -1;
  }
printf("alloc done");
  initialize(N, A, B);

  if( cudaMemcpy(A_DevPtr, A, sizeof(int) * N * N, cudaMemcpyHostToDevice)!=cudaSuccess){
	  printf("Failed copy of A");
  cudaFree(C_DevPtr);
  cudaFree(B_DevPtr);
  cudaFree(A_DevPtr);
	return -1;
  }
  if(cudaMemcpy(B_DevPtr, B, sizeof(int) * N * N, cudaMemcpyHostToDevice)!=cudaSuccess){
	  printf("Failed copy of B");
  cudaFree(C_DevPtr);
  cudaFree(B_DevPtr);
  cudaFree(A_DevPtr);
	return -1;
  }
  if(cudaMemset(C_DevPtr, 0, sizeof(int) * N * N)!=cudaSuccess){
	  printf("Failed memset C");
  cudaFree(C_DevPtr);
  cudaFree(B_DevPtr);
  cudaFree(A_DevPtr);
	return -1;
  }

  int NumberOfBlocksAxis = (N - 1) / 32 + 1;

  dim3 BlocksN(NumberOfBlocksAxis, NumberOfBlocksAxis, 1);
  dim3 ThreadsN(32, 32, 1);

  start = rtdsc();
  multiply<<<BlocksN, ThreadsN>>>(N, C_DevPtr, A_DevPtr, B_DevPtr);
cudaDeviceSynchronize();
  end = rtdsc();
   cudaError_t err = cudaGetLastError();
   if(err!=cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       

   }
  if(cudaMemcpy(C, C_DevPtr, sizeof(int) * N * N, cudaMemcpyDeviceToHost)!=cudaSuccess){
	  printf("copy from GPU  error");
  }

  result_str = check(N, C, A, B) ? "Correct" : "Wrong";
  printf("Execution time: %ld cycles\n%s", end - start, result_str);

  cudaFree(C_DevPtr);
  cudaFree(B_DevPtr);
  cudaFree(A_DevPtr);
  free(C);
  free(B);
  free(A);

  return 0;
}

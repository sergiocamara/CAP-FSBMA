#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/auxv.h>

#define CACHE_LINE_SIZE 64
#define LINE_LENGTH     (CACHE_LINE_SIZE / sizeof(int))
#define MIN(I,J)       (I) < (J) ? (I) : (J)

/*Parallel Square Matrix Multiplication */
uint64_t rtdsc() {
  uint32_t high, low;

  asm volatile ("rdtsc"
      : "=a" (low), "=d" (high) 
  );
  return ((uint64_t)(high) << 32 | (low));
}

void multiply(int N, int (*C)[N], int (*A)[N], int (*B)[N]) {
  int i, j, k;
  int ii, jj, kk;

  int i_bound;
  int j_bound;
  int k_bound;

//outer most loop parallel loop order (i,k,j)
//tiled execution
#pragma omp parallel for private(j, k, ii, jj, kk, i_bound, j_bound, k_bound)
  for (i = 0; i < (N / LINE_LENGTH + 1); ++i) {
    for (k = 0; k < (N / LINE_LENGTH + 1); ++k) {
      for (j = 0; j < (N / LINE_LENGTH + 1); ++j) {
        i_bound = MIN(N, (i + 1) * LINE_LENGTH);
        k_bound = MIN(N, (k + 1) * LINE_LENGTH);
        j_bound = MIN(N, (j + 1) * LINE_LENGTH);
      
        for (ii = i * LINE_LENGTH; ii < i_bound; ++ii) {
          for (kk = k * LINE_LENGTH; kk < k_bound; ++kk) {
            for (jj = j * LINE_LENGTH; jj < j_bound; ++jj)
              C[ii][jj] = C[ii][jj] + A[ii][kk] * B[kk][jj];
          }
        }
      }
    }
  }
}

_Bool check(int N, int (*C)[N], int (*A)[N], int (*B)[N]) {
  int i, j, k;
  int sum;

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      sum = 0;

      for (k = 0; k < N; ++k)
        sum = sum + A[i][k] * B[k][j];

      if (sum != C[i][j])
        return false;
    }
  }

  return true;
}

void initialize(int N, int (*A)[N], int (*B)[N]) {
  int i, j;

#pragma omp parallel for private(j)
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      A[i][j] = random() % 1024;

#pragma omp parallel for private(j) 
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j) 
      B[i][j] = random() % 1024;
}

int main(int argc, char *argv[]) {
  int  N;
  int *A;
  int *B;
  int *C;

  uint64_t start, end;
  const char *result_str;

  srand(getpid());

  if (argc < 2) {
    fprintf(stderr, "USAGE: %s <N>", argv[0]);
    exit(1);
  }

  N = atoi(argv[1]);
  A = malloc(sizeof(int) * N * N);
  B = malloc(sizeof(int) * N * N);
  C = calloc(N * N, sizeof(int));

  initialize(N, (int (*)[N])(A), (int (*)[N])(B));

  start = rtdsc();
  multiply(N, (int (*)[N])(C), (int (*)[N])(A), (int (*)[N])(B));
  end   = rtdsc();

  result_str = check(N, (int (*)[N])(C), (int (*)[N])(A), (int (*)[N])(B)) ? "Correct" : "Wrong";
  printf("Execution time: %ld cycles\n%s", end - start, result_str);

  free(C);
  free(B);
  free(A);

  return 0;
}

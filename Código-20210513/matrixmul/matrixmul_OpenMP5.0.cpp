#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstdbool>
#include <cstdint>

uint64_t rtdsc() {
  uint32_t high, low;

  asm volatile ("rdtsc"
      : "=a" (low), "=d" (high) 
  );
  return ((uint64_t)(high) << 32 | (low));
}

void multiply(int N, int *CC, int *AA, int *BB, uint32_t NumBlocks, uint32_t NumThreads) {
  int i, j, k;
  int sum;

#define A(I,J) (AA[(I) * N + (J)])
#define B(I,J) (BB[(I) * N + (J)])
#define C(I,J) (CC[(I) * N + (J)])

#pragma omp target enter data map(to:AA[0:N*N]) map(to:BB[0:N*N]) map(to:CC[0:N*N])

{
#pragma omp target teams num_teams(NumBlocks) thread_limit(NumThreads)
#pragma omp distribute collapse(2)  
    for (i = 0; i < N; ++i) {
      for (j = 0; j < N; ++j) {
 int       sum = 0;
        for (k = 0; k < N; ++k)
          C(i,j) += A(i, k) * B(k, j);
        
      }
    }
  }

#pragma omp target exit data map(from:CC[0:N*N])
}

_Bool check(int N, int *CC, int *AA, int *BB) {
  int i, j, k;
  int sum;

#undef A
#undef B
#undef C

#define A(I,J) AA[(I) * N + J]
#define B(I,J) BB[(I) * N + J]
#define C(I,J) CC[(I) * N + J]

  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      sum = 0;

      for (k = 0; k < N; ++k)
        sum = sum + A(i, k) * B(k, j);

      if (sum != C(i, j)) {
        std::printf("%d %d %d %d\n", i, j, sum, C(i, j));
        return false;
      }
    }
  }

  return true;
}

void initialize(int N, int *AA, int *BB) {
  int i, j;

#define A(I,J) AA[(I) * N + J]
#define B(I,J) BB[(I) * N + J]

#pragma omp parallel for private(j)
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      A(i, j) = random() % 50;

#pragma omp parallel for private(j) 
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j) 
      B(i, j) = random() % 100;
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

  N = std::atoi(argv[1]);
  A = new int[N * N];
  B = new int[N * N];
  C = reinterpret_cast<int *>(std::calloc(N * N, sizeof(int)));

  initialize(N, A, B);

  uint32_t NumBlocksSide = ((N - 1) / 32) + 1;
  uint32_t NumBlocks  = NumBlocksSide * NumBlocksSide;
  uint32_t NumThreads = 32 * 32;

  start = rtdsc();
  multiply(N, C, A, B, NumBlocks, NumThreads);
  end   = rtdsc();

  result_str = check(N, C, A, B) ? "Correct" : "Wrong";
  printf("Execution time: %ld cycles\n%s %d %d %d", end - start, result_str,C[1*N+1], A[1*N+1], B[1*N+1]);

  std::free(C);
  delete B;
  delete A;

  return 0;
}

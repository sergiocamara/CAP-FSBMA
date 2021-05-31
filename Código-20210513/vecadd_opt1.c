#include <stdio.h>
#include <omp.h>

// large enough to force into main memory
#define ARRAY_SIZE 80000000
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

// parallelized adding vector
void vector_add(double *c, double *a, double *b, int n) {
#pragma omp parallel for
   for (int i=0; i < n; i++){
      c[i] = a[i] + b[i];
   }
}

int main(int argc, char *argv[]){
   #pragma omp parallel
      if (omp_get_thread_num() == 0)
         printf("Running with %d thread(s)\n",omp_get_num_threads());

   double tstart = omp_get_wtime();

   #pragma omp parallel 
   { // 3. 
   
    #pragma omp parallel for // 2. add to parrallelize inicialization
    for (int i=0; i<ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    #pragma omp master // 3. 
        vector_add(c, a, b, ARRAY_SIZE);
   }
   printf("Runtime is %lf msecs\n", omp_get_wtime() - tstart);
}

  
  
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define ORDERS_OF_MAGNITUDE 1.0e9;
double PairwiseSumBySubtask(double* restrict var, long nstart, long nend);

double PairwiseSumByTask(double* restrict var, long ncells)
{
   double sum;
   #pragma omp parallel
   {
      #pragma omp master
      {
         sum = PairwiseSumBySubtask(var, 0, ncells);
      }
   }
   return(sum);
}

double PairwiseSumBySubtask(double* restrict var, long nstart, long nend)
{
   long nsize = nend - nstart;
   long nmid = nsize/2;
   double x,y;
   if (nsize == 1){
      return(var[nstart]);
   }

   #pragma omp task shared(x) mergeable final(nsize > 10)
   x = PairwiseSumBySubtask(var, nstart, nstart + nmid);
   #pragma omp task shared(y) mergeable final(nsize > 10)
   y = PairwiseSumBySubtask(var, nend - nmid, nend);
   #pragma omp taskwait

   return(x+y);
}

int main(int argc, char **argv)
{
#pragma omp parallel
   if (omp_get_thread_num() == 0) printf("\n  Running with %d thread(s)\n\n",omp_get_num_threads());

   for (int pow_of_two = 8; pow_of_two < 31; pow_of_two++){
      long ncells = (long)pow((double)2,(double)pow_of_two);
      long ncellsdiv2 = ncells/2;

      printf("SETTINGS INFO -- ncells %ld log %d\n",ncells,(int)log2((double)ncells));

      double high_value = 1.0e-1;
      double low_value  = 1.0e-1/ORDERS_OF_MAGNITUDE;
      double accurate_sum = (double)ncellsdiv2 * high_value +
                            (double)ncellsdiv2 * low_value;

      double *energy = (double *)malloc(ncells*sizeof(double));

      // Initialize with high values first
      for (long i = 0; i < ncells; i++){
         energy[i] = (i < ncellsdiv2) ? high_value : low_value;
      }

      double test_sum;
      double cpu_time;

//******************************************************

      double tstart = omp_get_wtime();

      test_sum = PairwiseSumByTask(energy, ncells);

      double tstop = omp_get_wtime();

      printf("  accurate sum %-17.16lg sum %-17.16lg diff %10.4lg relative diff %10.4lg runtime %lf",
             accurate_sum,test_sum,(test_sum-accurate_sum),((test_sum-accurate_sum)/accurate_sum), tstop - tstart);
      printf("   Pairwise Sum by Task\n\n");

//******************************************************

      free(energy);
   }
}




#include<stdio.h>
#include<omp.h>


int main() {
    double x = 2.0; 
    int n = 10;
    #pragma omp parallel for firstprivate(x) 
        for (int i=0; i < n; i++){
            //x = 1.0 ;
            double y = x*2.0; 
            printf("%6.2f\n", x);
        } 


    double z = x; 

    printf("%6.2f\n", z);
}

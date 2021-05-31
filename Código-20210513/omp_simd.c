
#include<stdio.h>
#include<omp.h>

int main() {
    int N = 10;
    int i, suma = 0; 
    
    int a[N], b[N], c[N];
    a[0] = 0;
    c[0] = 100;

    //#pragma omp parallel for
    #pragma omp simd
    for (i = 1; i < N; i++)
    {
        a[i] = a[i-1] + 1;
        b[i] = *c + 1;
    }
        


    for (i = 0; i < N; i++) {
        printf("%d\t", a[i]);
    }

    printf("\n%d\n", suma);
    



}
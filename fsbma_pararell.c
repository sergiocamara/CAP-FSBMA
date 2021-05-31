#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>


/* Los parámetros establecidos. Son fijos. */
#define WIDTH 1280
#define HEIGHT 720
#define BS 16
#define SA 24

float MSE(unsigned char *bloque_actual, unsigned char *bloque_referencia);

int main(void)
{
    unsigned char *ref; /* Datos del frame de referencia */
    unsigned char *act; /* Datos del frame actual */
    FILE *fref, *fact;
    unsigned char linea[WIDTH];
    char Vx[(HEIGHT / BS)][(WIDTH / BS)];
    char Vy[(HEIGHT / BS)][(WIDTH / BS)];
    float costes[(HEIGHT / BS)][(WIDTH / BS)];
    
    struct timespec inicio,fin;

    /* Reservamos memoria */
    ref = (unsigned char *)malloc((WIDTH + 2 * SA) * (HEIGHT + 2 * SA)); /* Extensión de bordes */
    if (ref == NULL)
    {
        printf("Error al obtener memoria para el frame de referencia\n");
        return -1;
    }
    act = (unsigned char *)malloc(WIDTH * HEIGHT);
    if (act == NULL)
    {
        printf("Error al obtener memoria para el frame actual\n");
        free(ref);
        return -2;
    }

    /* Leemos la información de los ficheros */
    fref = fopen("FrameReferencia.data", "r");
    if (fref == NULL)
    {
        printf("Error al abrir el archivo con los datos del frame de referencia\n");
        free(ref);
        free(act);
        return -3;
    }

    fact = fopen("FrameActual.data", "r");
    if (fact == NULL)
    {
        printf("Error al abrir el archivo con los datos del frame actual\n");
        free(ref);
        free(act);
        fclose(fref);
        return -3;
    }

    /* Frame actual es directo */
    fread(act, WIDTH * HEIGHT, sizeof(unsigned char), fact);
    /* Leer el frame de referencia debe tener en cuenta la extensión de bordes */

    /* Usado para depuración */
    /* int cnt1 = 0, cnt2 = 0; */
    for (unsigned int y = 0; y < HEIGHT + 2 * SA; y++)
    {
        if (y == 0)
        {
            fread(linea, WIDTH, sizeof(unsigned char), fref);
           /* cnt1++; */
        }
        else if (y > SA && y < HEIGHT + SA)
        {
            fread(linea, WIDTH, sizeof(unsigned char), fref);
            /* cnt1++; */
        }
        for (unsigned int x = 0; x < WIDTH + 2 * SA; x++)
        {
            if (x < SA)
            {
                ref[y * (WIDTH + 2 * SA) + x] = linea[0];
            }
            else if (x >= SA && x < WIDTH + SA)
            {
                ref[y * (WIDTH + 2 * SA) + x] = linea[x - SA];
            }
            else
            {
                ref[y * (WIDTH + 2 * SA) + x] = linea[WIDTH - 1];
            }
        }
    }
    double start; 
    double end; 
    start = omp_get_wtime(); 

    /* Inicializar los costes mínimos asociados a cada bloque */
    #pragma omp parallel for collapse(1) shared (costes) 
    for (unsigned int y = 0; y < HEIGHT / BS; y++)
    {
        for (unsigned int x = 0; x < WIDTH / BS; x++)
        {
            costes[y][x] = __FLT_MAX__;
        }
    }

    // clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&inicio);
    /* Los datos ahora sí que están preparados. Procedemos a calcular los costes */
    // #pragma omp parallel for collapse(2) private(x,y,costes)
    // #pragma omp parallel for collapse(1)
    // #pragma omp parallel for collapse(2)
    for (unsigned int y = 0; y < HEIGHT; y += BS)
    {
        for (unsigned int x = 0; x < WIDTH; x += BS)
        {
            /* Calcular MSE para todos los bloques en el área de búsqueda. 
           Las coordenadas en ref y act están alineadas.   */
        //    #pragma omp parallel for collapse(2) shared (Vx,Vy,costes) private(x,y)
            for (unsigned char j = 0; j < 2 * SA; j++)
            {
                for (unsigned char k = 0; k < 2 * SA; k++)
                {
                    /*Calcular MSE */
                    float coste_bloque = MSE(&act[y * WIDTH + x], &ref[(y + j) * (WIDTH + 2 * SA) + (x + k)]);
                    /* Podría haber una optimización. A igualdad de coste, elegir aquel cuyo vector de movimiento tenga la menor distancia
                    respecto al origen */
                    if (coste_bloque < costes[y / BS][x / BS])
                    {
                        costes[y / BS][x / BS] = coste_bloque;
                        Vx[y / BS][x / BS] = j - SA;
                        Vy[y / BS][x / BS] = k - SA;
                    }
                    
                }
            }
        }
     
    }

    // clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&fin);
    end = omp_get_wtime(); 

    printf("\nVectores de movimiento\n");
    for (unsigned char y = 0; y < HEIGHT / BS; y++)
    {
        for (unsigned char x = 0; x < WIDTH / BS; x++)
        {

            printf("[%d,%d] ", Vy[y][x], Vx[y][x]);
        }
        printf("\n");
    }

    // printf("Tiempo transcurrido: %ld.%ld segundos.\n", fin.tv_sec-inicio.tv_sec,(fin.tv_nsec-inicio.tv_nsec)%(int)(1e+9));
    printf("Work took %f seconds\n", end - start);

    fclose(fref);
    fclose(fact);
    free(ref);
    free(act);

    return 0;
}

float MSE(unsigned char *bloque_actual, unsigned char *bloque_referencia)
{
    float error = 0;
    for (unsigned char y = 0; y < BS; y++)
    {
        for (unsigned char x = 0; x < BS; x++)
        {

            error += pow((bloque_actual[y * WIDTH + x] - bloque_referencia[y * (WIDTH + 2 * SA) + x]), 2);
        }
       
    }
    return error / (BS*BS);
}

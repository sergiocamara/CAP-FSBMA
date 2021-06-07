secuential:
	gcc -o fsbma fsbma.c -lm
	./fsbma

parallel:
	gcc -fopenmp -o fsbma_parallel fsbma_parallel.c -lm
	./fsbma_parallel



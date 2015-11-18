// http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
/* C stuff */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

/* Cuda stuff */
#include <cuda_runtime_api.h>
#include <cuda.h>

#define	N	(2048*2048)
#define TH_PER_BLOCK	512

__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

void randomints(int* a){
   int i;
   for (i = 0; i < N; ++i)
    a[i] = 1;
}

int main(void) {
	int *a,*b,*c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);
	
	cudaMalloc((void**) &d_a, size);
	cudaMalloc((void**) &d_b, size);
	cudaMalloc((void**) &d_c, size);
	
	a = (int*)malloc(size); randomints(a);
	b = (int*)malloc(size); randomints(b);
	c = (int*)malloc(size);

	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<N/TH_PER_BLOCK,TH_PER_BLOCK>>>(d_a, d_b, d_c);
	
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	
	printf("It is %d\n", c[0]);
	
	free(a); free(b); free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}

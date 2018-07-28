/*
  Musa AYDIN, Fatih Sultan Mehmet Vakif University,
  Istanbul, Turkey
  maydin@fsm.edu.tr
  this program, calculates scaler dot products,
  using threads synchronization,
  first implementation in kernel each thread own private variable
  second, each thread share data using shared memory structure
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#define N 32
#define THREADS_PER_BLOCK 6

void init_matrix(int *a, int len) {
	int i;
	
	for ( i = 0; i < len; i++)
	{
		a[i] = i + 1;
	}
}
void print_array(int *a, int len) {
	int i;
	if (len > 16) {
		len = 16;
	}
	for (i = 0; i < len; i++)
	{
		printf("%d ", a[i]);
	}
	printf("\n");
}

__global__ void dotSharedMem(int *a, int *b, int *c) {
	// temp deðiþkeni her bir thread için private olan deðiþkendir,
	// her bir thread in kendi temp deðiþkeni vardýr.
	__shared__ int temp[THREADS_PER_BLOCK];
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	 //temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
	if (id < N) {
		temp[id] = a[id] * b[id];

		__syncthreads();

		if (0 == threadIdx.x) {
			int sum = 0;
			for (int i = 0; i < THREADS_PER_BLOCK; i++)
				sum += temp[i];
			atomicAdd(c, sum);
			// atomicAdd(&c[0], sum);
			//*c = sum;
		}
	}
	 

}
int main(void) {	

	int *a, *b, *c; // copies of a, b, c
	int *dev_a, *dev_b, *dev_c; // devices copies of a, b, c
	int size = N * sizeof(int); // allocate device copies of a, b, c

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, sizeof(int));

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(sizeof(int));
	//*c = 0;

	init_matrix(a, N);
	print_array(a, N);
	init_matrix(b, N);
	print_array(b, N);

	// copy inputs to device
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	int blck = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	// _blocks yani girid dim mtrisi kaç adet block a böleceðimizi belirliyoruz
	//	dim3 _blocks(blck, n / _threadsSize); // dimension of grid

	dim3 _blocks(blck, blck); // dimension of grid
	dim3 _threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK); // dimension of block


	dotSharedMem << < (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(dev_a, dev_b, dev_c);

	// copy device result back to host copy of c
	cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("dot product result = %d\n", *c);
	//printf("%d\n", c);
	//print_array(c, N);

	free(a); free(b); free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	getchar();
	return 0;
}



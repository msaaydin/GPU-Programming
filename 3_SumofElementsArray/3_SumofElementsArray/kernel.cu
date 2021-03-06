
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "gputimer.h"


#define NUM_THREADS 100
#define ARRAY_SIZE  10

#define BLOCK_WIDTH 5

void print_array(int *array, int size)
{
	printf("{ ");
	for (int i = 0; i < size; i++) { printf("%d ", array[i]); }
	printf("}\n");
}



__global__ void increment_atomic(int *g)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d. blok %d. index \n ", blockIdx.x, i);
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	atomicAdd(&g[i], 1);
	__syncthreads();
}

int main(int argc, char **argv)
{
	GpuTimer timer;
	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);


	int h_array[ARRAY_SIZE];
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int * d_array;
	cudaMalloc((void **)&d_array, ARRAY_BYTES);
	cudaMemset((void *)d_array, 0, ARRAY_BYTES);

	timer.Start();	

	//printf("***************atomic adds result..*************\n");
	//cudaDeviceSynchronize();
	increment_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_array);
	timer.Stop();

	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	print_array(h_array, ARRAY_SIZE);
	printf("Time elapsed = %g ms\n", timer.Elapsed());
	

	cudaFree(d_array);
	return 0;
}
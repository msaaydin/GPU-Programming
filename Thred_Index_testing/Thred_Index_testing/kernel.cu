
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#define Size 122
#define N 100
#define thread_size 10
// thread_index calculation
__global__ void thread_index_test(int *a) {
	//int id = threadIdx.x;
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < Size) {
		a[id] = threadIdx.x;
	}
}


int main(void) {
	int A[N][N] = { { 1,2 },{ 3,4 } };
	int B[N][N] = { { 5,6 },{ 7,8 } };
	int C[N][N] = { { 0,0 },{ 0,0 } };
	int *h_a;
	int *d_a;
	h_a = (int *)malloc(sizeof(int)*Size);
	cudaMalloc((void **)&d_a, sizeof(int)*Size);
	for (int i = 0; i < Size; i++) {
		h_a[i] = -1;
	}
	dim3 grid(Size, Size);
	cudaMemcpy(d_a, h_a, sizeof(int)*Size, cudaMemcpyHostToDevice);
	// kernel i çaðýrýyoruz...
	thread_index_test << <(Size + thread_size - 1) / thread_size, thread_size >> >(d_a);
	cudaMemcpy(h_a, d_a, sizeof(int)*Size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < Size; i++) {
		printf("%d = %d\n", i, h_a[i]);
	}
	cudaFree(d_a);
	free(h_a);

	return 0;
}


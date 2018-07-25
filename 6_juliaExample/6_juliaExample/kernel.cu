
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N   (4096 * 4096)

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main(void) {
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the CPU
	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	c = (int*)malloc(N * sizeof(int));

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i<N; i++) {
		a[i] = i+1;
		b[i] = 2 * i+1;
	}

	// copy the arrays 'a' and 'b' to the GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice));
	add << <(N + 128-1) / 128, 128 >> >(dev_a, dev_b, dev_c);
	//add << <128, 128 >> >(dev_a, dev_b, dev_c);


	// copy the array 'c' back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost));

	// verify that the GPU did the work we requested
	int temp;
	for (int i = 0; i<N; i+=150) {
		if (c[i] == 0) {
			temp = i;
			break;
		}
		printf("  %d + %d = %d\n", a[i], b[i], c[i]);
		
	}

	// free the memory we allocated on the GPU
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	// free the memory we allocated on the CPU
	free(a);
	free(b);
	free(c);

	return 0;
}



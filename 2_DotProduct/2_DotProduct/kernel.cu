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








int main(void) {
	float   *a, *b, c, *partial_c;
	float   *dev_a, *dev_b, *dev_partial_c;

	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

	// allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float));

	// fill in the host memory with data
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy(dev_a, a, N*sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(float),
		cudaMemcpyHostToDevice);

	dot << <blocksPerGrid, threadsPerBlock >> >(dev_a, dev_b,
		dev_partial_c);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy(partial_c, dev_partial_c,
		blocksPerGrid*sizeof(float),
		cudaMemcpyDeviceToHost);

	// finish up on the CPU side
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c,
		2 * sum_squares((float)(N - 1)));

	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
}

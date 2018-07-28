
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <stdio.h>
#include<stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1<<16
int log2(int i)
{
	int r = 0;
	while (i >>= 1) r++;
	return r;
}

int bit_reverse(int w, int bits)
{
	int r = 0;
	for (int i = 0; i < bits; i++)
	{
		int bit = (w & (1 << i)) >> i;
		r |= bit << (bits - i - 1);
	}
	return r;
}

__global__ void naive_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int myItem = d_in[myId];
	int myBin = myItem % BIN_COUNT;
	d_bins[myBin]++;
}

__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	if (myId < N) {
		int myItem = d_in[myId];
		//int myBin = myItem % BIN_COUNT;
		atomicAdd(&(d_bins[myItem]), 1);
	}

}
void printAr(int *a, int len) {
	printf("****************************\n");
	for (int i = 0; i < len; i++)
	{
		printf("%d ", a[i]);
	}
	printf("\n***************************\n");
}

int main(int argc, char **argv)
{
	GpuTimer Timer;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		printf("error: no devices supporting CUDA.\n");

		exit(1);
	}
	int dev = 0;
	cudaSetDevice(dev);

	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (int)devProps.totalGlobalMem,
			(int)devProps.major, (int)devProps.minor,
			(int)devProps.clockRate);
	}

	const int ARRAY_SIZE = N;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
	const int BIN_COUNT = 256;
	const int BIN_BYTES = BIN_COUNT * sizeof(int);
	int *h_a = new int[ARRAY_SIZE];
	// generate the input array on the host
	//int h_in[ARRAY_SIZE];
	//int *h_in;
	//h_in = (int *)malloc(ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		//h_in[i] = rand() % 256;//bit_reverse(i, log2(ARRAY_SIZE));
		h_a[i] = rand() % 256;//i + 1;
	}
	printf("array print\n");
	//printAr();
	printAr(h_a, 32);

	int h_bins[BIN_COUNT];
	for (int i = 0; i < BIN_COUNT; i++) {
		h_bins[i] = 0;
	}

	// declare GPU memory pointers
	int * d_in, *d_bins;

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_bins, BIN_BYTES);
	Timer.Start();
	// transfer the arrays to the GPU
	cudaMemcpy(d_in, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

	int whichKernel = 1;
	if (argc == 2) {
		whichKernel = 2;
	}

	// launch the kernel
	switch (whichKernel) {
	case 0:
		printf("Running naive histo\n");
		naive_histo << <ARRAY_SIZE / 64, 64 >> >(d_bins, d_in, BIN_COUNT);
		break;
	case 1:
		printf("Running simple histo\n");
		simple_histo << <ARRAY_SIZE / 64, 64 >> >(d_bins, d_in, BIN_COUNT);
		break;
	default:
		printf("error: ran no kernel\n");
		exit(1);
	}

	// copy back the sum from GPU
	cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);
	Timer.Stop();
	int sum = 0;
	for (int i = 0; i < BIN_COUNT; i++) {
		printf("bin %d: count %d\n", i, h_bins[i]);
		sum += h_bins[i];
	}
	printf("histogtram execution time = %f", Timer.Elapsed());
	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_bins);
	//free(&h_in);
	delete[] h_a;
	getchar();
	return 0;
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h" 
#include <stdio.h>
#define n 4
#define m 4
#include <math.h>
#define TILE_WIDTH 2 // blockDim ile ayný bu yani thread size 

//for addition
__global__  void kernel_matrix_addition(int *array1, int *array2, int *result, int WIDTH1)
{
	// calculate thread id
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	int offset = col + row * blockDim.x * gridDim.x;

	//int idx = i + j*N;
	if (col < n && row < m)
		result[row*WIDTH1 + col] = array1[row*WIDTH1 + col] + array2[row*WIDTH1 + col];
	//if (col < n && row < m)
	//result[offset] = array1[offset] + array2[offset];
}

/*
__global__ void MatrixMul(float *Md, float *Nd, float *Pd, const int WIDTH1)

{

// calculate thread id
unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x;
unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y;
for (int k = 0; k < WIDTH1; k++)
{
Pd[row*WIDTH1 + col] += Md[row * WIDTH1 + k] * Nd[k * WIDTH1 + col];
}
}*/
// shared

__global__ void MatrixMulSh(int *Md, int *Nd, int *Pd, const int WIDTH1)
{

	//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
	// calculate thread id
	unsigned int col = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y*blockIdx.y + threadIdx.y;
	if (col < n && row < m) {
		for (int i = 0; i < WIDTH1 / TILE_WIDTH; i++) // m indicate number of phase
		{
			Mds[threadIdx.y][threadIdx.x] = Md[row*WIDTH1 + (i*TILE_WIDTH + threadIdx.x)];
			Nds[threadIdx.y][threadIdx.x] = Nd[(i*TILE_WIDTH + threadIdx.y) * WIDTH1 + col];
			__syncthreads(); // for syncronizeing the threads
			for (int k = 0; k < TILE_WIDTH; k++)
				Pd[row*WIDTH1 + col] += Mds[threadIdx.x][k] * Nds[k][threadIdx.y];
			__syncthreads(); // for syncronizeing the threads
		}
	}


}


long sumMATrix(int *C, int len) {
	int i;
	long sum = 0;
	for (i = 0; i < len*len; i++)
	{
		sum += C[i];
	}
	return sum;
}
void printMatrix(int *C, int len) {
	printf("*************print matrix***************\n ");
	for (int i = 0; i < len*len; i++)
	{
		printf("%d ", C[i]);
	}
	printf("****************************\n ");

}


int main()
{
	int _threadsSize = 2;

	int *A, *B, *C, *dev_A, *dev_B, *result_d;
	A = (int*)malloc(n*m*sizeof(int));
	B = (int*)malloc(n*m*sizeof(int));
	C = (int*)malloc(n*m*sizeof(int));

	cudaMalloc((void**)&dev_A, n*m*sizeof(int));
	cudaMalloc((void**)&dev_B, n*m*sizeof(int));
	cudaMalloc((void**)&result_d, n*m*sizeof(int));



	long sum = 0;
	for (int i = 0; i < n*m; i++) {
		A[i] = rand()%10;
		B[i] = rand()%10;
		sum += A[i];
	}
	printMatrix(A, m);
	printMatrix(B, m);
	printf("A matrisi toplami = %ld\n", sum);
	cudaError_t status;
	status = cudaMemcpy(dev_A, A, n*m*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		fprintf(stderr, "a matrisini kopyalarken olusan hata: %s\n", cudaGetErrorString(status));

	status = cudaMemcpy(dev_B, B, n*m*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		fprintf(stderr, "b matrisini kopyalarken olusan hata: %s\n", cudaGetErrorString(status));
	int blck = (n + _threadsSize - 1) / _threadsSize;
	// _blocks yani girid dim mtrisi kaç adet block a böleceðimizi belirliyoruz
	//	dim3 _blocks(blck, n / _threadsSize); // dimension of grid

	dim3 _blocks(blck, blck); // dimension of grid
	dim3 _threads(_threadsSize, _threadsSize); // dimension of block

											   //kernel_matrix_addition << <_blocks, _threads >> > (array1_d, array2_d, M_result_array_d, N);
	kernel_matrix_addition << <_blocks, _threads >> > (dev_A, dev_B, result_d, n);

	// all gpu function blocked till kernel is working
	//copy back result_array_d to result_array_h
	//cudaMemcpy(M_result_array_h, M_result_array_d, r*r*sizeof(int), cudaMemcpyDeviceToHost);
	status = cudaMemcpy(C, result_d, m*n*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		fprintf(stderr, "device tan host a geri donuste olusan hata: %s\n", cudaGetErrorString(status));
	//printf the result array
	printf("sum of A+B matrix = %ld\n", sumMATrix(C, m));
	cudaDeviceSynchronize();
	// matrix çarpýmý için kernel çaðrýmý
	status = cudaMemcpy(dev_A, A, n*m*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		fprintf(stderr, "a matrisini kopyalarken olusan hata: %s\n", cudaGetErrorString(status));

	status = cudaMemcpy(dev_B, B, n*m*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		fprintf(stderr, "b matrisini kopyalarken olusan hata: %s\n", cudaGetErrorString(status));
	C = (int*)malloc(n*m*sizeof(int));
	MatrixMulSh << <_blocks, _threads >> > (dev_A, dev_B, result_d, n);
	status = cudaMemcpy(C, result_d, m*n*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		fprintf(stderr, "device tan host a geri donuste olusan hata: %s\n", cudaGetErrorString(status));
	printMatrix(C, m);

	free(A);
	free(B);
	free(C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(result_d);
	getchar();
	return 0;
}



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <conio.h>
#include<ctime>
#include <iostream>
#define N 256
#define threads_per_block 128 


int** Matrix_Alloc(int** matrix, int rows, int columns)
{
	int i;
	matrix = (int**)malloc(sizeof(int)*rows*columns);
	if (matrix != NULL)
	{
		for (i = 0; i < rows; i++)
		{
			matrix[i] = (int*)malloc(sizeof(int)*columns);
			if (matrix[i] == NULL)
			{
				return NULL;
			}
		}
	}
	return matrix;
}
void Matrix_Free(int** matrix, int rows)
{
	int i;
	for (i = 0; i < rows; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}
void Matrix_Print(int** matrix, int rows, int columns)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
		{
			printf("%d - ", matrix[i][j]);
		}
		putchar('\n');
	}
}

void print_Mat(int** a, int len) {
	printf("*************************\n");
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			printf("%d - ", a[i][j]);
		}
		printf("\n");

	}
	printf("*************************\n");

}
long sum_Mat(int **a, int len) {
	printf("*************************\n");
	int sum = 0;
	for (int i = 0; i < len; i++)
	{
		for (int j = 0; j < len; j++)
		{
			sum += a[i][j];
		}
	}
	printf("*************************\n");
	return sum;
}
__global__ void add2Dmatrix(int *a, int *b, int *c)
{
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	int globalIndex = col + row * N*N;
	if (col > N  || row > N) {
		c[globalIndex] = a[globalIndex] + b[globalIndex];
	}

}
int main()
{
	int *dev_a, *dev_b = 0, *dev_c;
	

	int **h_a = NULL, **h_b = NULL, **h_c = NULL;
	/*Allocating the matrix*/
	h_a = Matrix_Alloc(h_a, N, N);
	h_b = Matrix_Alloc(h_b, N, N);
	h_c = Matrix_Alloc(h_c, N, N);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_a[i][j] = 5;
			h_b[i][j] = 8;
		}
	}

	printf("sum of element matrix = %d\n", sum_Mat(h_a, N));

	Matrix_Print(h_a, 16, 16);



	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, N*N * sizeof(int));
	cudaMalloc((void**)&dev_b, N*N * sizeof(int));
	cudaMalloc((void**)&dev_c, N*N * sizeof(int));

	//copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
	cudaMemcpy(dev_a, h_a, N*N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, h_b, N*N * sizeof(int), cudaMemcpyHostToDevice);

	add2Dmatrix << <(N+(threads_per_block-1)) / threads_per_block, threads_per_block >> >(dev_a, dev_b, dev_c);
	cudaMemcpy(h_c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	printf("**********************\n");
	printf("sum of element matrix = %d\n", sum_Mat(h_c, N));

	Matrix_Print(h_c, 16, 16);


	Matrix_Free(h_a, N);
	Matrix_Free(h_b, N);
	Matrix_Free(h_c, N);
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return 0;
}

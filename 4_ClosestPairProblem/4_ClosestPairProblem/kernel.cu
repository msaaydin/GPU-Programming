
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <conio.h>
#include<ctime>
#include <iostream>
#include "FindClosestCPU.h"

using namespace std;
__global__ void FindClosestGPU(float3d* points, int* indices, int count) {
	if (count <= 1) return;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < count) {
		float3d thisPoint = points[idx];
		float smallerSoFar = 3.40282e38f;
		for (int i = 0; i < count; i++)
		{
			if (i == idx)continue;
			float dist = (thisPoint.x - points[i].x) * (thisPoint.x - points[i].x);
			dist += (thisPoint.y - points[i].y) * (thisPoint.y - points[i].y);
			dist += (thisPoint.z - points[i].z) * (thisPoint.z - points[i].z);
			if (dist < smallerSoFar) {
				smallerSoFar = dist;
				indices[idx] = i;
			}

		}
	}
}
int main()
{

	int iteration = 1;
	int blockdim = 128; // threadsize, thread number
	// number of points
	const int count = 10000;
	int *indexOfClosest = new int[count];
	float3d *points = new float3d[count];
	float3d* d_points;
	int* d_indexofClosest;
	for (int i = 0; i < count; i++)
	{
		points[i].x = (float)((rand() % 10000) - 5000);
		points[i].y = (float)((rand() % 10000) - 5000);
		points[i].z = (float)((rand() % 10000) - 5000);

	}
	long fatstest = 1000000;
	// alogirtmayý 10 defa çalýþtýr
	cout << "CPU calisma sonucu " << endl;

	/*for (int i = 0; i < iteration; i++)
	{
		long starttime = clock();

		FindClosestCPU(points, indexOfClosest, count);
		long finishtime = clock();
		cout << i + 1 << ". calisma sonucu gecen sure = " << (finishtime - starttime) << "  ms" << endl;
	}*/
	cout << "CPU bulunan index sonuc=" << endl;
	for (int i = 0; i < iteration; i++)
	{
		cout << i << "." << indexOfClosest[i] << endl;
	}
	cudaMalloc((void**)&d_points, sizeof(float3d)*count);
	cudaMalloc((void**)&d_indexofClosest, sizeof(int)*count);
	cudaMemcpy(d_points, points, sizeof(float3d)*count, cudaMemcpyHostToDevice);
	cout << "GPU calisma sonucu " << endl;
	
	for (int i = 0; i < iteration; i++)
	{
		long start = clock();
		FindClosestGPU <<<(count / blockdim), blockdim >>>(d_points, d_indexofClosest, count);
		cudaMemcpy(points, d_points, sizeof(float3d)*count, cudaMemcpyDeviceToHost);
		cudaMemcpy(indexOfClosest, d_indexofClosest, sizeof(int)*count, cudaMemcpyDeviceToHost);

		long finish = clock();
		cout << i + 1 << ". calisma sonucu gecen sure = " << (finish - start) << "  ms" << endl;

	}
	// bulunan sonucun ilk 10 elemanýný yazdýralým..
	cout << "GPU bulunan index sonuc=" << endl;
	for (int i = 0; i < iteration; i++)
	{
		cout << i << "." << indexOfClosest[i] << endl;
	}
	delete[] indexOfClosest;
	delete[] points;
	cudaFree(d_points);
	cudaFree(d_indexofClosest);
	return 0;
}


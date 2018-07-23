// closestpair.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <conio.h>
#include<ctime>
#include <iostream>
#include "FindClosestCPU.h"
using namespace std;
int main()
{
	// number of points
	const int count = 10000;
	int *indexOfClosest = new int[count];
	float3 *points = new float3[count];
	for (int i = 0; i < count; i++)
	{
		points[i].x = (float)((rand() % 10000) - 5000);
		points[i].y = (float)((rand() % 10000) - 5000);
		points[i].z = (float)((rand() % 10000) - 5000);

	}
	long fatstest = 1000000;
	// alogirtmayý 20 defa çalýþtýr
	for (int i = 0; i < 10; i++)
	{
		long starttime = clock();

		FindClosestCPU(points, indexOfClosest, count);
		long finishtime = clock();
		cout << i << ". calisma sonucu gecen sure = " << (finishtime - starttime) << "  ms" << endl;
	}

	// bulunan sonucun ilk 10 elemanýný yazdýralým..
	cout << "sonuc=" << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << i << "." << indexOfClosest[i]<<endl;
	}
	delete [] indexOfClosest;
	delete[] points;
	return 0;
}


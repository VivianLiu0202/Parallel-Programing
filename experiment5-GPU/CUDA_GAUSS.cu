#include "cuda_runtime.h"
#include<device_launch_parameters.h>
#include <iostream>
#include <random>
#include <ctime>
#include<windows.h>
#define MATRIX_SIZE 128
#define MATRIX_MAX_NUMBER 16
#define BLOCK_SIZE 128
long long head, tail, freq;
void timestart()
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
}
void timestop()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time
	std::cout << ((tail - head) * 1000.0 / freq) / 1000 << std::endl;


}
typedef double matrix[MATRIX_SIZE + 1][MATRIX_SIZE + 1];

matrix A;
matrix A_GPUresult;
double b[MATRIX_SIZE];
//double b[MATRIX_SIZE];
//double y[MATRIX_SIZE];

__host__
void generateMatrix();

//__host__
//void generateVectors();

__host__
void printMatrix(matrix mat);

__host__
void solveOnCPU();

__host__
bool solveOnGPU();

__host__
void generateMatrix()
{
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		for (int j = 0; j < MATRIX_SIZE + 1; j++)
		{
			if (i == j)
				A[i][j] = (double)(rand() % MATRIX_MAX_NUMBER) + 5.0;
			else
				A[i][j] = (double)(rand() % MATRIX_MAX_NUMBER) + 1.0;
		}
	}
}

__host__
void printMatrix(matrix mat)
{
	for (int i = 0; i < MATRIX_SIZE + 1; i++)
	{
		std::cout << "[";
		for (int j = 0; j < MATRIX_SIZE; j++)
			std::cout << " " << mat[i][j] << ",";
		std::cout << " " << mat[i][MATRIX_SIZE] << " ]\n";
	}
}

__host__
void solveOnCPU()
{
	for (int k = 0; k < MATRIX_SIZE; k++)
	{
		double temp = A[k][k];
		double* selectedRow = A[k];
		for (int j = k; j < MATRIX_SIZE + 1; j++)
		{
			selectedRow[j] /= temp;
		}
		for (int i = k + 1; i < MATRIX_SIZE; i++)
		{
			temp = A[i][k];
			for (int j = k; j < MATRIX_SIZE + 1; j++)
			{
				A[i][j] -= selectedRow[j] * temp;
			}
		}
	}
	for (int j = MATRIX_SIZE - 1; j > 0; j--)
	{
		for (int i = 0; i < j; i++)
		{
			A[i][MATRIX_SIZE] -= A[j][MATRIX_SIZE] * A[i][j];
		}
	}
}

int main()
{
	srand((unsigned int)time(NULL));
	int totalFail = 0;
	for (int j = 0; j < 10; j++)
	{
		generateMatrix();
		timestart();
		solveOnGPU();
		timestop();
		timestart();
		solveOnCPU();
		timestop();
		std::cout << std::endl;

		int fail = 0;
		for (int i = 0; i < MATRIX_SIZE; i++)
		{
			if (abs(A_GPUresult[MATRIX_SIZE][i] - A[i][MATRIX_SIZE]) > 0.01)
			{
				fail++;
			}
		}
		if (fail != 0)
			std::cout << "@";
		else
			//std::cout << ".";
			totalFail += fail;
	}
	return 0;
}

__constant__ int k;

__global__
void gpuSolveBottom(matrix d_A)
{
	int j = (blockIdx.x * blockDim.x + threadIdx.x) + k;

	__shared__ double temp;
	temp = d_A[k][k];
	double selectedRow = d_A[k][j] / temp;

	__syncthreads();

	for (int i = k + 1; i < MATRIX_SIZE; i++)
	{
		temp = d_A[i][k];
		d_A[i][j] -= selectedRow * temp;
		__syncthreads();
	}

	d_A[j][k] = selectedRow;
}

__global__
void gpuSolveTop(matrix d_A)
{
	int i = (blockIdx.x * blockDim.x + threadIdx.x);

	for (int j = MATRIX_SIZE - 1; j > 0; j--)
	{
		if (i < j)
		{
			d_A[MATRIX_SIZE][i] -= d_A[MATRIX_SIZE][j] * d_A[j][i];
			__syncthreads();
		}
	}
}


__host__
bool solveOnGPU()
{
	cudaError_t cudaStatus;
	matrix* d_A;
	int sizeOfMatrix = (MATRIX_SIZE + 1) * (MATRIX_SIZE + 1) * sizeof(double);
	cudaStatus = cudaMalloc((void**)&d_A, sizeOfMatrix);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed on d_A!\n";
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_A, A, sizeOfMatrix, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!\n" << cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}
	for (int i = 0; i < MATRIX_SIZE; i++)
	{
		cudaStatus = cudaMemcpyToSymbol(k, &i, sizeof(int));
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaMemcpyToSymbol failed at iteration " << k << "!\n";
			goto Error;
		}
		gpuSolveBottom << <1, MATRIX_SIZE + 1 - i >> > (*d_A);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "gpuSortEven kernel call failed at iteration " << k << "!\n"
				<< cudaGetErrorString(cudaStatus) << std::endl;
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching gpuSolveBottom!\n";
			goto Error;
		}
	}
	gpuSolveTop << <1, MATRIX_SIZE >> > (*d_A);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "gpuSortEven kernel call failed at iteration " << k << "!\n"
			<< cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!\n";
		goto Error;
	}
	cudaStatus = cudaMemcpy(A_GPUresult, d_A, sizeOfMatrix, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy (cudaMemcpyDeviceToHost) failed!\n";
		goto Error;
	}
Error:
	cudaFree(d_A);
	return false;
}
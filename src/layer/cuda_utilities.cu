#include <iostream>
#include <cuda_fp16.h>
#include "cuda_utilities.h"
#include "../../config.h"

#define TILE_WIDTH 32

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}

__global__ void matrixMultiplicationKernel_1(float* A, float* B, float* result, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) 
    {
        float val = 0.0f;

        // Tính tích vô hướng của hàng A và cột B
        for (int i = 0; i < n; i++) 
        {
            val += A[row * n + i] * B[i * k + col];
        }

        result[row * k + col] = val;
    }
}

__global__ void matrixMultiplicationKernel_2(float* A, float* B, float* result, int m, int n, int k)
{
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    float val = 0.0f;

    // Duyệt qua các "tiled blocks" để thực hiện phép nhân ma trận
    for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // Nạp dữ liệu từ A vào shared memory tile_A
        if (row < m && (i * TILE_WIDTH + threadIdx.x) < n) 
        {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];
        } 
        else 
        {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Nạp dữ liệu từ B vào shared memory tile_B
        if (col < k && (i * TILE_WIDTH + threadIdx.y) < n) 
        {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * k + col];
        } 
        else 
        {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Đồng bộ hóa các thread trong block để đảm bảo mọi thread đã nạp xong dữ liệu vào shared memory
        __syncthreads();

        // Tính tích của tile_A và tile_B
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        // Đồng bộ hóa các thread trong block trước khi tiếp tục lặp qua các tiles tiếp theo
        __syncthreads();
    }

    if (row < m && col < k) {
        result[row * k + col] = val;
    }
}

__global__ void matrixMultiplicationKernel_3(float* A, float* B, float* result, int m, int n, int k) 
{
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    float val = 0.0f;

    // Duyệt qua các "tiled blocks" để thực hiện phép nhân ma trận
    for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // Nạp dữ liệu từ A vào shared memory tile_A
        if (row < m && (i * TILE_WIDTH + threadIdx.x) < n) 
        {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];
        } 
        else 
        {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Nạp dữ liệu từ B vào shared memory tile_B
        if (col < k && (i * TILE_WIDTH + threadIdx.y) < n) 
        {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * k + col];
        } 
        else 
        {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Đồng bộ hóa các thread trong block để đảm bảo mọi thread đã nạp xong dữ liệu vào shared memory
        __syncthreads();

        // Tính toán giá trị trong tile, sử dụng unrolling
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) 
        {
            val += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x]; 
        }
        // Đồng bộ hóa trước khi nạp tile tiếp theo
        __syncthreads();
    }

    if (row < m && col < k) {
        result[row * k + col] = val;
    }
}

__global__ void matrixMultiplicationKernel_4(float* A, float* B, float* result, int m, int n, int k) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) 
    {
        half value = __float2half(0.0f);
        
        for (int i = 0; i < n; i++) 
        {
            // Chuyển từ float sang half
            half a = __float2half(A[row * n + i]); 
            half b = __float2half(B[i * k + col]);  

            // Nhân và cộng FP16
            value = __hadd(value, __hmul(a, b));  
        }

        // Lưu kết quả (chuyển từ FP16 về float)
        result[row * k + col] = __half2float(value);
    }
}

void matrixMultiplicationGPUWrapper(float* A, float *B, float *result, int m, int n, int k, int version)
{	
    dim3 blockSize(32, 32);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    const int size_A = m * n * sizeof(float);
    const int size_B = n * k * sizeof(float);
    const int size_result = m * k * sizeof(float);

    float *d_A, *d_B, *d_result;
    CHECK(cudaMalloc(&d_A, size_A));
    CHECK(cudaMalloc(&d_B, size_B));
    CHECK(cudaMalloc(&d_result, size_result));

    CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    switch (version) 
    {
        case 1: 
            matrixMultiplicationKernel_1<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k);
            break;
        case 2:
            matrixMultiplicationKernel_2<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k);
            break;
        case 3:
            matrixMultiplicationKernel_3<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k);
            break;
        case 4:
            matrixMultiplicationKernel_4<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k);
            break;
    }
    CHECK(cudaGetLastError());
    
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(result, d_result, size_result, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_result));
}
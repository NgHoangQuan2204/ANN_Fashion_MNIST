#include <iostream>
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

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

static GpuTimer timer;
void startTimer()
{
    timer.Start();
}

float stopTimer()
{
    timer.Stop();

	return timer.Elapsed();
}

__host__ __device__ int idx1D(int r, int c, int colSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return r * colSz + c;
}

__host__ __device__ int idx1D_col(int r, int c, int rowSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return c * rowSz + r;
}

__global__ void matrixMultiplicationKernel_1(float* A, float* B, float* result
                                                        , int m, int n, int k, int image)
{
    // Xác định chỉ số hàng và cột trong ma trận kết quả
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Kiểm tra điều kiện biên
    if (row < m && col < k) {
        float val = 0.0f;

        // Tính tích vô hướng của hàng A và cột B
        for (int i = 0; i < n; i++) {
            val += A[row * n + i] * B[i * k + col];
        }

        // Ghi kết quả vào ma trận C
        result[row * k + col] = val;
    }
}

__global__ void matrixMultiplicationKernel_2(float* A, float* B, float* result, int m, int n, int k, int image)
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
        if (row < m && (i * TILE_WIDTH + threadIdx.x) < n) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Nạp dữ liệu từ B vào shared memory tile_B
        if (col < k && (i * TILE_WIDTH + threadIdx.y) < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * k + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Đồng bộ hóa các thread trong block để đảm bảo mọi thread đã nạp xong dữ liệu vào shared memory
        __syncthreads();

        // Tính tích của tile_A và tile_B
        for (int j = 0; j < TILE_WIDTH; j++) {
            val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        // Đồng bộ hóa các thread trong block trước khi tiếp tục lặp qua các tiles tiếp theo
        __syncthreads();
    }

    // Ghi giá trị tính được vào ma trận kết quả C
    if (row < m && col < k) {
        result[row * k + col] = val;
    }
}

#define UNROLL_FACTOR 4
__global__ void matrixMultiplicationKernel_3(float* restrict A, float* restrict B, float* restrict result,
                                                     int m, int n, int k) {
    // Shared memory tiles cho ma trận A và B
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    float val = 0.0f;

    // Duyệt qua các "tiled blocks" để thực hiện phép nhân ma trận
    for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // Nạp dữ liệu từ A vào shared memory tile_A
        if (row < m && (i * TILE_WIDTH + threadIdx.x) < n) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * TILE_WIDTH + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Nạp dữ liệu từ B vào shared memory tile_B
        if (col < k && (i * TILE_WIDTH + threadIdx.y) < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * k + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Đồng bộ hóa các thread trong block để đảm bảo mọi thread đã nạp xong dữ liệu vào shared memory
        __syncthreads();

        // Tính toán giá trị trong tile, sử dụng unrolling
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            // Unroll vòng lặp tính toán nhân A và B
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; ++u) {
                val += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x] + u * (TILE_WIDTH / UNROLL_FACTOR);
            }
        }

        // Đồng bộ hóa trước khi nạp tile tiếp theo
        __syncthreads();
    }

    // Ghi kết quả vào ma trận đầu ra
    if (row < m && col < k) {
        result[row * k + col] = val;
    }
}
void matrixMultiplicationGPUWrapper(float* A, float *B, float *result, int m, int n, int k, int i, int version)
{	
    // Kích thước block và grid
    dim3 blockSize(32, 32);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Kích thước bộ nhớ
    const int size_A = m * n * sizeof(float);
    const int size_B = n * k * sizeof(float);
    const int size_result = m * k * sizeof(float);

    // Cấp phát bộ nhớ trên GPU
    float *d_A, *d_B, *d_result;
    CHECK(cudaMalloc(&d_A, size_A));
    CHECK(cudaMalloc(&d_B, size_B));
    CHECK(cudaMalloc(&d_result, size_result));

    // Copy dữ liệu từ CPU sang GPU
    CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    // Gọi kernel
    switch (version) {
        case 1: 
            matrixMultiplicationKernel_1<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k, i);
            break;
        case 2:
            matrixMultiplicationKernel_2<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k, i);
            break;
        case 3:
            matrixMultiplicationKernel_3<<<gridSize, blockSize>>>(d_A, d_B, d_result, m, n, k, i);
            break;
    }
    CHECK(cudaGetLastError());
    
    // Đồng bộ GPU để đảm bảo kernel hoàn thành
    CHECK(cudaDeviceSynchronize());

    // Copy kết quả từ GPU sang CPU
    CHECK(cudaMemcpy(result, d_result, size_result, cudaMemcpyDeviceToHost));

    // Giải phóng bộ nhớ GPU
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_result));
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_functions.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cstdlib>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

#define ROW 1024
#define COLUMN ROW
#define EPS 1e-5

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)


__global__ void SATwithGPU1(float* table, float* R, float* C, float* S)
{
    __shared__ float tmp[32][32];
    __shared__ float tmp_R[32];
    __shared__ float tmp_C[32];
    __shared__ float tmp_S;
    int column_index = blockIdx.x * 32 + threadIdx.x;
    int row_index = blockIdx.y * 32 + threadIdx.y;
    tmp[threadIdx.y][threadIdx.x] = table[row_index * COLUMN + column_index];
    int Block_index = blockIdx.x + blockIdx.y * 32;
    int Block_index_inverse = blockIdx.y + blockIdx.x * 32;
    __syncthreads();
    if (blockIdx.x < 31)
    {
        if (threadIdx.y == 1)
        {
            tmp_R[threadIdx.x] = 0;
            for (int i = 0; i < 32; i++)
            {
                tmp_R[threadIdx.x] += tmp[threadIdx.x][i];
            }
            R[Block_index_inverse * 32 + threadIdx.x] = tmp_R[threadIdx.x];
        }
    }
    if (blockIdx.y < 31)
    {
        if (threadIdx.y == 0)
        {
            tmp_C[threadIdx.x] = 0;
            for (int i = 0; i < 32; i++)
            {
                tmp_C[threadIdx.x] += tmp[i][threadIdx.x];
            }
            C[Block_index * 32 + threadIdx.x] = tmp_C[threadIdx.x];
        }
    }
    if (blockIdx.x < 31 && blockIdx.y < 31)
    {
        if(threadIdx.x == 0 && threadIdx.y == 0)
		{
            tmp_S = 0;
			for (int i = 0; i < 32; i++)
			{
                tmp_S += tmp_C[i];
			}
            S[Block_index] = tmp_S;
		}
    }
}

__global__ void SATwithGPU_ROW3(float* table, float* SAT, unsigned int row, unsigned int column)
{
    extern __shared__ float tmp[];
    int r = blockIdx.x;
    int c = threadIdx.x;
    int i = r * 2 * blockDim.x;
    int a = c;
    int b = c + column / 2;
    int bankOffsetA = CONFLICT_FREE_OFFSET(a);
    int bankOffsetB = CONFLICT_FREE_OFFSET(b);
    tmp[a + bankOffsetA] = table[i + a];
    tmp[b + bankOffsetB] = table[i + b];
    // reduction
    int offset = 1;
    for (int d = column >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (c < d)
        {
            int ai = offset * (2 * c + 1) - 1;
            int bi = offset * (2 * c + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
        offset <<= 1;
    }
    //reduction reverse
    offset = column / 2;
    for (int d = 2; d < column; d <<= 1)
    {
        __syncthreads();
        offset >>= 1;
        if (c < d - 1)
        {
            int ai = 2 * offset - 1 + c * 2 * offset;//2 * offset * (c + 1) - 1 = offset * (2 * c + 2) - 1 
            int bi = ai + offset; // offset * (2 * c + 3) - 1
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
    }
    __syncthreads();
    SAT[i + a] = tmp[a + bankOffsetA];
    SAT[i + b] = tmp[b + bankOffsetB];
}

__global__ void SATwithGPU_COLUMN3(float* table, float* SAT, unsigned int row, unsigned int column)
{
    extern __shared__ float tmp[];
    int r = blockIdx.x;
    int c = threadIdx.x;
    int a = c;
    int b = c + column / 2;
    int bankOffsetA = CONFLICT_FREE_OFFSET(a);
    int bankOffsetB = CONFLICT_FREE_OFFSET(b);
    tmp[a + bankOffsetA] = table[r + a * row];
    tmp[b + bankOffsetB] = table[r + b * row];
    // reduction
    int offset = 1;
    for (int d = column >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (c < d)
        {
            int ai = offset * (2 * c + 1) - 1;
            int bi = offset * (2 * c + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
        offset <<= 1;
    }
    //reduction reverse
    offset = column / 2;
    for (int d = 2; d < column; d <<= 1)
    {
        __syncthreads();
        offset >>= 1;
        if (c < d - 1)
        {
            int ai = 2 * offset - 1 + c * 2 * offset;
            int bi = ai + offset;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            tmp[bi] += tmp[ai];
        }
    }
    __syncthreads();
    SAT[r + a * row] = tmp[a + bankOffsetA];
    SAT[r + b * row] = tmp[b + bankOffsetB];
}

__global__ void SATwithGPU2(float* table, float* SAT, float* R, float* C, float* S)
{
    __shared__ float tmp[32][32];
    __shared__ float tmp_R[32][32];
    int column_index = blockIdx.x * 32 + threadIdx.x;
    int row_index = blockIdx.y * 32 + threadIdx.y;
    tmp[threadIdx.y][threadIdx.x] = table[row_index * COLUMN + column_index];
    int Block_index = blockIdx.x + blockIdx.y * 32;
    int Block_index_inverse = blockIdx.y + blockIdx.x * 32;
    if (threadIdx.x == 0 && blockIdx.x > 0 && threadIdx.y > 0)
    {
        tmp[threadIdx.y][0] += R[(Block_index_inverse - 32) * 32 + threadIdx.y];
    }
    if (threadIdx.y == 0 && blockIdx.y > 0 && threadIdx.x > 0)
    {
        tmp[0][threadIdx.x] += C[(Block_index - 32) * 32 + threadIdx.x];
    }
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        if (blockIdx.x > 0) tmp[0][0] += R[(Block_index_inverse - 32) * 32];
        if (blockIdx.y > 0) tmp[0][0] += C[(Block_index - 32) * 32];
        if (blockIdx.x > 0 && blockIdx.y > 0) tmp[0][0] += S[Block_index - 33];
    }
    __syncthreads();
    //tmp 做一次列方向的前缀和
    if (threadIdx.y == 0)
    {
        for (int j = 1; j < 32; j++)
        {
			tmp[j][threadIdx.x] += tmp[j - 1][threadIdx.x];
        }
    }
    __syncthreads();
    //转置
    tmp_R[threadIdx.y][threadIdx.x] = tmp[threadIdx.x][threadIdx.y];
    //__syncthreads();
    //tmp_R 做一次列方向的前缀和
    if (threadIdx.y == 0)
    {
        for (int j = 1; j < 32; j++)
        {
            tmp_R[j][threadIdx.x] += tmp_R[j - 1][threadIdx.x];
        }
    }
    __syncthreads();
    SAT[row_index * COLUMN + column_index] = tmp_R[threadIdx.x][threadIdx.y];
}

void SATwithCPU(float* table, float* SAT, unsigned int row, unsigned int column)
{
    for (int i = 0; i < row; i++)
    {
        SAT[i * column] = table[i * column];
        for (int j = 1; j < column; j++)
        {
            SAT[i * column + j] = SAT[i * column + j - 1] + table[i * column + j];
        }
    }
    for (int j = 0; j < column; j++)
    {
        for (int i = 1; i < row; i++)
        {
            SAT[i * column + j] = SAT[(i - 1) * column + j] + SAT[i * column + j];
        }
    }
}

bool verify(float* a, float* b, int* s, int* t)
{
    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COLUMN; j++)
        {
            if (fabs(a[i * COLUMN + j] - b[i * COLUMN + j]) > EPS * a[i * COLUMN + j])
            {
                *s = i;
                *t = j;
                return false;
            }
        }
    }
    return true;
}

int main()
{
    float* table = (float*)malloc(sizeof(float) * ROW * COLUMN);
    float* SAT = (float*)malloc(sizeof(float) * ROW * COLUMN);
    float* SAT_C = (float*)malloc(sizeof(float) * ROW * COLUMN);
    float* R = (float*)malloc(sizeof(float) * ROW * 32);
    float* C = (float*)malloc(sizeof(float) * COLUMN * 32);
    float* S = (float*)malloc(sizeof(float) * 32 * 32);
    float* dev_table;
    float* dev_SAT;
    float* dev_R;
    float* dev_C;
    float* dev_S;
    cudaMalloc((void**)&dev_table, sizeof(float) * ROW * COLUMN);
    cudaMalloc((void**)&dev_SAT, sizeof(float) * ROW * COLUMN);
    cudaMalloc((void**)&dev_R, sizeof(float) * ROW * 32);
    cudaMalloc((void**)&dev_C, sizeof(float) * COLUMN * 32);
    cudaMalloc((void**)&dev_S, sizeof(float) * 32 * 32);

    cudaMemset(dev_R, 0, sizeof(float) * ROW * 32);
    cudaMemset(dev_C, 0, sizeof(float) * COLUMN * 32);
    cudaMemset(dev_S, 0, sizeof(float) * 32 * 32);

    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);

    srand(static_cast <unsigned> (time(0)));
    for (int i = 0; i < ROW; i++)
    {
        for (int j = 0; j < COLUMN; j++)
        {
            table[i + j * COLUMN] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 256));;
        }
    }

    cudaMemcpy(dev_table, table, sizeof(float) * ROW * COLUMN, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_SAT, SAT, sizeof(float) * ROW * COLUMN, cudaMemcpyHostToDevice);

    float esp_time;
    clock_t start, stop;
    start = clock();
    for (int i = 0; i < 100; i++)
    {
        SATwithCPU(table, SAT_C, ROW, COLUMN);
    }
    stop = clock();
    esp_time = (float)(stop - start) / CLOCKS_PER_SEC;
    printf("CPU Timecost: %f\n", esp_time);

    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    float ms;
    //warmup
    SATwithGPU1 << <dimGrid, dimBlock >> > (dev_table, dev_R, dev_C, dev_S);
    SATwithGPU_COLUMN3 << <1024, 16, 32 * sizeof(float) >> > (dev_R, dev_R, 1024, 32);
    SATwithGPU_COLUMN3 << <1024, 16, 32 * sizeof(float) >> > (dev_C, dev_C, 1024, 32);
    SATwithGPU_ROW3 << <32, 16, 16* sizeof(float) >> > (dev_S, dev_S, 32, 32);
    SATwithGPU_COLUMN3 << <32, 16, 16 * sizeof(float) >> > (dev_S, dev_S, 32, 32);
    SATwithGPU2 << <dimGrid, dimBlock >> > (dev_table, dev_SAT, dev_R, dev_C, dev_S);

    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < 100; i++)
    {
        SATwithGPU1 << <dimGrid, dimBlock >> > (dev_table, dev_R, dev_C, dev_S);
        SATwithGPU_COLUMN3 << <1024, 16, 32 * sizeof(float) >> > (dev_R, dev_R, 1024, 32);
        SATwithGPU_COLUMN3 << <1024, 16, 32 * sizeof(float) >> > (dev_C, dev_C, 1024, 32);
        SATwithGPU_ROW3 << <32, 16, 16 * sizeof(float) >> > (dev_S, dev_S, 32, 32);
        SATwithGPU_COLUMN3 << <32, 16, 16 * sizeof(float) >> > (dev_S, dev_S, 32, 32);
        SATwithGPU2 << <dimGrid, dimBlock >> > (dev_table, dev_SAT, dev_R, dev_C, dev_S);
    }

    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("GPU Timecost: %f\n", ms / 1000);

    cudaDeviceSynchronize();
    cudaMemcpy(SAT, dev_SAT, sizeof(float) * ROW * COLUMN, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(S, dev_S, sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(R, dev_R, sizeof(float) * ROW * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, dev_C, sizeof(float) * COLUMN * 32, cudaMemcpyDeviceToHost);

    /*for (int i = 63; i < 68; i++)
    {
        for (int j = 28; j <= 32; j++)
        {
			printf("%.2f ", SAT[i * COLUMN + j]);
		}
        printf("\n");
    }
    printf("\n");
    for (int i = 63; i < 68; i++)
    {
        for (int j = 28; j <= 32; j++)
        {
            printf("%.2f ", SAT_C[i * COLUMN + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("%.2f", C[64]);
    printf("\n");
    printf("%.2f ", R[COLUMN + 63]);
    printf("\n");
    printf("%.2f ", S[32]);
    printf("\n");*/

    printf("*****************************************\n");
    printf("%.2f\n", SAT[ROW * COLUMN - 1]);
    printf("%.2f\n", SAT_C[ROW * COLUMN - 1]);

    int i, j;
    if (verify(SAT_C, SAT, &i, &j))
    {
        printf("YES\n");
    }
    else
    {
        printf("NO\n");
        printf("%d %d\n", i, j);
        printf("%.2f\n", SAT[i * COLUMN + j]);
        printf("%.2f\n", SAT_C[i * COLUMN + j]);
    }

    cudaFree(dev_table);
    cudaFree(dev_SAT);
    cudaFree(dev_R);
    cudaFree(dev_C);
    cudaFree(dev_S);
    free(SAT);
    free(SAT_C);
    free(table);
    free(R);
    free(C);
    free(S);
    return 0;
}
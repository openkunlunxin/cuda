#include <stdio.h>
#include <cuda.h>
#include <chrono>

// 最大 constant memory 数据量（64KB）
#define CONST_DATA_MAX (64 * 1024 / sizeof(float))

// constant memory 声明
__constant__ float constData[CONST_DATA_MAX];

// 测试 kernel
__global__ void testConstCache(float* out, int dataSize, int iterations) {
    int idx = threadIdx.x; // 只用一个 block
    float temp = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        // 访问 constant memory，循环绕 dataSize
        temp += constData[(idx + i) % dataSize];
    }

    out[idx] = temp;
}

int main() {
    const int THREADS_PER_BLOCK = 512; // 可调，保证访问压力
    const int BLOCKS = 1;             // 只用一个 block
    const int ITER = 100000;          // 放大循环次数

    // 主机端准备 constant memory 数据
    float h_constData[CONST_DATA_MAX];
    for (int i = 0; i < CONST_DATA_MAX; ++i) h_constData[i] = (float)i;

    // 拷贝到 constant memory
    cudaMemcpyToSymbol(constData, h_constData, sizeof(float) * CONST_DATA_MAX);

    // 输出数据
    float* h_out = new float[THREADS_PER_BLOCK];
    float* d_out;
    cudaMalloc(&d_out, THREADS_PER_BLOCK * sizeof(float));

    printf("dataSize(bytes)\ttime(ms)\n");

    // 测试不同 dataSize
    for (int dataSize = 32; dataSize <= CONST_DATA_MAX; dataSize += 128) {
        // 计时 CUDA event
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        testConstCache<<<BLOCKS, THREADS_PER_BLOCK>>>(d_out, dataSize, ITER);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("%d\t%.3f\n", dataSize * (int)sizeof(float), ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 取回数据（可选，避免编译优化）
    cudaMemcpy(h_out, d_out, THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    delete[] h_out;

    return 0;
}

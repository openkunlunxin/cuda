#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define WARMUP 10000
#define ITERATIONS 100000

// 创建随机的指针追逐模式，防止预取
__global__ void createChasePattern(int* pattern, int size, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    // 创建伪随机但确定性的访问模式
    unsigned int idx = tid;
    for (int i = 0; i < 10; i++) {
        idx = (idx * 1103515245 + 12345) % size;
    }
    pattern[tid] = idx;
}

__global__ void pointerChaseLatency(int* data, int start_idx, float* result) {
    // 只用一个线程测量
    if (threadIdx.x + blockIdx.x * blockDim.x != 0) return;
    
    int idx = start_idx;
    unsigned long long start, end;
    float total_cycles = 0.0f;
    
    // 预热
    for (int i = 0; i < WARMUP; i++) {
        idx = data[idx];
    }
    
    // 正式测量
    for (int iter = 0; iter < ITERATIONS; iter++) {
        start = clock64();
        
        // 多次访问取平均
        for (int j = 0; j < 10; j++) {
            idx = data[idx];
        }
        
        end = clock64();
        total_cycles += (end - start);
        
        // 防止优化
        if (idx == -1) break;
    }
    
    result[0] = total_cycles / (ITERATIONS * 10.0f);
}

void measureMemoryLatency() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1000.0f / 1000.0f);
    
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    
    // 测试不同缓存级别
    int test_sizes[] = {
        12 * 1024,      // L1缓存大小 (约)
        48 * 1024,      // L1 + 部分L2
        64 * 1024,
	96 * 1024,
	128 * 1024,
	160 * 1024,
	192 * 1024,
	208 * 1024,
	224 * 1024,
	248 * 1024,
	256 * 1024,     // L2缓存 (部分)
        2 * 1024 * 1024, // 超过L2
        16 * 1024 * 1024,	// 全局内存
	32 * 1024 * 1024,
	48 * 1024 * 1024,
	56 * 1024 * 1024,
	64 * 1024 * 1024,
	72 * 1024 * 1024,
	80 * 1024 * 1024,
	96 * 1024 * 1024,
	112 * 1024 * 1024,
	128 * 1024 * 1024,
	144 * 1024 * 1024,
	160 * 1024 * 1024,
	192 * 1024 * 1024,
	256 * 1024 * 1024,
	512 * 1024 * 1024

    };
    
    for (int size_kb : test_sizes) {
        int elements = size_kb / sizeof(int);
        int* d_data;
        cudaMalloc(&d_data, elements * sizeof(int));
        
        // 创建指针追逐模式
        createChasePattern<<<(elements + 255)/256, 256>>>(d_data, elements, 42);
        
        // 测量延迟
        pointerChaseLatency<<<1, 1>>>(d_data, 0, d_result);
        
        float cycles;
        cudaMemcpy(&cycles, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        // 转换为纳秒
        float latency_ns = cycles / (prop.clockRate * 1000.0f) * 1e9f;
        
        printf("Size: %6d KB - Latency: %6.2f cycles, %6.2f ns\n", 
               size_kb/1024, cycles, latency_ns);
        
        cudaFree(d_data);
    }
    
    cudaFree(d_result);
}

int main(){
	measureMemoryLatency();
	return 0;
}


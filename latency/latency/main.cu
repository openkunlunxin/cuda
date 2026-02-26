#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define N 1000000
#define WARMUP 10000000
#define ITERATIONS 10000000

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

// Fisher-Yates洗牌算法 - 原地版本
void fisherYatesShuffle(int *pattern, int size) {
    std::vector<int> arr(size);
    
    // 初始化顺序数组
    for (int i = 0; i < size; i++) {
        arr[i] = i;
    }
    
    // 使用高质量随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 从后向前洗牌
    for (int k = 0; k<10; k++){
    for (int i = size-1; i > 0; i--) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        std::swap(arr[i], arr[j]);
    }
    }


    for (int i=0; i<size; i++){
	int next_idx = (i+1)%size;
	pattern[arr[i]] = arr[next_idx];
    }

    int real_nums=0;
    int real_size=0;
    int idx_hash[size]={0};
    int idx = 0;
    for(int i=0; i<ITERATIONS; i++){
        idx_hash[idx] += 1;
        idx = pattern[idx];
    }
    for (int i=0; i<size; i++){
        if(idx_hash[i] != 0) real_nums += 1;
    }

    real_size = real_nums * sizeof(int);

    printf("SIZE: %6f\tKB REAL SIZE: %6f\t KB ", size*sizeof(int)/1024.0, real_size/1024.0);
}

void createChasePattern_host(int *pattern, int size, int seed) {
	int real_nums=0;
	int real_size=0;
	int idx_hash[size]={0};
	for (int i=0; i<size; i++) {
		unsigned int idx = i;
		for (int j = 0; j<seed; j++){
			idx = (idx+31415926) % size;
		}
		pattern[i] = idx;
		//idx_hash[idx] += 1;
		//printf("%d,",idx);
	}
	int idx = 0;
	for(int i=0; i<ITERATIONS; i++){
		idx_hash[idx] += 1;
		idx = pattern[idx];
	}
	for (int i=0; i<size; i++){
		if(idx_hash[i] != 0) real_nums += 1;
	}

	real_size = real_nums * sizeof(int);

        printf("SIZE: %6f\tKB REAL SIZE: %6f\t KB ", size*sizeof(int)/1024.0, real_size/1024.0);
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
    #pragma unroll 1
    for (int iter = 0; iter < ITERATIONS; iter++) {
        start = clock64();
        
        // 多次访问取平均
	#pragma unroll 1
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
    // int test_sizes[] = {
    //   64 * 1024
    //};

    // int test_sizes2[] =  * 1024};
    
    //for (int size_kb : test_sizes) {
    for (int size_kb=2048; size_kb<=2*1024*1024; size_kb+=2048){
        int elements = size_kb / sizeof(int);
	int* d_data; 
	int  h_data[elements];
	
	cudaMalloc(&d_data, elements * sizeof(int));
        
	fisherYatesShuffle(h_data, elements);
	
	// copy
	cudaMemcpy(d_data, &h_data, elements*sizeof(int), cudaMemcpyHostToDevice);
	
        // 测量延迟
        pointerChaseLatency<<<1, 1>>>(d_data, 0, d_result);
        
        float cycles;
        cudaMemcpy(&cycles, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        // 转换为纳秒
        float latency_ns = cycles / (prop.clockRate * 1000.0f) * 1e9f;
        
        //printf("Size: %6f KB - Latency: %6.2f cycles, %6.2f ns\n", 
        //       size_kb/1024.0, cycles, latency_ns);
	printf("%6f, %6.2f\n",size_kb/1024.0, cycles);
        cudaFree(d_data); 
    }
    cudaFree(d_result);
}

int main(){
	measureMemoryLatency();
	return 0;
}


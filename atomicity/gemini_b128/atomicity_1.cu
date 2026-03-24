#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>

#define PATTERN_LOW  0xDEADBEEF
#define PATTERN_HIGH 0xCAFEBABE
#define NUM_THREADS  1024 // 32个Warp

struct DetectionResult {
    unsigned long long torn_read_count;
    uint32_t sample[4];
};

__global__ void prove_b128_non_atomic_improved(DetectionResult* res) {
    // 强制 16 字节对齐
    __shared__ __align__(16) uint32_t smem_addr0[4];
    // 额外申请一块空间制造 Bank 干扰
    __shared__ __align__(16) uint32_t smem_noise[32]; 

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // 获取地址
    uint32_t ptr_addr0;
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; cvt.u32.u64 %0, %p; }" : "=r"(ptr_addr0) : "l"(&smem_addr0[0]));
    
    // 初始化
    if (tid == 0) {
        for(int i=0; i<4; i++) smem_addr0[i] = 0;
    }
    __syncthreads();

    // --- Warp 0: Writer (高频翻转) ---
    if (warp_id == 0) {
        if (threadIdx.x == 0) { // 仅一个线程写，模拟单指令 b128
            uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
            for (int i = 0; i < 5000000; ++i) {
                // 写新值
                asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                              : : "r"(ptr_addr0), "r"(l), "r"(l), "r"(h), "r"(h));
                // 极短延迟
                asm volatile (""); 
                // 写旧值
                asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                              : : "r"(ptr_addr0), "r"(z), "r"(z), "r"(z), "r"(z));
            }
        }
    }
    // --- Warp 1-31: Readers (高压读) ---
    else {
        uint32_t r0, r1, r2, r3;
        uint32_t dummy; 
        for (int i = 0; i < 1000000; ++i) {
            // 1. 制造 Bank 噪声：读取同 Bank 的不同偏移，试图拉长访问窗口
            dummy = smem_noise[tid % 32]; 

            // 2. 核心读取：使用 .volatile 强制访问物理存储
            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_addr0) : "memory"
            );

            // 3. 撕裂检测 (Torn Read)
            // 如果 b128 是原子的，不应该出现 (低位是新 && 高位是旧) 或 (低位是旧 && 高位是新)
            bool low_is_new = (r0 == PATTERN_LOW);
            bool high_is_new = (r2 == PATTERN_HIGH);
            bool low_is_old = (r0 == 0);
            bool high_is_old = (r2 == 0);

            if ((low_is_new && high_is_old) || (low_is_old && high_is_new)) {
                atomicAdd(&res->torn_read_count, 1);
                // 仅保存第一个样本
                if (res->sample[0] == 0) {
                    res->sample[0] = r0; res->sample[1] = r1;
                    res->sample[2] = r2; res->sample[3] = r3;
                }
            }
            
            // 防止编译器把 dummy 优化掉
            if(dummy == 0x1) asm volatile(""); 
        }
    }
}

int main() {
    DetectionResult *d_res, h_res;
    cudaMalloc(&d_res, sizeof(DetectionResult));
    cudaMemset(d_res, 0, sizeof(DetectionResult));

    std::cout << "Starting H100 b128 Atomicity Stress Test..." << std::endl;
    prove_b128_non_atomic_improved<<<1, NUM_THREADS>>>(d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(DetectionResult), cudaMemcpyDeviceToHost);

    std::cout << "\nResults:" << std::endl;
    std::cout << "Torn Reads Detected: " << h_res.torn_read_count << std::endl;

    if (h_res.torn_read_count > 0) {
        std::cout << "SUCCESS: Non-atomicity of b128 PROVED." << std::endl;
        printf("Sample: [0x%08X 0x%08X 0x%08X 0x%08X]\n", 
               h_res.sample[3], h_res.sample[2], h_res.sample[1], h_res.sample[0]);
    } else {
        std::cout << "FAILED: No torn reads. Hardware was too fast or alignment made it too stable." << std::endl;
    }

    cudaFree(d_res);
    return 0;
}

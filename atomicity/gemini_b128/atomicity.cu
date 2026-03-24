#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <iostream>

#define PATTERN_LOW  0xDEADBEEF
#define PATTERN_HIGH 0xCAFEBABE

struct DetectionResult {
    unsigned long long torn_read_count;
    uint32_t sample[4];
};

__global__ void prove_b128_non_atomic_final(DetectionResult* res) {
    // 目标地址：16字节对齐
    __shared__ __align__(16) uint32_t smem_target[4];
    // 干扰区：映射到相同的 Banks (CUDA中每4字节循环一个Bank，共32个Bank)
    __shared__ __align__(16) uint32_t smem_noise[1024];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    uint32_t ptr_target;
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; cvt.u32.u64 %0, %p; }" : "=r"(ptr_target) : "l"(&smem_target[0]));

    // 初始化
    if (tid == 0) {
        for(int i=0; i<4; i++) smem_target[i] = 0;
    }
    __syncthreads();

    // --- Warp 0: Writer (核心变量翻转者) ---
    if (warp_id == 0 && lane_id == 0) {
        uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
        for (int i = 0; i < 100000000; ++i) { // 增加到1亿次翻转
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(ptr_target), "r"(l), "r"(l), "r"(h), "r"(h));
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(ptr_target), "r"(z), "r"(z), "r"(z), "r"(z));
        }
    }
    // --- Warp 1: Noise Maker (制造物理Bank拥堵) ---
    else if (warp_id == 1) {
        uint32_t noise_val = tid;
        for (int i = 0; i < 5000000; ++i) {
            // 访问与 smem_target 相同的 Bank 索引 (index % 32 == 0)
            int target_bank_idx = (lane_id * 32) % 1024; 
            // 混合读写：使用原子操作强制 Bank 锁定或产生长时间流水线停顿
            atomicXor(&smem_noise[target_bank_idx], noise_val);
            // 紧接着一个读访问
            noise_val ^= smem_noise[(target_bank_idx + 1) % 1024];
        }
    }
    // --- Warp 2-31: High-Speed Readers (高速采样扫描) ---
    else {
        uint32_t r0, r1, r2, r3;
        for (int i = 0; i < 2000000; ++i) {
            // 在读取前加入极其微小的随机扰动，打破 Warp 间的同步性
            if ((lane_id & 3) == 0) asm volatile("nanosleep.u32 10;");

            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_target) : "memory"
            );

            // 撕裂检测逻辑：
            // 如果读取不是原子的，可能读到：[低位新, 高位旧] 或 [低位旧, 高位新]
            bool low_is_new = (r0 == PATTERN_LOW);
            bool high_is_new = (r2 == PATTERN_HIGH);
            bool low_is_old = (r0 == 0);
            bool high_is_old = (r2 == 0);

            if ((low_is_new && high_is_old) || (low_is_old && high_is_new)) {
                atomicAdd(&res->torn_read_count, 1);
                // 仅保存第一次捕获的结果
                if (res->sample[0] == 0) {
                    res->sample[0] = r0; res->sample[1] = r1;
                    res->sample[2] = r2; res->sample[3] = r3;
                }
            }
        }
    }
}


int main() {
    DetectionResult *d_res, h_res;
    cudaMalloc(&d_res, sizeof(DetectionResult));
    cudaMemset(d_res, 0, sizeof(DetectionResult));

    std::cout << "H100 b128 Atomicity Stress Test (High Interference Mode)..." << std::endl;
    
    // 启动 1024 个线程（32 个 Warp）
     prove_b128_non_atomic_final <<<1, 1024>>>(d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(DetectionResult), cudaMemcpyDeviceToHost);

    std::cout << "\n--- Result Report ---\n";
    std::cout << "Torn Reads Found: " << h_res.torn_read_count << std::endl;

    if (h_res.torn_read_count > 0) {
        std::cout << "STATUS: [PROVED] b128 is NOT atomic." << std::endl;
        printf("Torn Data: [H: 0x%08X %08X | L: 0x%08X %08X]\n", 
               h_res.sample[3], h_res.sample[2], h_res.sample[1], h_res.sample[0]);
    } else {
        std::cout << "STATUS: [FAILED] No torn reads detected." << std::endl;
        std::cout << "Note: H100 hardware might have extremely strong write-combining for aligned b128." << std::endl;
    }

    cudaFree(d_res);
    return 0;
}

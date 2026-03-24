#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>

#define PATTERN_LOW  0xDEADBEEF
#define PATTERN_HIGH 0xCAFEBABE

struct DetectionResult {
    unsigned long long torn_read_count;
    uint32_t sample[4];
};

__global__ void prove_b128_non_atomic_extreme(DetectionResult* res) {
    // 16 字节对齐的测试目标
    __shared__ __align__(16) uint32_t smem_addr0[4];
    // 干扰区，用于制造 Bank 压力
    __shared__ __align__(16) uint32_t smem_noise[1024];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    uint32_t ptr_addr0;
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; cvt.u32.u64 %0, %p; }" : "=r"(ptr_addr0) : "l"(&smem_addr0[0]));

    // 初始化
    if (tid == 0) {
        smem_addr0[0] = 0; smem_addr0[1] = 0; smem_addr0[2] = 0; smem_addr0[3] = 0;
    }
    __syncthreads();

    // --- Warp 0: Writer (极限频率翻转) ---
    if (warp_id == 0 && lane_id == 0) {
        uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
        // 增加循环次数到 5000 万次
        for (int i = 0; i < 50000000; ++i) {
            // 写入全新值 (b128)
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_addr0), "r"(l), "r"(l), "r"(h), "r"(h));
            // 写入全旧值 (0)
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_addr0), "r"(z), "r"(z), "r"(z), "r"(z));
        }
    }
    // --- Warp 1-31: Readers (制造 Bank 冲突泥沼) ---
    else if (warp_id > 0) {
        uint32_t r0, r1, r2, r3;
        
        for (int i = 0; i < 2000000; ++i) {
            // 关键改进：增加随机抖动，让各线程的采样点在时间轴上错开
            // 避免所有 Reader 都在同一时刻发起请求
            if (lane_id % 4 == 0) {
                asm volatile ("nanosleep.u32 20;"); 
            }

            // 核心读取：使用 .volatile 强制物理访存
            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_addr0) : "memory"
            );

            // 撕裂检测：检查是否存在“半新半旧”的状态
            // 逻辑：如果读到了低位的 DEADBEEF，但高位还是 0；或者高位是 CAFEBABE，低位是 0。
            bool has_new = (r0 == PATTERN_LOW || r2 == PATTERN_HIGH);
            bool has_old = (r0 == 0 || r2 == 0);
            
            // 排除掉纯新（全新）和纯旧（全旧）的情况
            bool is_torn = has_new && has_old;

            if (is_torn) {
                atomicAdd(&res->torn_read_count, 1);
                // 仅保存第一个捕获到的样本用于验证
                if (res->sample[0] == 0) {
                    res->sample[0] = r0; res->sample[1] = r1;
                    res->sample[2] = r2; res->sample[3] = r3;
                }
            }

            // 制造 Bank 噪声干扰，增加控制器负担
            smem_noise[tid] = r0 ^ i; 
        }
    }
}

int main() {
    DetectionResult *d_res, h_res;
    cudaMalloc(&d_res, sizeof(DetectionResult));
    cudaMemset(d_res, 0, sizeof(DetectionResult));

    std::cout << "H100 b128 Atomicity Stress Test (High Interference Mode)..." << std::endl;
    
    // 启动 1024 个线程（32 个 Warp）
    prove_b128_non_atomic_extreme<<<1, 1024>>>(d_res);
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

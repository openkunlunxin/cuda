#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>

#define PATTERN_LOW  0xDEADBEEF
#define PATTERN_HIGH 0xCAFEBABE
#define NUM_THREADS  1024 // 32 个 Warp

// 用于保存检测结果的结构体
struct DetectionResult {
    unsigned long long torn_v4_count;
    uint32_t sample_v4[4];
    unsigned long long torn_u64_count;
    uint32_t sample_u64[4];
};

__global__ void prove_b128_atomicity_comparison(DetectionResult* res) {
    // 强制 16 字节对齐的测试目标 A 和 B
    __shared__ __align__(16) uint32_t smem_target_v4[4];
    __shared__ __align__(16) uint32_t smem_target_u64[4];
    
    // 干扰区，与目标地址映射到相同的 Banks
    __shared__ __align__(16) uint32_t smem_noise[NUM_THREADS];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 获取通用指针
    uint32_t ptr_v4, ptr_u64;
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; cvt.u32.u64 %0, %p; }" : "=r"(ptr_v4) : "l"(&smem_target_v4[0]));
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; cvt.u32.u64 %0, %p; }" : "=r"(ptr_u64) : "l"(&smem_target_u64[0]));
    
    // 初始化
    if (tid == 0) {
        for(int i=0; i<4; i++) { smem_target_v4[i] = 0; smem_target_u64[i] = 0; }
    }
    __syncthreads();

    // 模式翻转循环次数
    const int flip_loops = 50000000;
    // 采样读取循环次数
    const int read_loops = 1000000;

    // --- 角色分配 ---

    // Warp 0:情况 A 的 Writer (单条 st.shared.v4.u32)
    if (warp_id == 0 && lane_id == 0) {
        uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
        for (int i = 0; i < flip_loops; ++i) {
            // 情况 A：单指令写入 128 位
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_v4), "r"(l), "r"(l), "r"(h), "r"(h));
            // 写入全 0
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_v4), "r"(z), "r"(z), "r"(z), "r"(z));
        }
    }
    // Warp 1: 情况 B 的 Writer (双条 st.shared.u64)
    else if (warp_id == 1 && lane_id == 0) {
        uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
        // 显式拆分数据
        uint64_t new_val_l = ((uint64_t)l << 32) | l;
        uint64_t new_val_h = ((uint64_t)h << 32) | h;
        uint64_t old_val = 0;

        for (int i = 0; i < flip_loops; ++i) {
            // 情况 B：双指令写入 128 位
            // 注意： Little-Endian，先写低 64 位
            asm volatile ("st.shared.u64 [%0], %1;" :: "r"(ptr_u64), "l"(new_val_l));
            // 这里故意不加 fence，看 Reader 能不能插队
            asm volatile ("st.shared.u64 [%0+8], %1;" :: "r"(ptr_u64), "l"(new_val_h));
            
            // 写入全 0
            asm volatile ("st.shared.u64 [%0], %1;" :: "r"(ptr_u64), "l"(old_val));
            asm volatile ("st.shared.u64 [%0+8], %1;" :: "r"(ptr_u64), "l"(old_val));
        }
    }
    // Warp 2: 情况 B 的独立 Noise Maker (制造 Bank 拥堵)
    else if (warp_id == 2) {
        uint32_t noise_val = tid;
        for (int i = 0; i < flip_loops / 2; ++i) {
            // 混合读写干扰，增加 Bank Arbiter 负担
            int bank_idx = (lane_id * 32) % NUM_THREADS; 
            atomicXor(&smem_noise[bank_idx], noise_val);
            noise_val ^= smem_noise[(bank_idx + 1) % NUM_THREADS];
        }
    }
    // Warp 3-17: 情况 A 的 Readers (高速采样 v4)
    else if (warp_id >= 3 && warp_id <= 17) {
        uint32_t r0, r1, r2, r3;
        for (int i = 0; i < read_loops; ++i) {
            // 在读取前加入极其微小的随机扰动
            if ((lane_id & 3) == 0) asm volatile("nanosleep.u32 10;");

            // 情况 A 读取：使用 .volatile 强制物理访存
            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_v4) : "memory"
            );

            // 撕裂检测：高位是新的，低位是旧的
            if (r0 == 0 && r2 == PATTERN_HIGH) {
                atomicAdd(&res->torn_v4_count, 1);
                if (res->sample_v4[0] == 0) {
                    res->sample_v4[0] = r0; res->sample_v4[1] = r1;
                    res->sample_v4[2] = r2; res->sample_v4[3] = r3;
                }
            }
        }
    }
    // Warp 18-31: 情况 B 的 Readers (高速采样 u64)
    else if (warp_id >= 18) {
        uint32_t r0, r1, r2, r3;
        for (int i = 0; i < read_loops; ++i) {
            if ((lane_id & 3) == 0) asm volatile("nanosleep.u32 10;");

            // 情况 B 读取
            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_u64) : "memory"
            );

            // 撕裂检测
            if (r0 == 0 && r2 == PATTERN_HIGH) {
                atomicAdd(&res->torn_u64_count, 1);
                if (res->sample_u64[0] == 0) {
                    res->sample_u64[0] = r0; res->sample_u64[1] = r1;
                    res->sample_u64[2] = r2; res->sample_u64[3] = r3;
                }
            }
        }
    }
}

int main() {
    DetectionResult *d_res, h_res;
    cudaMalloc(&d_res, sizeof(DetectionResult));
    cudaMemset(d_res, 0, sizeof(DetectionResult));

    std::cout << "Starting H100 Atomicity Comparison: Aligned Single v4 vs. Double u64..." << std::endl;
    std::cout << "This will run for several seconds." << std::endl;
    
    // 启动 1024 个线程
    prove_b128_atomicity_comparison<<<1, NUM_THREADS>>>(d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(DetectionResult), cudaMemcpyDeviceToHost);

    std::cout << "\n--- Comparison Report ---\n";
    std::cout << "[Aligned Case A: Aligned Single st.shared.v4.u32]\n";
    std::cout << "  Torn Reads Found: " << h_res.torn_v4_count << std::endl;
    if (h_res.torn_v4_count > 0) {
        printf("  Sample: [H: 0x%08X %08X | L: 0x%08X %08X]\n", 
               h_res.sample_v4[3], h_res.sample_v4[2], h_res.sample_v4[1], h_res.sample_v4[0]);
    } else {
        std::cout << "  STATUS: Likely Atomic (Single Transaction).\n";
    }

    std::cout << "\n[Aligned Case B: Aligned Double st.shared.u64]\n";
    std::cout << "  Torn Reads Found: " << h_res.torn_u64_count << std::endl;
    if (h_res.torn_u64_count > 0) {
        printf("  Sample: [H: 0x%08X %08X | L: 0x%08X %08X]\n", 
               h_res.sample_u64[3], h_res.sample_u64[2], h_res.sample_u64[1], h_res.sample_u64[0]);
    } else {
        std::cout << "  STATUS: Surprise! Still not tearing. Hardware is merging writes very strongly.\n";
    }

    std::cout << "\nAnalysis:\n";
    if (h_res.torn_v4_count == 0 && h_res.torn_u64_count > 0) {
        std::cout << "STATUS: PROVED. Single v4 instruction provides偽原子性protection, which dual u64 does not.\n";
    } else if (h_res.torn_v4_count == 0 && h_res.torn_u64_count == 0) {
        std::cout << "STATUS: NOT PROVED. H100 hardware is too fast or write-combining is too robust to tear even dual writes.\n";
    } else if (h_res.torn_v4_count > 0) {
        std::cout << "STATUS: FAILED. Even single v4 instruction is tearing. Non-atomicity confirmed.\n";
    }

    cudaFree(d_res);
    return 0;
}

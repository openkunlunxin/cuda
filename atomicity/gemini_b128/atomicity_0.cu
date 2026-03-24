#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

// 定义数据模式
#define PATTERN_LOW  0xDEADBEEF
#define PATTERN_HIGH 0xCAFEBABE

// 用于保存检测结果的结构体
struct DetectionResult {
    uint32_t torn_read_count;
    uint32_t raw_data[4]; // 保存捕获到的撕裂数据用于验证
};

// --- CUDA Kernel ---
// 为了精确控制，我们强制单 Block 运行，通过 warp ID 分配角色
__global__ void prove_b128_non_atomic_kernel(DetectionResult* final_result) {
    // 声明全局对齐的共享内存 addr0
    // 使用 __align__(16) 确保 b128/v4.u32 存取对齐要求
    __shared__ __align__(16) uint32_t smem_addr0[4];

    // 初始化 Shared Mem 为 0 (旧数据)
    if (threadIdx.x == 0) {
        smem_addr0[0] = 0; smem_addr0[1] = 0; smem_addr0[2] = 0; smem_addr0[3] = 0;
    }
    __syncthreads(); // 确保初始化完成

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 获取 smem_addr0 的 32 位通用指针（PTX 使用）
    uint32_t smem_ptr;
    asm volatile("{ .reg .b64 %%desc_ptr; cvta.shared.u64 %%desc_ptr, %1; cvt.u32.u64 %0, %%desc_ptr; }" : "=r"(smem_ptr) : "l"(&smem_addr0[0]));

    // --- 角色分配 ---

    // Warp 0: Writer - 负责不断写入新数据和旧数据
    if (warp_id == 0) {
        // 只有 lane 0 执行存储，代表 Warp 级别的 b128 存取
        if (lane_id == 0) {
            uint32_t new_val0 = PATTERN_LOW;
            uint32_t new_val1 = PATTERN_LOW;
            uint32_t new_val2 = PATTERN_HIGH;
            uint32_t new_val3 = PATTERN_HIGH;

            uint32_t old_val = 0;

            // 循环写入，制造撕裂机会
            for (int i = 0; i < 1000000; ++i) {
                // 1. 写入新数据模式 (v4.u32)
                asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                              : : "r"(smem_ptr), "r"(new_val0), "r"(new_val1), "r"(new_val2), "r"(new_val3));
                
                // 插入极短延迟或内存屏障，让旧数据有机会刷入，增加 Reader 捕获复杂度
                asm volatile ("membar.cta;"); 

                // 2. 写入旧数据模式 (全 0)
                asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                              : : "r"(smem_ptr), "r"(old_val), "r"(old_val), "r"(old_val), "r"(old_val));
                
                asm volatile ("membar.cta;");
            }
        }
    }

    // Warp 1 & Warp 2: Readers - 负责高频读取并检测撕裂
    else if (warp_id == 1 || warp_id == 2) {
        // 所有的线程都参与读取，制造大量的 Bank Conflict 和流量，拉长 tearing window
        uint32_t r0, r1, r2, r3;
        uint32_t local_torn_count = 0;
        uint32_t captured_torn_data[4] = {0, 0, 0, 0};

        // 高频读取循环
        for (int i = 0; i < 2000000; ++i) {
            // 向量化读取 (v4.u32)
            asm volatile ("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                          : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr));

            // --- 检测逻辑 ---
            // 原子性意味着要么全旧(0)，要么全新(PATTERN_LOW/HIGH组合)
            // 撕裂读意味着：低位是新的，高位是旧的（或者相反，取决于硬件拆分顺序，Little-Endian通常先低后高）
            
            bool is_old = (r0 == 0 && r1 == 0 && r2 == 0 && r3 == 0);
            bool is_new = (r0 == PATTERN_LOW && r1 == PATTERN_LOW && r2 == PATTERN_HIGH && r3 == PATTERN_HIGH);

            if (!is_old && !is_new) {
                // 检测到非原子性的中间状态！
                local_torn_count++;
                
                // 保存第一次捕获到的撕裂数据用于 Host 端展示
                if (local_torn_count == 1) {
                    captured_torn_data[0] = r0; captured_torn_data[1] = r1;
                    captured_torn_data[2] = r2; captured_torn_data[3] = r3;
                }
            }
            // 读循环中不需要显式 Barrier，我们希望越乱越好
        }

        // 将本线程的检测结果累加到全局
        if (local_torn_count > 0) {
            atomicAdd(&(final_result->torn_read_count), local_torn_count);
            
            // 如果我是这个 Warp 中第一个发现撕裂的，负责把数据样本写回 Global
            if (local_torn_count >= 1 && lane_id == __ffs(unsigned(atomicAnd(&local_torn_count, 0xFFFFFFFF))) - 1) {
                 // 简单的原子操作确保只写一次样本（这里简化了逻辑，实际中可能需要更严谨的同步）
                 if (atomicCAS(&(final_result->raw_data[0]), 0, captured_torn_data[0]) == 0) {
                     final_result->raw_data[1] = captured_torn_data[1];
                     final_result->raw_data[2] = captured_torn_data[2];
                     final_result->raw_data[3] = captured_torn_data[3];
                 }
            }
        }
    }
}

int main() {
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Testing on Device: " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;

    // 分配 Global Memory 用于接收结果
    DetectionResult *d_result, h_result;
    cudaMalloc(&d_result, sizeof(DetectionResult));
    cudaMemset(d_result, 0, sizeof(DetectionResult));

    // 启动 Kernel
    // 使用单 Block，96个线程 (3个 Warp: 1 Writer, 2 Readers)
    std::cout << "Launching Kernel. This might take a few seconds..." << std::endl;
    prove_b128_non_atomic_kernel<<<1, 96>>>(d_result);
    
    // 等待完成
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 拷贝结果回 Host
    cudaMemcpy(&h_result, d_result, sizeof(DetectionResult), cudaMemcpyDeviceToHost);

    // --- 分析结果 ---
    std::cout << "\n--- Test Analysis ---\n";
    std::cout << "Total Torn Reads Detected: " << h_result.torn_read_count << std::endl;

    if (h_result.torn_read_count > 0) {
        std::cout << "STATUS: PROVED! b128 shared memory access is NOT atomic.\n";
        std::cout << "Captured Torn Data Sample (hex):\n";
        std::cout << "  [Word 3 (High)]: 0x" << std::hex << std::setw(8) << std::setfill('0') << h_result.raw_data[3] << "\n";
        std::cout << "  [Word 2]:        0x" << std::hex << std::setw(8) << std::setfill('0') << h_result.raw_data[2] << "\n";
        std::cout << "  [Word 1]:        0x" << std::hex << std::setw(8) << std::setfill('0') << h_result.raw_data[1] << "\n";
        std::cout << "  [Word 0 (Low)]:  0x" << std::hex << std::setw(8) << std::setfill('0') << h_result.raw_data[0] << std::dec << "\n";
        
        // 验证捕获到的数据是否符合撕裂特征
        // 特征：数据的四个Word中，有的属于新模式，有的属于旧模式（0）
        uint32_t raw0 = h_result.raw_data[0];
        uint32_t raw2 = h_result.raw_data[2];
        if ((raw0 == PATTERN_LOW && raw2 == 0) || (raw0 == 0 && raw2 == PATTERN_HIGH)) {
             std::cout << "Verification: Data splitting confirmed (Partial New, Partial Old).\n";
        }
    } else {
        std::cout << "STATUS: NOT PROVED. No torn reads detected in this run.\n";
        std::cout << "Suggestions: Increase loop counts in kernel, or run the test multiple times.\n";
    }

    // 清理
    cudaFree(d_result);
    return 0;
}

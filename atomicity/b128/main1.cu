#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <vector>

// 定义 128KB 共享内存的大小（以 uint32_t 为单位）
#define SMEM_SIZE_BYTES (128 * 1024)
#define SMEM_U32_COUNT (SMEM_SIZE_BYTES / sizeof(uint32_t))

__global__ void hopper_asymmetric_smem_kernel(uint32_t* global_out) {
    // 使用动态共享内存
    extern __shared__ uint32_t s_data[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 获取 addr0 的 32位共享内存指针（PTX 使用）
    uint32_t smem_ptr_addr0 = __cvta_generic_to_shared(&s_data[0]);

    // --- 初始化：协作式清零 128KB ---
    for (int i = tid; i < SMEM_U32_COUNT; i += blockDim.x) {
        s_data[i] = 0;
    }
    __syncthreads(); // 确保初始化完成

    // --- Warp 0: 频繁对 128KB 进行 st.shared.b64 ---
    if (warp_id == 0) {
        unsigned long long* s_ptr_64 = (unsigned long long*)s_data;
        // 避开前 16 字节 (addr0)，防止干扰 Warp 1 的数据写入测试
        #pragma unroll 4
        for (int i = 0; i < 1000; ++i) {
            uint32_t idx = 2 + ((lane_id + i * 32) % (SMEM_U32_COUNT / 2 - 2));
            s_ptr_64[idx] = 0xDEADC0DEDEADC0DEULL;
        }
    }


    // --- Warp 2: 循环读取 (Polling) addr0，使用 ld.shared.v4.u32 ---
    else if (warp_id == 2) {
        if (lane_id == 0) {
            uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            bool found = true;
            int count = 0;

            while (found) {
                // 执行向量化读取
                asm volatile (
                    "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr_addr0) : "memory"
                );

                // 检查是否读到了 Warp 1 写入的第一个特征数据 0x11111111
                if (r1 == r0*2 && r2==r0*3 && r3==r0*4) {
                    found = true;
                }

		count +=1;

		if(count == 10000) {
			found = false;
		}

                // 避免死循环过于消耗硬件资源，可以加极短延迟
                __nanosleep(1); 
            }

	    __threadfence();

            // 写回到全局内存验证
            global_out[0] = r0;
            global_out[1] = r1;
            global_out[2] = r2;
            global_out[3] = r3;
        }
    }
    // --- Warp 1: 对 addr0 执行向量化 st.shared.v4.u32 (b128) ---
    else if (warp_id == 1) {
        // 为了演示效果，延迟写入
        for(int k=0; k<5; k++) { __nanosleep(10); }

        if (lane_id == 0) {
            uint32_t val0 = 0x11111111;
            uint32_t val1 = 0x22222222;
            uint32_t val2 = 0x33333333;
            uint32_t val3 = 0x44444444;

            // 内存屏障确保之前的操作不会被重排
            //__threadfence_block();
            for (int i=0; i<1000; i++){
            	asm volatile (
                	"st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                	: : "r"(smem_ptr_addr0), "r"(val0), "r"(val1), "r"(val2), "r"(val3) : "memory"
            	);
		val0 += 4;
		val1 += 8;
		val2 += 12;
		val3 += 16;
	    }
        }
    }
}

int main() {
    // 1. 设置设备并获取动态共享内存限制
    int device = 0;
    cudaSetDevice(device);

    int max_smem_per_block;
    cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    std::cout << "Max shared memory per block: " << max_smem_per_block / 1024 << " KB" << std::endl;

    // 2. 配置 Kernel 使用 128KB 动态共享内存
    int required_smem = 128 * 1024;
    cudaFuncSetAttribute(hopper_asymmetric_smem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required_smem);

    // 3. 分配 Global Memory
    uint32_t *d_out, h_out[4];
    cudaMalloc(&d_out, 4 * sizeof(uint32_t));
    cudaMemset(d_out, 0, 4 * sizeof(uint32_t));

    // 4. 启动 Kernel (1个 Block, 3个 Warp = 96 线程)
    std::cout << "Launching kernel with 3 warps and 128KB SMEM..." << std::endl;
    hopper_asymmetric_smem_kernel<<<1, 96, required_smem>>>(d_out);

    // 5. 检查结果
    cudaMemcpy(h_out, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "Warp 2 polled results:" << std::endl;
    bool success = true;
    uint32_t expected[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};

    for (int i = 0; i < 4; ++i) {
        printf("  [Word %d]: 0x%08X ", i, h_out[i]);
        if (h_out[i] == h_out[0] * (i+1)) printf("(Correct)\n");
        else { printf("(Wrong! Expected 0x%08X)\n", expected[i]); success = false; }
    }

    if (success) std::cout << "SUCCESS: Warp 2 correctly detected Warp 1's store!" << std::endl;
    else std::cout << "FAILED." << std::endl;

    cudaFree(d_out);
    return 0;
}

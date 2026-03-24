#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// 48KB Shared Memory
#define SMEM_SIZE (48 * 1024)
#define SMEM_UINT32_COUNT (SMEM_SIZE / sizeof(uint32_t))

// 检查 CUDA 错误的宏
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                             \
            exit(1);                                                     \
        }                                                                \
    } while (0)

__global__ void hopper_v4_verify_kernel(uint32_t* global_out) {
    // 静态申请 48KB Shared Memory
    __shared__ uint32_t s_data[SMEM_UINT32_COUNT];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 获取 addr0 的共享内存 32 位地址寄存器
    uint32_t smem_ptr_32 = __cvta_generic_to_shared(&s_data[0]);

    // --- Warp 0: 频繁写入 48KB 的其余部分 ---
    if (warp_id == 0) {
        // 从索引 4 开始写 (避开前 16 字节的 addr0)，防止随机覆盖导致验证失败
        for (int i = 0; i < 500; ++i) {
            uint32_t idx = 4 + ((lane_id + i * 32) % (SMEM_UINT32_COUNT - 4));
            s_data[idx] = 0xDEADC0DE;
        }
    }

    // --- Warp 1: 对 addr0 执行 st.shared.v4.u32 (128-bit) ---
    else if (warp_id == 1) {
        if (lane_id == 0) {
            uint32_t src0 = 0x11111111;
            uint32_t src1 = 0x22222222;
            uint32_t src2 = 0x33333333;
            uint32_t src3 = 0x44444444;

            asm volatile (
                "st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                : : "r"(smem_ptr_32), "r"(src0), "r"(src1), "r"(src2), "r"(src3) : "memory"
            );
        }
    }

    // --- Warp 2: 对 addr0 执行 ld.shared.v4.u32 (128-bit) ---
    else if (warp_id == 2) {
        // 粗略同步，确保 Warp 1 写入完成
        __threadfence_block();
        __nanosleep(500); 

        if (lane_id == 0) {
            uint32_t r0, r1, r2, r3;
            do{
	   	asm volatile (
                	"ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                	: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_ptr_32)
            	);
	    }while(r1 != 0x22222222);

            // 写回到全局内存
            global_out[0] = r0;
            global_out[1] = r1;
            global_out[2] = r2;
            global_out[3] = r3;
        }
    }
    __syncthreads();	
}

int main() {
    const int num_elements = 4;
    uint32_t h_out[num_elements] = {0};
    uint32_t* d_out;

    printf("Starting H100 Warp Communication Kernel...\n");

    // 1. 分配内存
    CHECK_CUDA(cudaMalloc(&d_out, num_elements * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_out, 0, num_elements * sizeof(uint32_t)));

    // 2. 启动 Kernel (1个 Block, 96个线程 = 3个 Warp)
    // H100 上默认支持 48KB 静态 Shared Memory，无需额外配置
    hopper_v4_verify_kernel<<<1, 96>>>(d_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3. 拷贝结果回 Host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 4. 验证数据
    uint32_t expected[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
    bool success = true;

    printf("Results from Warp 2 (ld.shared.v4.u32):\n");
    for (int i = 0; i < 4; i++) {
        printf("  Offset %d (bytes %2d-%2d): 0x%08X ", i, i*4, (i+1)*4-1, h_out[i]);
        if (h_out[i] == expected[i]) {
            printf("[OK]\n");
        } else {
            printf("[FAIL] (Expected 0x%08X)\n", expected[i]);
            success = false;
        }
    }

    if (success) {
        printf("\nVerification SUCCESS: Warp 2 correctly read 128-bit data from Warp 1.\n");
    } else {
        printf("\nVerification FAILED.\n");
    }

    // 5. 清理
    cudaFree(d_out);
    return 0;
}

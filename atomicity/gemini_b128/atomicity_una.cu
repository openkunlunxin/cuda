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

__global__ void prove_unaligned_b128_tearing(DetectionResult* res) {
    // 申请 32 字节空间，但我们故意不从起始位置写
    __shared__ __align__(16) uint32_t smem_raw[8]; 
    
    // 关键点：将指针偏移 4 字节（1个 uint32），使其 4 字节对齐但非 16 字节对齐
    uint32_t ptr_unaligned;
    asm volatile("{ .reg .b64 %p; cvta.shared.u64 %p, %1; add.u64 %p, %p, 4; cvt.u32.u64 %0, %p; }" 
                 : "=r"(ptr_unaligned) : "l"(&smem_raw[0]));

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 初始化
    if (tid == 0) {
        for(int i=0; i<8; i++) smem_raw[i] = 0;
    }
    __syncthreads();

    // --- Warp 0: Writer (非对齐写入) ---
    if (warp_id == 0 && lane_id == 0) {
        uint32_t l = PATTERN_LOW, h = PATTERN_HIGH, z = 0;
        for (int i = 0; i < 10000000; ++i) {
            // st.shared.v4 执行非对齐写入，硬件必拆分
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_unaligned), "r"(l), "r"(l), "r"(h), "r"(h));
            asm volatile ("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" 
                          :: "r"(ptr_unaligned), "r"(z), "r"(z), "r"(z), "r"(z));
        }
    }
    // --- Warp 1-31: Readers (高频采样) ---
    else {
        uint32_t r0, r1, r2, r3;
        for (int i = 0; i < 1000000; ++i) {
            // 同样执行非对齐读取
            asm volatile (
                "ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" 
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(ptr_unaligned) : "memory"
            );

            // 撕裂检测
            bool has_new = (r0 == PATTERN_LOW || r2 == PATTERN_HIGH);
            bool has_old = (r0 == 0 || r2 == 0);

            if (has_new && has_old) {
                atomicAdd(&res->torn_read_count, 1);
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

    std::cout << "Running UNALIGNED b128 Tearing Test..." << std::endl;
    prove_unaligned_b128_tearing<<<1, 1024>>>(d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(DetectionResult), cudaMemcpyDeviceToHost);

    std::cout << "\nResults for Unaligned Access:" << std::endl;
    std::cout << "Torn Reads Found: " << h_res.torn_read_count << std::endl;

    if (h_res.torn_read_count > 0) {
        std::cout << "STATUS: [CONFIRMED] Unaligned b128 is heavily torn." << std::endl;
        printf("Example Sample: [0x%08X 0x%08X 0x%08X 0x%08X]\n", 
               h_res.sample[3], h_res.sample[2], h_res.sample[1], h_res.sample[0]);
    } else {
        std::cout << "STATUS: [UNEXPECTED] Still no tearing. H100 alignment handling is incredibly robust." << std::endl;
    }

    cudaFree(d_res);
    return 0;
}

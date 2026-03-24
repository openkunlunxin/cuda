#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>

__device__ __forceinline__ void my_pipeline_memcpy_async_cg(
    void* __restrict__ shared_ptr, 
    const void* __restrict__ global_ptr, 
    size_t size) {
    
    // 将通用指针转换为共享内存的 32 位偏移地址
    unsigned smem_addr = __cvta_generic_to_shared(shared_ptr);

    // 根据 size 生成对应的 cp.async.cg 指令
    // 注意：size 必须是编译时常量或 4, 8, 16 之一
    if (size == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" 
                     : : "r"(smem_addr), "l"(global_ptr));
    } else if (size == 8) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" 
                     : : "r"(smem_addr), "l"(global_ptr));
    } else if (size == 4) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 4;\n" 
                     : : "r"(smem_addr), "l"(global_ptr));
    }
}

__global__ void race_condition_kernel(int* global_addr, int* result_out) {
    // 1. 声明 Shared Memory
    __shared__ int s_data[4];
    
    // for(int i=0; i<100000; i++){
    // 2. 第一次 Store: 将 global_addr 设为 1
    // 使用 atomicExch 或 volatile 确保写入不被编译器优化掉
    //   global_addr[i] = 1;
    //}
    //__syncthreads();
    //__threadfence();
    //__threadfence();

    int s_data1;

    s_data1 = global_addr[3];

    // 3. 发射异步拷贝: Global -> Shared
    // 我们想看看这一步捕获的是 1 还是 2
    //__pipeline_memcpy_async(&s_data, &global_addr[0], sizeof(int));
    my_pipeline_memcpy_async_cg(&s_data, &global_addr[0], 4*sizeof(int));
    
    //__pipeline_commit();
    
    // for(int i=0; i<100000; i++){
    // 4. 第二次 Store: 立即将 global_addr 修改为 2
    //    global_addr[99999-i] = 2;
    //}

    // 5. 等待异步拷贝完成
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    int s_data0;

    s_data0 = global_addr[1];


    // 6. 将 Shared Memory 的结果写回到输出显存以便 Host 查看
    *result_out = s_data[0]+s_data0+s_data1;
}

int test() {
    int N = 1000000;
    int h_addr[1000000] = {0};
    int *d_addr, *d_result;
    int h_result = 0;
    
    // 分配显存
    cudaMalloc(&d_addr, N*sizeof(int));
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_addr, &h_addr, N*sizeof(int), cudaMemcpyHostToDevice);
    // 启动 Kernel (1个 Block, 1个 Thread 即可观察)
    race_condition_kernel<<<1, 1>>>(d_addr, d_result);

    // 拷贝结果回 Host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << h_result << std::endl;
    //std::cout << "--------------------------------------" << std::endl;
    //std::cout << "Shared Memory captured value: " << h_result << std::endl;
    if (h_result == 1) {
    //    std::cout << "Result: Captured the INITIAL store (1)." << std::endl;
    } else {
    //    std::cout << "Result: Captured the SECOND store (2) - Race condition occurred!" << std::endl;
    }
    //std::cout << "--------------------------------------" << std::endl;

    // 这里的 d_addr 最终一定是 2
    int h_final_global = 0;
    cudaMemcpy(&h_final_global, d_addr, sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout << "Final Global Memory value: " << h_final_global << std::endl;

    cudaFree(d_addr);
    cudaFree(d_result);

    return 0;
}

int main() {
	for(int i=0; i<1; i++){
		test();
	}
}

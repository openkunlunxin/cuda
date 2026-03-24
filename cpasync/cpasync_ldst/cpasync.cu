#include <cuda_runtime.h>

#include <cuda_pipeline.h>
#include <iostream>

__global__ void race_condition_kernel(int* global_addr, int* result_out) {
    // 1. 声明 Shared Memory
    __shared__ int s_data;
    
    for(int i=0; i<100000; i++){
    // 2. 第一次 Store: 将 global_addr 设为 1
    // 使用 atomicExch 或 volatile 确保写入不被编译器优化掉
       global_addr[i] = 1;
    }
    //__syncthreads();
    //__threadfence();
    //__threadfence();

    // 3. 发射异步拷贝: Global -> Shared
    // 我们想看看这一步捕获的是 1 还是 2
    __pipeline_memcpy_async(&s_data, &global_addr[99999], sizeof(int));
    //__pipeline_commit();
    
    for(int i=0; i<100000; i++){
    // 4. 第二次 Store: 立即将 global_addr 修改为 2
        global_addr[99999-i] = 2;
    }

    // 5. 等待异步拷贝完成
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();


    // 6. 将 Shared Memory 的结果写回到输出显存以便 Host 查看
    *result_out = s_data;
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
	for(int i=0; i<50; i++){
		test();
	}
}

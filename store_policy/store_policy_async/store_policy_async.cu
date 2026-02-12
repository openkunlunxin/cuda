#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/pipeline>

#define  N 4

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// device kernel: use global load and store
__global__ void device_memcpy_kernel(int* dst, int* src) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        // global load from src, global store to dst
        // int dst0 = src[idx] + 1;
        // int dst1 = src[idx+1] + 1 + 1;
        // int dst2 = src[idx+2] + 1 + 2;
        // int dst3 = src[idx+3] + 1 + 3;
        // __stcg(&dst[idx+0], dst0);
        // __stcg(&dst[idx+1], dst1);
        // __stcg(&dst[idx+2], dst2);
        // __stcg(&dst[idx+3], dst3);
        dst[idx+0] = src[idx+0] + 1    ;
        dst[idx+1] = src[idx+1] + 1 + 1;
        dst[idx+2] = src[idx+2] + 1 + 2;
        dst[idx+3] = src[idx+3] + 1 + 3;
    }


    //===============================================TMA+1===============================================//
    static constexpr size_t buf_len = N;
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operations should be 16 byte aligned.
    __shared__ alignas(16) int smem_data[buf_len];

    // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
    //    b) Make initialized barrier visible in async proxy.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) { 
        init(&bar, blockDim.x);                      // a)
        ptx::fence_proxy_async(ptx::space_shared);   // b)
    }
    __syncthreads();

    // 2. Initiate TMA transfer to copy global to shared memory.
    if (threadIdx.x == 1) {
        // 3a. cuda::memcpy_async arrives on the barrier and communicates
        //     how many bytes are expected to come in (the transaction count)
        cuda::memcpy_async(
            smem_data, 
            dst, 
            cuda::aligned_size_t<16>(sizeof(smem_data)),
            bar
        );
    }
    // 3b. All threads arrive on the barrier
    barrier::arrival_token token = bar.arrive();
    
    // 3c. Wait for the data to have arrived.
    bar.wait(std::move(token));

    // 4. Compute saxpy and write back to shared memory
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
        smem_data[i] += 1;
    }

    // 5. Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async(ptx::space_shared);   // b)
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // 6. Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        ptx::cp_async_bulk(
            ptx::space_global,
            ptx::space_shared,
            src, smem_data, sizeof(smem_data));
        // 7. Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    } 
    //=========================================================TMA END ==================================================// 
    // __threadfence();
        //__syncthreads();
        

}


int main() {
    int *h_src = nullptr, *h_dst = nullptr;
    int *d_src = nullptr, *d_dst = nullptr;

    // host memory alloc
    h_src = (int*)malloc(N * sizeof(int));
    h_dst = (int*)malloc(N * sizeof(int));

    for (size_t i = 0; i < N; ++i){
        h_src[i] = static_cast<int>(i+11);
        h_dst[i] = static_cast<int>(0);
    }

    // device memory alloc
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(int)));

    for (size_t i = 0; i < N; ++i) {
        std::cout <<"B:\t"<< h_src[i] << " \t:\t " << h_dst[i] << std::endl;
    }

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    device_memcpy_kernel<<<blocks, threads>>>(d_dst, d_src);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_src, d_src, N * sizeof(int), cudaMemcpyDeviceToHost));

    // cout
    for (size_t i = 0; i < N; ++i) {
        std::cout <<"A:\t"<< h_src[i] << " \t:\t " << h_dst[i] << std::endl;
    }
    
    // verify
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_src[i] != h_dst[i] + 1) {
            std::cerr << "Mismatch at " << i << ": " << h_src[i] << " != " << h_dst[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "Device-to-device memcpy (global load/store) success!" << std::endl;

    // clean up
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return 0;
}

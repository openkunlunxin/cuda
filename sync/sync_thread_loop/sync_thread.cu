#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/pipeline>

#define  N 65536

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


__device__ int flag = 12138;

// device kernel: use global load and store
// __global__ void device_memcpy_kernel(int *pcie_addr) {
// //__global__ void w_atk_r(volatile int *pcie_addr) {
//     __shared__ volatile int sync;
//     int tid = threadIdx.x;
//     int tmp = 0;
//     if(tid == 511) {
//         tmp = pcie_addr[0];
//         tmp += 0xdeadbeef;
//         sync = tmp;
//     } else {
//         while(1) {
//             pcie_addr[1] == 1;
//             if(sync == 0xdeadbeef) {
//                 break;
//             }
//         }
//     }
// }

__global__ void device_memcpy_kernel(int *pcie_addr) {
//__global__ void w_atk_r(volatile int *pcie_addr) {
    __shared__ volatile int sync;
    int tid = threadIdx.x;
    int tmp = 0;
    if(tid == 0) {
        tmp = pcie_addr[0];
        tmp += 0xdeadbeef;
        sync = tmp;
    } else {
        while(1) {
            pcie_addr[1] = 1;
            if(sync == 0xdeadbeef) {
                break;
            }
        }
    }
}


int main() {
    int *h_src = nullptr, *h_dst = nullptr;
    int *d_src = nullptr, *d_dst = nullptr;
    int *pcie_addr ;
    CHECK_CUDA(cudaMalloc(&pcie_addr, N*sizeof(int)));

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

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(pcie_addr, d_dst, N * sizeof(int), cudaMemcpyHostToDevice));
    
    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    device_memcpy_kernel<<<blocks, threads>>>(pcie_addr);

    // CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_src, d_src, N * sizeof(int), cudaMemcpyDeviceToHost));

    // clean up
    free(h_src);
    free(h_dst);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return 0;
}

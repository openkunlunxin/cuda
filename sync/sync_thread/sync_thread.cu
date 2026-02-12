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


__device__ int flag = 12138;

// device kernel: use global load and store
__global__ void device_memcpy_kernel(int* dst, int* src) {
    size_t idx = threadIdx.x;

    if(idx < 16){
        if(idx%32==0) printf("tid 0-15 start\n");
        if(idx%32==0) printf("tid 0-15 middle\n");
        __syncthreads();
        int flag_tmp = flag;//__ldca(&flag) ;
        while (flag_tmp == 12138){
            if(idx%32==0) printf("tid 0-15 middle \t%d\n", flag_tmp);
            flag_tmp = flag;//__ldca(&flag) ;
            //if(idx%32==0) printf("tid 0-15 middle \t%d\n", flag_tmp);
        };
        if(idx%32==0) printf("tid 0-15 end\n");
    } else if(idx < 32) {
        if(idx%32==16) printf("tid 16-32 start\n");
        if(idx%32==16) printf("tid 16-32 middle\n");
        if(idx%32==16) printf("tid 16-32 end\n");
    } else if(idx < 64) {
        if(idx%32==0) printf("tid 32-63 start \n");
        flag = 12139;
        __syncthreads();
        if(idx%32==0) printf("tid 32-63 middle\n");
        if(idx%32==0) printf("tid 32-63 end\n");
    }
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

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel: device-to-device copy
    int threads = 64;
    int blocks = (N + threads - 1) / threads;
    device_memcpy_kernel<<<blocks, threads>>>(d_dst, d_src);

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

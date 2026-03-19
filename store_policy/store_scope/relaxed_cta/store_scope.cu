#include <iostream>
#include <cuda_runtime.h>
#include "./cuda_scope_store.h"

#define UNROLL_SIZE 16
#define VEC_SIZE    4
#define BLOCK_SIZE  256
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
__global__ void device_memst_kernel(int* dst, int* src, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
		int tmp0 = 11;
		/*
		cuda_scope::store_cta(&dst[idx+0 ], tmp0);
		cuda_scope::store_cluster(&dst[idx+32], tmp0);
		cuda_scope::store_gpu(&dst[idx+1 ], tmp0);
		cuda_scope::store_sys(&dst[idx+2 ], tmp0);
		*/
		cuda_scope::store_cta_relaxed(&dst[idx+0 ], tmp0);
		cuda_scope::store_cta_relaxed(&dst[idx+32], tmp0);
		cuda_scope::store_cta_relaxed(&dst[idx+2 ], tmp0);
		cuda_scope::store_cta_relaxed(&dst[idx+3 ], tmp0);
		
		__syncthreads();
                int src0 = dst[idx] + 1;
                int src1 = dst[idx+1] + 1 + 1;
                int src2 = dst[idx+2] + 1 + 2;
                int src3 = dst[idx+3] + 1 + 3;
		
		dst[idx + 64] = src0 + src1 + src2 + src3;	

		// dst[idx+64] = tmp0;
		// dst[idx+96] = tmp0;
		// dst[idx+128] = tmp0;
		// dst[idx+160] = tmp0;
		// dst[idx+192] = tmp0;
		// dst[idx+224] = tmp0;
		// dst[idx+256] = tmp0;
		// dst[idx+1] = tmp0;
		// dst[idx+288] = tmp0;
		// dst[idx+320] = tmp0;
		// dst[idx+352] = tmp0;
		// dst[idx+384] = tmp0;
		// dst[idx+416] = tmp0;
		// dst[idx+448] = tmp0;
		// dst[idx+480] = tmp0;
		// dst[idx+512] = tmp0;
		// dst[idx+2] = tmp0;
    }
}

int main() {
    const size_t N = 256*4;
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

    for (size_t i = 0; i < 10; ++i) {
        std::cout <<"B:\t"<< h_src[i] << " \t:\t " << h_dst[i] << std::endl;
    }

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (threads + threads - 1) / threads;
    device_memst_kernel<<<blocks, threads>>>(d_dst, d_src, N);
	// device_memst_1block_kernel<<<blocks, threads>>>(d_dst, d_src, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_src, d_src, N * sizeof(int), cudaMemcpyDeviceToHost));

    // cout
    for (size_t i = 0; i < 10; ++i) {
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

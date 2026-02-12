#include <iostream>
#include <cuda_runtime.h>

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
__global__ void device_memcpy_kernel(int* dst, int* src, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && idx%32 < 8) {
        // global load from src, global store to dst
        dst[idx/32*8 + idx%32] = src[idx/32*8 + idx%32] + 1;
	    // __syncthreads();
	    // src[idx] = dst[idx] + 1;
    }
}


int main() {
    const size_t N = 1 << 8; // 1M ints, about 4MB
    int *h_src = nullptr, *h_dst = nullptr;
    int *d_src = nullptr, *d_dst = nullptr;

    // host memory alloc
    h_src = (int*)malloc(N * sizeof(int));
    h_dst = (int*)malloc(N * sizeof(int));

    for (size_t i = 0; i < N; ++i)
        h_src[i] = static_cast<int>(i);

    // device memory alloc
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(int)));

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    device_memcpy_kernel<<<blocks, threads>>>(d_dst, d_src, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_src, d_src, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    // verify
    bool correct = true;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << h_src[i] << " \t:\t " << h_dst[i] << std::endl;
    }

    for (size_t i = 0; i < N; ++i) {
        if (h_src[i] + 1 != h_dst[i]) {
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

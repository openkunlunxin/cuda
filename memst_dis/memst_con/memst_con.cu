#include <iostream>
#include <cuda_runtime.h>

#define UNROLL_SIZE 16
#define VEC_SIZE    4
constexpr int vec_size = 4;
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t)*vec_size) aligned_vector {
    scalar_t val[vec_size];
};

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
__global__ void device_memst_kernel(int* dst, const int* src, const int n) {
    int   thrx = blockIdx.x * blockDim.x + threadIdx.x;
    using vec_t = aligned_vector<int, vec_size>;
	const vec_t *x_vec = reinterpret_cast<const vec_t*>(src);
          vec_t *y_vec = reinterpret_cast<vec_t*>(dst);

    int threads_needed = n/UNROLL_SIZE;

    // if( thrx < threads_needed){
    //     #pragma unroll
    //     for (int i = 0; i < UNROLL_SIZE/VEC_SIZE; i ++) {
    //         y_vec[thrx + blockDim.x * gridDim.x * i] = tmp;
    //     }
    // }

	if( thrx < threads_needed){
        #pragma unroll
        for (int i = 0; i < UNROLL_SIZE/VEC_SIZE; i ++) {
            y_vec[thrx + blockDim.x * gridDim.x * i].val[0] = x_vec[thrx + blockDim.x * gridDim.x * i].val[0];
			y_vec[thrx + blockDim.x * gridDim.x * i].val[1] = x_vec[thrx + blockDim.x * gridDim.x * i].val[1];
			y_vec[thrx + blockDim.x * gridDim.x * i].val[2] = x_vec[thrx + blockDim.x * gridDim.x * i].val[2];
			y_vec[thrx + blockDim.x * gridDim.x * i].val[3] = x_vec[thrx + blockDim.x * gridDim.x * i].val[3];
        }
    }

}


int main() {
    const int N = 1<<4;
    int *h_src = nullptr, *h_dst = nullptr;
    int *d_src = nullptr, *d_dst = nullptr;

    // host memory alloc
    h_src = (int*)malloc(N * sizeof(int));
    h_dst = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; ++i)
        h_src[i] = static_cast<int>(i);

    // device memory alloc
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(int)));

    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (N/UNROLL_SIZE + threads - 1) / threads;
    device_memst_kernel<<<blocks, threads>>>(d_dst, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(int), cudaMemcpyDeviceToHost));

    // verify
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (h_src[i]%4 != h_dst[i]) {
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
#include <iostream>
#include <cuda_runtime.h>

#define VEC_SIZE    4
constexpr int vec_size = VEC_SIZE;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t)*vec_size) aligned_vector {
    scalar_t val[vec_size];
};

// device kernel: use global load and store
template<typename T>
__global__ void device_memcpy_kernel(T* dst, T* src, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	using vec_t  = aligned_vector<T, vec_size>;
	vec_t *x_vec = reinterpret_cast<vec_t*>(src);
    vec_t *y_vec = reinterpret_cast<vec_t*>(dst);

    if (idx == 0) {
		y_vec[0].val[0] = x_vec[0].val[0];
		y_vec[0].val[1] = x_vec[0].val[1];
		__syncthreads();
		y_vec[0].val[2] = x_vec[0].val[2];
		y_vec[0].val[3] = x_vec[0].val[3];
		// y_vec[0]        = x_vec[0]       ;
		// y_vec[1].val[0] = x_vec[1].val[0];
		// __syncthreads();
		x_vec[0].val[0] = y_vec[0].val[0];
    }
}

template<typename T>
int test() {
    const size_t N = 1 << 12; // 1M ints, about 4MB
    T *h_src = nullptr, *h_dst = nullptr;
    T *d_src = nullptr, *d_dst = nullptr;

    // host memory alloc
    h_src = (T*)malloc(N * sizeof(T));
    h_dst = (T*)malloc(N * sizeof(T));

    for (size_t i = 0; i < N; ++i)
        h_src[i] = static_cast<T>(i);

    // device memory alloc
    CHECK_CUDA(cudaMalloc(&d_src, N * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_dst, N * sizeof(T)));

    // host2device: init src
    CHECK_CUDA(cudaMemcpy(d_src, h_src, N * sizeof(T), cudaMemcpyHostToDevice));

    // launch kernel: device-to-device copy
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    device_memcpy_kernel<T><<<blocks, threads>>>(d_dst, d_src, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // device2host: copy result back to host
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, N * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_src, d_src, N * sizeof(T), cudaMemcpyDeviceToHost));
    
    // verify
    bool correct = true;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << int(h_src[i]) << " \t:\t " << int(h_dst[i]) << std::endl;
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

int main(){
	test<char>();
}
#include <stdio.h>
#include <cuda_runtime.h>

#define VALUE 33

// 定义模板 kernel
template <typename T>
__global__ void AtomicAddKernel(T* data, T* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        T old = atomicAdd(&data[0], (T)VALUE, (T)VALUE);
	out[tid] = old;
    }
}

int main() {
    int N = 32;

    using T  =  unsigned int;
    T h_data[N], h_out[N];
    T* d_data, *d_out;

    for(int i=0; i<N; i++){
	h_data [i] = 0;
    }

    cudaMalloc((void**)&d_data, N * sizeof(T));
    cudaMalloc((void**)&d_out, N * sizeof(T));
    cudaMemcpy(d_data, &h_data, N*sizeof(T), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    AtomicAddKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_data, d_out, N);

    cudaMemcpy(h_out, d_out, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data, d_data, N * sizeof(T), cudaMemcpyDeviceToHost);

    printf("Final value: %f\n", (double)h_data[0]);
    printf("Out array:\n");
    for (int i = 0; i < 32; ++i) {
        printf("out[%d] = %f\t", i, (double)h_out[i]);
    }
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_out);
    return 0;
}

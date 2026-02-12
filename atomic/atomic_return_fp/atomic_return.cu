#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// 定义模板 kernel
template <typename T>
__global__ void AtomicAddKernel(T* data, T* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // atomicAdd 返回操作前的旧值。不同类型调用不同 atomicAdd
        T old = atomicAdd(&data[0], (T)0.1);
        out[tid] = old;
    }
    __syncthreads();
    if (tid == 0){
	    T 
	    T old = atomicAdd(

}

void printFloatBits(float f) {
	uint32_t bits = *reinterpret_cast<uint32_t*>(&f); // 取得float的原始比特
	for (int i = 31; i >= 0; --i) {
	        std::cout << ((bits >> i) & 1);
	        if (i % 8 == 0 && i != 0) std::cout << ' '; // 可选：每8位加空格				    
	}
	std::cout << std::endl;
}

int atomic_test() {
    int N = 32;

    // 以 float 为例，也可以改为 int/double/unsigned int等
    using T  = float; // 修改这里可以切换类型
    T  h_data[N], h_out[N];
    T* d_data, *d_out;
    T  h_data_cmp[N];

    for(int i=0; i<N; i++){
	h_data [i] = 0;
	h_data_cmp[i] = 0;
    }

    cudaMalloc((void**)&d_data, N * sizeof(T));
    cudaMalloc((void**)&d_out, N * sizeof(T));
    cudaMemcpy(d_data, &h_data, N*sizeof(T), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动模板 kernel，类型为 T
    AtomicAddKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_data, d_out, N);
    
    for (int i=0; i<N/32; i++){
    	h_data_cmp[0] += (float)0.1 * 32;
    }
    cudaMemcpy(h_out, d_out, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_data, d_data, N * sizeof(T), cudaMemcpyDeviceToHost);

    //printf("Final value: %f\n", (double)h_data[0]);
    printFloatBits(h_data[0]);
    printFloatBits(h_data_cmp[0]);


    printf("Out array:\n");
    for (int i = 0; i < 32; ++i) {
        printf("out[%d] = %f\t", i, (double)h_out[i]);
    }
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_out);
    return 0;
}

int main(){
	atomic_test();
}


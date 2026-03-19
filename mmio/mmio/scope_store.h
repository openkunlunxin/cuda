// cuda_scoped_store.h
#pragma once

// #ifdef __CUDA_ARCH__

namespace cuda_scope {

enum class scope { cta, cluster, gpu, sys };

// 主模板函数
template<typename T>
__device__ void store_cta(T* addr, T val) {  
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.release.cta.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.release.cta.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.release.cta.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.release.cta.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_cluster(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.release.cluster.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.release.cluster.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.release.cluster.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.release.cluster.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_gpu(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.release.gpu.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.release.gpu.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.release.gpu.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.release.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_sys(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.release.sys.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.release.sys.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.release.sys.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.release.sys.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

// 主模板函数
template<typename T>
__device__ void store_cta_relaxed(T* addr, T val) {  
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.relaxed.cta.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.relaxed.cta.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.relaxed.cta.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.relaxed.cta.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_cluster_relaxed(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.relaxed.cluster.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.relaxed.cluster.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.relaxed.cluster.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.relaxed.cluster.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_gpu_relaxed(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.relaxed.gpu.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.relaxed.gpu.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.relaxed.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_sys_relaxed(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.relaxed.sys.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.relaxed.sys.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.relaxed.sys.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.relaxed.sys.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

} // namespace cuda

// #endif // __CUDA_ARCH__

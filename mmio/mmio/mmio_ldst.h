// cuda_scoped_store.h
#pragma once

// #ifdef __CUDA_ARCH__

namespace mmio_ldst {

enum class scope { cta, cluster, gpu, sys };

// 主模板函数
template<typename T>
__device__ void store_mmio_gpu(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.async.mmio.release.gpu.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.async.mmio.release.gpu.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.async.mmio.release.gpu.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.async.mmio.release.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void store_mmio_sys(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.async.mmio.release.sys.global.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.async.mmio.release.sys.global.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.async.mmio.release.sys.global.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.async.mmio.release.sys.global.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

// 主模板函数
template<typename T>
__device__ void store_mmio_gpu_relaxed(T* addr, T val) {       
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.relaxed.mmio.sys.u8 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.relaxed.mmio.sys.u16 [%0], %1;" : : "l"(addr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" : : "l"(addr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.relaxed.mmio.sys.u64 [%0], %1;" : : "l"(addr), "l"(val));
    }
}

template<typename T>
__device__ void load_mmio_gpu(T* addr, T val) {
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.async.mmio.release.gpu.u8  %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.async.mmio.release.gpu.u16 %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.async.mmio.release.gpu.u32 %0, [%1];" : : "r"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.async.mmio.release.gpu.u64 %0, [%1];" : : "l"(val), "l"(addr));
    }
}

template<typename T>
__device__ void load_mmio_sys(T* addr, T val) {
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.async.mmio.release.sys.global.u8  %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.async.mmio.release.sys.global.u16 %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.async.mmio.release.sys.global.u32 %0, [%1];" : : "r"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.async.mmio.release.sys.global.u64 %0, [%1];" : : "l"(val), "l"(addr));
    }
}

// 主模板函数
template<typename T>
__device__ void load_mmio_gpu_relaxed(T* addr, T val) {
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.relaxed.mmio.sys.u8 %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.relaxed.mmio.sys.u16 %0, [%1];" : : "h"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : : "r"(val), "l"(addr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.relaxed.mmio.sys.u64 %0, [%1];" : : "l"(val), "l"(addr));
    }
}

} // namespace cuda

// #endif // __CUDA_ARCH__

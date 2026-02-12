#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "memst_con.fatbin.c"
extern void __device_stub__Z19device_memst_kernelPiPKii(int *, const int *, const int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z19device_memst_kernelPiPKii(int *__par0, const int *__par1, const int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(int *, const int *, const int))device_memst_kernel)));}
# 25 "memst_con.cu"
void device_memst_kernel( int *__cuda_0,const int *__cuda_1,const int __cuda_2)
# 25 "memst_con.cu"
{__device_stub__Z19device_memst_kernelPiPKii( __cuda_0,__cuda_1,__cuda_2);
# 67 "memst_con.cu"
}
# 1 "memst_con.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T3) {  __nv_dummy_param_ref(__T3); __nv_save_fatbinhandle_for_managed_rt(__T3); __cudaRegisterEntry(__T3, ((void ( *)(int *, const int *, const int))device_memst_kernel), _Z19device_memst_kernelPiPKii, 256); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop

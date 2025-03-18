#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

typedef int cudaError_t;

static void *(*ori_malloc)(size_t size);
static cudaError_t (*ori_cudaMalloc)(void** devPtr, size_t size);

// static pthread_mutex_t api_mutex = PTHREAD_MUTEX_INITIALIZER;

void __attribute__((constructor))
init_ori(void)
{
  // ori_malloc = dlsym(RTLD_NEXT, "malloc");
  void *cuda_handle = dlopen("/usr/local/cuda/lib64/libcudart.so", RTLD_LAZY);
  ori_cudaMalloc = dlsym(cuda_handle, "cudaMalloc");
}

// void *malloc(size_t size)
// {
//   time_t time_stamp = time(NULL);
//   void *ret = ori_malloc(size);
//   char c[256];
//   sprintf(c, "malloc,%ld,%p,%ld\n", time_stamp,ret,size);
//   fprintf(stderr, "%s", c);
//   return ret;
// }

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
  time_t time_stamp = time(NULL);
  cudaError_t ret = ori_cudaMalloc(devPtr, size);
  char c[256];
  sprintf(c, "cudaMalloc,%ld,%p,%ld\n", time_stamp,devPtr,size);
  fprintf(stderr, "%s", c);
  return ret;
}
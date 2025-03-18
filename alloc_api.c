#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static void *(*ori_malloc)(size_t size);

void __attribute__((constructor))
init_ori_malloc(void)
{
  ori_malloc = dlsym(RTLD_NEXT, "malloc");
}

void *malloc(size_t size)
{
  time_t time_stamp = time(NULL);
  void *ret = ori_malloc(size);
  char c[256];
  sprintf(c, "malloc,%ld,%p,%ld\n", time_stamp,ret,size);
  fprintf(stderr, "%s", c);
  return ret;
}
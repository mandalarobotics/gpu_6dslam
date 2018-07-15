#ifndef _FALLBACK_ALLOCATOR_CUH_
#define _FALLBACK_ALLOCATOR_CUH_

#include <thrust/functional.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

class fallback_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;
    // allocate's job to is allocate host memory as a functional fallback when cudaMalloc fails
    char *allocate(std::ptrdiff_t n);
    // deallocate's job to is inspect where the pointer lives and free it appropriately
    void deallocate(char *ptr, size_t n);
};

#endif

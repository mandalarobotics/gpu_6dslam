#ifndef __CUDAEXCEPTION_H__
#define __CUDAEXCEPTION_H__

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

void throw_cuda_error(cudaError_t code, const char *file, int line);
void throw_cuda_error(cublasStatus_t code, const char *file, int line);
void throw_cuda_error(cusolverStatus_t code, const char *file, int line);
void throw_cuda_error(cusparseStatus_t code, const char *file, int line);


class MyCudaError:public std::runtime_error
{
public:
	MyCudaError(int errCode, const std::string & errSrc, const std::string & errMsg)
		: std::runtime_error(errMsg), err(errCode),  source(errSrc) {

	}

	~MyCudaError(){};

	int err;
	std::string source;
};

#endif

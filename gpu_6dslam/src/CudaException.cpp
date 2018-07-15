#include "CudaException.h"
#include "helper_cuda.h"

void throw_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess){
	std::cout << "CUDA error: " << code << " " << _cudaGetErrorEnum(code) << " " << file << " " << line << std::endl;


	std::stringstream ss;
    ss << "cuda: " << file << "(" << line << ")";
    std::string file_and_line = ss.str();
    throw MyCudaError(code, file_and_line, _cudaGetErrorEnum(code));
  }
}

void throw_cuda_error(cublasStatus_t code, const char *file, int line)
{
  if(code != CUBLAS_STATUS_SUCCESS){
	std::cout << "CUDA error: " << code << " " << _cudaGetErrorEnum(code) << " " << file << " " << line << std::endl;

	std::stringstream ss;
	ss << "cublas: " << file << "(" << line << ")";
	std::string file_and_line = ss.str();
    throw MyCudaError(code, file_and_line, _cudaGetErrorEnum(code));
  }
}

void throw_cuda_error(cusolverStatus_t code, const char *file, int line)
{
  if(code != CUSOLVER_STATUS_SUCCESS){
	std::cout << "CUDA error: " << code << " " << _cudaGetErrorEnum(code) << " " << file << " " << line << std::endl;

	std::stringstream ss;
	ss << "cublas: " << file << "(" << line << ")";
	std::string file_and_line = ss.str();
	throw MyCudaError(code, file_and_line, _cudaGetErrorEnum(code));
  }
}

void throw_cuda_error(cusparseStatus_t code, const char *file, int line)
{
  if(code != CUSPARSE_STATUS_SUCCESS)
  {
	std::cout << "CUDA error: " << code << " " << _cudaGetErrorEnum(code) << " " << file << " " << line << std::endl;

	std::stringstream ss;
	ss << "cublas: " << file << "(" << line << ")";
	std::string file_and_line = ss.str();
	throw MyCudaError(code, file_and_line, _cudaGetErrorEnum(code));
  }
}


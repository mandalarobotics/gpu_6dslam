#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


#include <iostream>
#include <iomanip>

#include "CCUDAAXBSolverWrapper.h"
#include "helper_cuda.h"
#include "lesson_16.h"
#include "helper_cusolver.h"

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#define TEST

CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper(bool _CCUDA_AX_B_SolverWrapperDEBUG, int cuda_device) {

	CCUDA_AX_B_SolverWrapperDEBUG = _CCUDA_AX_B_SolverWrapperDEBUG;

	handle = NULL;
	cublasHandle = NULL; // used in residual evaluation
	stream = NULL;

	cudaError_t errCUDA = ::cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	errCUDA = cudaSetDevice(cuda_device);
	assert(::cudaSuccess == errCUDA);


	checkCudaErrors(cusolver_status = cusolverDnCreate(&handle));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "cusolverDnCreate(&handle)");
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	checkCudaErrors(cublas_status = cublasCreate(&cublasHandle));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasCreate(&cublasHandle)");
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	errCUDA = cudaStreamCreate(&stream);
		throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	checkCudaErrors(cusolver_status = cusolverDnSetStream(handle, stream));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "cusolverDnSetStream(handle, stream)");
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	checkCudaErrors(cublas_status = cublasSetStream(cublasHandle, stream));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasSetStream(cublasHandle, stream)");
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	//this->d_A = 0;
	//this->d_x = 0;

	//this->d_P = 0;
	//this->d_AtP = 0;
	//this->d_AtPA = 0;
	//this->d_l = 0;
	//this->d_AtPl = 0;

	//this->d_a = 0;
	//this->d_b = 0;
	//this->d_c = 0;

	//this->info = 0;
	//this->buffer = 0;
	//this->ipiv = 0;
	//this->tau = 0;
}

CCUDA_AX_B_SolverWrapper::~CCUDA_AX_B_SolverWrapper() {
	cudaError_t errCUDA = ::cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	if (handle)
	{
	  	checkCudaErrors(cusolver_status = cusolverDnDestroy(handle));
	    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "checkCudaErrors(cusolverDnDestroy(handle))");
	   	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	}

	if (cublasHandle)
	{
	   	checkCudaErrors(cublas_status = cublasDestroy(cublasHandle));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasDestroy(cublasHandle)");
	    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	}

	if (stream)
	{
	   	errCUDA = cudaStreamDestroy(stream);
	   		throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	}

	//cudaFree(this->d_A);
	this->d_A.dispose();

	//cudaFree(this->d_x);
	this->d_x.dispose();

	//cudaFree(this->d_P);
	this->d_P.dispose();

	//cudaFree(this->d_AtP);
	this->d_AtP.dispose();

	//cudaFree(this->d_AtPA);
	this->d_AtPA.dispose();

	//cudaFree(this->d_l);
	this->d_l.dispose();

	//cudaFree(this->d_AtPl);
	this->d_AtPl.dispose();

	//cudaFree(this->d_a);
	this->d_a.dispose();

	//cudaFree(this->d_b);
	this->d_b.dispose();

	//cudaFree(this->d_c);
	this->d_c.dispose();

	//cudaFree(this->info);
	this->info.dispose();

	//cudaFree(this->buffer);
	this->buffer.dispose();

	//cudaFree(this->ipiv);
	this->ipiv.dispose();

	//cudaFree(this->tau);
	this->tau.dispose();
}

double CCUDA_AX_B_SolverWrapper::Solve(double *a,double *b,double *x, int a_rows, int a_cols, int b_cols, char method)
{
		clock_t begin_time;
		double solve_time;

		begin_time = clock();

		    int rowsA = 0; // number of rows of A
		    int colsA = 0; // number of columns of A
		    int lda   = 0; // leading dimension in dense matrix

		    lda = a_rows;
		    rowsA = a_rows;
		    colsA = a_cols;

		       	// verify if A is symmetric or not.
		        if ( method == chol )
		        {
		            int issym = 1;
		            for(int j = 0 ; j < colsA ; j++)
		            {
		                for(int i = j ; i < rowsA ; i++)
		                {
		                    double Aij = a[i + j*lda];
		                    double Aji = a[j + i*lda];
		                    if ( Aij != Aji )
		                    {
		                        issym = 0;
		                        break;
		                    }
		                }
		            }
		            if (!issym)
		            {
		                printf("Error: A has no symmetric pattern, please use LU or QR \n");
		                exit(EXIT_FAILURE);
		            }
		        }

		    cudaError_t errCUDA = ::cudaSuccess;
		    //errCUDA = cudaMalloc((void **)&d_A, sizeof(double)*lda*colsA);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    d_A.init(lda*colsA);
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


		    //errCUDA = cudaMalloc((void **)&d_x, sizeof(double)*colsA);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    d_x.init(colsA);
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


		    //errCUDA = cudaMalloc((void **)&d_b, sizeof(double)*rowsA);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    d_b.init(rowsA);
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

		    //errCUDA = cudaMemcpy(d_A, a, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    this->d_A.copyFromHostToDevice(a, lda*colsA);
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



		    //errCUDA = cudaMemcpy(d_b, b, sizeof(double)*rowsA, cudaMemcpyHostToDevice);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    this->d_b.copyFromHostToDevice(b, rowsA);
			throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



		    if ( method == chol)
		    {
		         linearSolverCHOL(handle, rowsA, d_A.data, lda, d_b.data, d_x.data);
		    }
		    else if ( method == lu )
		    {
		         linearSolverLU(handle, rowsA, d_A.data, lda, d_b.data, d_x.data);
		    }
		    else if ( method ==  qr)
		    {
		         linearSolverQR(handle, rowsA, d_A.data, lda, d_b.data, d_x.data);
		    }
		    else
		    {
		        fprintf(stderr, "Error: %d is unknown function\n", method);
		    }

		    //errCUDA = cudaMemcpy(x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost);
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    this->d_x.copyFromDeviceToHost(x);
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



			//errCUDA = cudaFree(this->d_A); this->d_A = 0;
		    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    this->d_A.dispose();
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

			//errCUDA = cudaFree(this->d_x); this->d_x = 0;
			//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		    this->d_x.dispose();
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

			//errCUDA = cudaFree(this->d_b); this->d_b = 0;
			//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
			this->d_b.dispose();
		    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


		solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
return solve_time;
}



double CCUDA_AX_B_SolverWrapper::Compute_AtP(int threads, double *A, double *P, double *AtP, int rows, int columns)
{
	clock_t begin_time = clock();
	double solve_time = 0.0;

	cudaError_t errCUDA = ::cudaSuccess;
	//errCUDA = cudaMalloc((void **)&this->d_A, sizeof(double)*rows*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_A.init(rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaMalloc((void **)&this->d_P, sizeof(double)*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_P.init(columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc((void **)&this->d_AtP, sizeof(double)*rows*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_AtP.init(rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(this->d_A, A, sizeof(double)*rows*columns, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_A.copyFromHostToDevice(A, rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMemcpy(this->d_P, P, sizeof(double)*columns, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_P.copyFromHostToDevice(P, columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	errCUDA = cudaCompute_AtP(threads, this->d_A.data, this->d_P.data, this->d_AtP.data, rows, columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(AtP, d_AtP, sizeof(double)*rows*columns, cudaMemcpyDeviceToHost);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_AtP.copyFromDeviceToHost(AtP);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaFree(this->d_A); this->d_A = 0;
	//  	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_A.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


//	errCUDA = cudaFree(this->d_P); this->d_P = 0;
	//   	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_P.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(this->d_AtP); this->d_AtP = 0;
	//  	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_AtP.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
	return solve_time;
}

double CCUDA_AX_B_SolverWrapper::Multiply(double *a, double *b,double *c, int a_rows, int a_cols, int b_cols)
{
	clock_t begin_time;
	double solve_time;
	begin_time = clock();

	cudaError_t errCUDA = ::cudaSuccess;
	//errCUDA = cudaMalloc((void **)&this->d_a, sizeof(double)*a_rows*a_cols);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_a.init(a_rows*a_cols);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc((void **)&this->d_b, sizeof(double)*a_cols*b_cols);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_b.init(a_cols*b_cols);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaMalloc((void **)&this->d_c, sizeof(double)*a_rows*b_cols);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_c.init(a_rows*b_cols);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(this->d_a, a, sizeof(double)*a_rows*a_cols, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_a.copyFromHostToDevice(a, a_rows*a_cols);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMemcpy(this->d_b, b, sizeof(double)*a_cols*b_cols, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_b.copyFromHostToDevice(b, a_cols*b_cols);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



		/*enum cublasStatus_t {
		    CUBLAS_STATUS_SUCCESS,
		    CUBLAS_STATUS_NOT_INITIALIZED,
		    CUBLAS_STATUS_ALLOC_FAILED,
		    CUBLAS_STATUS_INVALID_VALUE,
		    CUBLAS_STATUS_ARCH_MISMATCH,
		    CUBLAS_STATUS_MAPPING_ERROR,
		    CUBLAS_STATUS_EXECUTION_FAILED,
		    CUBLAS_STATUS_INTERNAL_ERROR,
		    CUBLAS_STATUS_NOT_SUPPORTED,
		    CUBLAS_STATUS_LICENSE_ERROR,
		}*/
	//ToDo cublasStatus_t error check is missing
	multiplyCUBLAS( cublasHandle, this->d_a.data, this->d_b.data, this->d_c.data, a_rows, a_cols, b_cols);
	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMemcpy(c, this->d_c, sizeof(double)*a_rows*b_cols, cudaMemcpyDeviceToHost);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_c.copyFromDeviceToHost(c);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(d_a); this->d_a = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_a.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	//errCUDA = cudaFree(d_b); this->d_b = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_b.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	//errCUDA = cudaFree(d_c); this->d_c = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_c.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
	return solve_time;
}


CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error CCUDA_AX_B_SolverWrapper::Solve_ATPA_ATPl_x(int threads, double *A, double *P, double *l, double *x,
		int rows, int columns, CCUDA_AX_B_SolverWrapper::Solver_Method solver_method)
{
	cudaError_t errCUDA = ::cudaSuccess;
	//errCUDA = cudaMalloc((void **)&d_A, sizeof(double)*rows*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_A.init(rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(d_A, A, sizeof(double)*rows*columns, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_A.copyFromHostToDevice(A, rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc((void **)&d_P, sizeof(double)*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_P.init(columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaMemcpy(d_P, P, sizeof(double)*columns, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_P.copyFromHostToDevice(P, columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);




	//errCUDA = cudaMalloc((void **)&d_AtP, sizeof(double)*rows*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_AtP.init(rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	errCUDA = cudaCompute_AtP(threads, d_A.data, d_P.data, d_AtP.data, rows, columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(d_P);
	//   	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_P.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaMalloc((void **)&d_AtPA, sizeof(double)*rows*rows); //
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_AtPA.init(rows*rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	multiplyCUBLAS( cublasHandle, d_AtP.data, d_A.data, d_AtPA.data, rows, columns, rows);
	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(d_A);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_A.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc((void **)&d_l, sizeof(double)*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_l.init(columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaMemcpy(d_l, l, sizeof(double)*columns, cudaMemcpyHostToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_l.copyFromHostToDevice(l, columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);




	//errCUDA = cudaMalloc((void **)&d_AtPl, sizeof(double)*rows);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_AtPl.init(rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);




	multiplyCUBLAS(cublasHandle, d_AtP.data, d_l.data, d_AtPl.data, rows, columns, 1);
	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaFree(d_l);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_l.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(d_AtP);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_AtP.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc((void **)&d_x, sizeof(double)*rows);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_x.init(rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	if ( solver_method == chol)
	{
		linearSolverCHOL(handle, rows, d_AtPA.data, rows, d_AtPl.data, d_x.data);
		//errCUDA = cudaGetLastError();
		//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else if ( solver_method == lu )
	{
		linearSolverLU(handle, rows, d_AtPA.data, rows, d_AtPl.data, d_x.data);
		//errCUDA = cudaGetLastError();
		//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else if ( solver_method ==  qr)
	{
	    linearSolverQR(handle, rows, d_AtPA.data, rows, d_AtPl.data, d_x.data);
	    //errCUDA = cudaGetLastError();
	    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	    throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else
	{
		return fail_problem_with_CUDA_AX_B_Solver;
	}

	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(x, d_x, sizeof(double)*rows, cudaMemcpyDeviceToHost);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_x.copyFromDeviceToHost(x);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	//errCUDA = cudaFree(d_AtPA);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_AtPA.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(d_AtPl);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_AtPl.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaFree(d_x);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_x.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	return success;
}

CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error CCUDA_AX_B_SolverWrapper::Solve_ATPA_ATPl_x_data_on_GPU(int threads, double *_d_A, double *_d_P, double *_d_l,
			double *x, int rows, int columns, Solver_Method solver_method)
{
	cudaError_t errCUDA = ::cudaSuccess;
	//errCUDA = cudaMalloc((void **)&this->d_AtP, sizeof(double)*rows*columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_AtP.init(rows*columns);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	errCUDA = cudaCompute_AtP(threads, _d_A, _d_P, this->d_AtP.data, rows, columns);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMalloc((void **)&this->d_AtPA, sizeof(double)*rows*rows); //
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


	this->d_AtPA.init(rows*rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	multiplyCUBLAS( cublasHandle, this->d_AtP.data, _d_A, this->d_AtPA.data, rows, columns, rows);
	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMalloc((void **)&this->d_AtPl, sizeof(double)*rows);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_AtPl.init(rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	multiplyCUBLAS(cublasHandle, this->d_AtP.data, _d_l, this->d_AtPl.data, rows, columns, 1);
	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(this->d_AtP); this->d_AtP = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_AtP.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMalloc((void **)&this->d_x, sizeof(double)*rows);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_x.init(rows);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	if ( solver_method == chol)
	{
		linearSolverCHOL(handle, rows, this->d_AtPA.data, rows, this->d_AtPl.data, this->d_x.data);
		//errCUDA = cudaGetLastError();
		//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else if ( solver_method == lu )
	{
		linearSolverLU(handle, rows, this->d_AtPA.data, rows, this->d_AtPl.data, this->d_x.data);
		//errCUDA = cudaGetLastError();
		//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else if ( solver_method ==  qr)
	{
		linearSolverQR(handle, rows, this->d_AtPA.data, rows, this->d_AtPl.data, this->d_x.data);
		//errCUDA = cudaGetLastError();
		//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
		throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
	}
	else
	{
		return fail_problem_with_CUDA_AX_B_Solver;
	}

	//errCUDA = cudaGetLastError();
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaMemcpy(x, this->d_x, sizeof(double)*rows, cudaMemcpyDeviceToHost);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_x.copyFromDeviceToHost(x);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaFree(this->d_AtPA); this->d_AtPA = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_AtPA.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaFree(this->d_AtPl); this->d_AtPl = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	this->d_AtPl.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(this->d_x); this->d_x = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	this->d_x.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	return success;
}

////////////////////////////////////////////////////////////////////////////////////
/*
 *  solve A*x = b by Cholesky factorization
 *
 */

int CCUDA_AX_B_SolverWrapper::linearSolverCHOL(
	    cusolverDnHandle_t handle,
	    int n,
	    const double *Acopy,
	    int lda,
	    const double *b,
	    double *x)
{
	cudaError_t errCUDA = ::cudaSuccess;
	int bufferSize = 0;

	int h_info = 0;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	//cusolverStatus_t
	checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize));

	//errCUDA = cudaMalloc(&info, sizeof(int));
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	info.init(1);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc(&buffer, sizeof(double)*bufferSize);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	buffer.init(bufferSize);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaMalloc(&d_A, sizeof(double)*lda*n);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_A.init(lda*n);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	// prepare a copy of A because potrf will overwrite A with L
	//errCUDA = cudaMemcpy(d_A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	//d_A.copyFromDeviceToDevice()

	cudaMemcpy(d_A.data, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	errCUDA = cudaMemset(info.data, 0, sizeof(int));
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, d_A.data, lda, buffer.data, bufferSize, info.data));
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);



	errCUDA = cudaMemcpy(&h_info, info.data, sizeof(int), cudaMemcpyDeviceToHost);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	if ( 0 != h_info ){
		fprintf(stderr, "Error: linearSolverCHOL failed\n");
	}

	errCUDA = cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice);
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, d_A.data, lda, x, n, info.data));

	checkCudaErrors(cudaDeviceSynchronize());

	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);


	//errCUDA = cudaFree(info); info = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	info.dispose();
	throw_cuda_error(cudaGetLastError(), __FILE__, __LINE__);

	//errCUDA = cudaFree(buffer); buffer = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	buffer.dispose();
	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


	//errCUDA = cudaFree(d_A); d_A = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
	d_A.dispose();
	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	return 0;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int CCUDA_AX_B_SolverWrapper::linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
	cudaError_t errCUDA = ::cudaSuccess;
    int bufferSize = 0;

    int h_info = 0;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    //errCUDA = cudaMalloc(&info, sizeof(int));
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    info.init(1);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    //errCUDA = cudaMalloc(&buffer, sizeof(double)*bufferSize);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    buffer.init(bufferSize);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    //errCUDA = cudaMalloc(&d_A, sizeof(double)*lda*n);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    d_A.init(lda*n);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    //errCUDA = cudaMalloc(&ipiv, sizeof(int)*n);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    ipiv.init(n);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    // prepare a copy of A because getrf will overwrite A with L
    errCUDA = cudaMemcpy(d_A.data, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    errCUDA = cudaMemset(info.data, 0, sizeof(int));
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    checkCudaErrors(cusolverDnDgetrf(handle, n, n, d_A.data, lda, buffer.data, ipiv.data, info.data));
    //errCUDA = cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    errCUDA = cudaMemcpy(&h_info, info.data, sizeof(int), cudaMemcpyDeviceToHost);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    if ( 0 != h_info ){
        fprintf(stderr, "Error: linearSolverLU failed\n");
    }

    errCUDA = cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice);
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, d_A.data, lda, ipiv.data, x, n, info.data));
    errCUDA = cudaDeviceSynchronize();
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    	//errCUDA = cudaFree(info  ); info = 0;
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    info.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    	//errCUDA = cudaFree(buffer); buffer = 0;
		//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    buffer.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	//errCUDA = cudaFree(d_A); d_A = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    d_A.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    //errCUDA = cudaFree(ipiv); ipiv = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    ipiv.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    return 0;
}

/*
 *  solve A*x = b by QR
 *
 */
int CCUDA_AX_B_SolverWrapper::linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
	cudaError_t errCUDA = ::cudaSuccess;

    cublasHandle_t cublasHandle = NULL;
    int bufferSize = 0;

    int h_info = 0;
    const double one = 1.0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    //errCUDA = cudaMalloc(&info, sizeof(int));
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    info.init(1);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    	//errCUDA = cudaMalloc(&buffer, sizeof(double)*bufferSize);
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    buffer.init(bufferSize);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    //errCUDA = cudaMalloc(&d_A, sizeof(double)*lda*n);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	d_A.init(lda*n);
	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    //errCUDA = cudaMalloc ((void**)&tau, sizeof(double)*n);
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	tau.init(n);
	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    errCUDA = cudaMemcpy(d_A.data, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice);
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    //errCUDA = cudaMemset(info, 0, sizeof(int));
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    cudaMemset(info.data, 0, sizeof(int));
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, d_A.data, lda, tau.data, buffer.data, bufferSize, info.data));
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    errCUDA = cudaMemcpy(&h_info, info.data, sizeof(int), cudaMemcpyDeviceToHost);
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    if ( 0 != h_info ){
        fprintf(stderr, "Error: linearSolverQR failed\n");
    }

    errCUDA = cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        d_A.data,
        lda,
        tau.data,
        x,
        n,
        buffer.data,
        bufferSize,
        info.data));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         d_A.data,
         lda,
         x,
         n));
    errCUDA = cudaDeviceSynchronize();
    	//throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    //errCUDA = cudaFree(info  ); info = 0;
    //	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    info.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);


   // errCUDA = cudaFree(buffer); buffer = 0;
//		throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    buffer.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	//errCUDA = cudaFree(d_A); d_A = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

    d_A.dispose();
    throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	//errCUDA = cudaFree(tau); tau = 0;
	//	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);

	tau.dispose();
	throw_on_cuda_error(errCUDA, __FILE__, __LINE__);
    return 0;
}

cublasStatus_t CCUDA_AX_B_SolverWrapper::multiplyCUBLAS( cublasHandle_t handle, const double *d_a, const double *d_b, double *d_c, int a_rows, int a_cols, int b_cols)
{
	const double alpha = 1.0;
	const double beta  = 0.0;

	return cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			a_rows, b_cols, a_cols,
			&alpha,
			d_a, a_rows,
			d_b, a_cols,
			&beta, d_c, a_rows);
}


void CCUDA_AX_B_SolverWrapper::cout_cudaError_t(cudaError_t err, string message)
{

	switch(err)
	{
		case ::cudaSuccess:
		{
			std::cout << message << " ::cudaSuccess"  << std::endl;
			break;
		}
		case ::cudaErrorMissingConfiguration:
		{
			std::cout<< message << " ::cudaErrorMissingConfiguration"  << std::endl;
			break;
		}
		case ::cudaErrorMemoryAllocation:
		{
			std::cout<< message << " ::cudaErrorMemoryAllocation"  << std::endl;
			break;
		}
		case ::cudaErrorInitializationError:
		{
			std::cout<< message <<" ::cudaErrorInitializationError"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchFailure:
		{
			std::cout<< message<<" ::cudaErrorLaunchFailure"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchTimeout:
		{
			std::cout<< message << "::cudaErrorLaunchTimeout"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchOutOfResources:
		{
			std::cout<<message<< " ::cudaErrorLaunchOutOfResources"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDeviceFunction:
		{
			std::cout<< message <<" ::cudaErrorInvalidDeviceFunction"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidConfiguration:
		{
			std::cout<< message << " ::cudaErrorInvalidConfiguration"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDevice:
		{
			std::cout<< message << " ::cudaErrorInvalidDevice"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidValue:
		{
			std::cout<< message << " ::cudaErrorInvalidValue"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidPitchValue:
		{
			std::cout<< message << "::cudaErrorInvalidPitchValue"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidSymbol:
		{
			std::cout<< message << " ::cudaErrorInvalidSymbol"  << std::endl;
			break;
		}
		case ::cudaErrorUnmapBufferObjectFailed:
		{
			std::cout<< message << " ::cudaErrorUnmapBufferObjectFailed"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidHostPointer:
		{
			std::cout<< message << " ::cudaErrorInvalidHostPointer"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDevicePointer:
		{
			std::cout<< message << " ::cudaErrorInvalidDevicePointer"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidTexture:
		{
			std::cout<< message << " ::cudaErrorInvalidTexture"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidTextureBinding:
		{
			std::cout<< message << " ::cudaErrorInvalidTextureBinding"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidChannelDescriptor:
		{
			std::cout<< message << " ::cudaErrorInvalidChannelDescriptor"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidMemcpyDirection:
		{
			std::cout<< message << " ::cudaErrorInvalidMemcpyDirection"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidFilterSetting:
		{
			std::cout<< message << " ::cudaErrorInvalidFilterSetting"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidNormSetting:
		{
			std::cout<< message << " ::cudaErrorInvalidNormSetting"  << std::endl;
			break;
		}
		case ::cudaErrorUnknown:
		{
			std::cout<< message << " ::cudaErrorUnknown"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidResourceHandle:
		{
			std::cout<< message << " ::cudaErrorInvalidResourceHandle"  << std::endl;
			break;
		}
		case ::cudaErrorInsufficientDriver:
		{
			std::cout<< message << " ::cudaErrorInsufficientDriver"  << std::endl;
			break;
		}
		case ::cudaErrorSetOnActiveProcess:
		{
			std::cout<< message << " ::cudaErrorSetOnActiveProcess"  << std::endl;
			break;
		}
		case ::cudaErrorStartupFailure:
		{
			std::cout<< message << " ::cudaErrorStartupFailure"  << std::endl;
			break;
		}
		case ::cudaErrorIllegalAddress:
		{
			std::cout<< message << " ::cudaErrorIllegalAddress"  << std::endl;
			break;
		}
		default:
		{
			std::cout<< message << " error_code: "  << err << std::endl;
			break;
		}
	}
}

void CCUDA_AX_B_SolverWrapper::cout_cusolverStatus_t(cusolverStatus_t err, string message)
{
	switch(err)
	{
		case ::CUSOLVER_STATUS_SUCCESS:
		{
			std::cout << message << " ::CUSOLVER_STATUS_SUCCESS"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_NOT_INITIALIZED:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_NOT_INITIALIZED"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_ALLOC_FAILED:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_ALLOC_FAILED"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_ARCH_MISMATCH:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_ARCH_MISMATCH"  << std::endl;
			break;
		}
		default:
		{
			std::cout<< message << " error_code: "  << err << std::endl;
			break;
		}
	}
}

void CCUDA_AX_B_SolverWrapper::cout_cublasStatus_t(cublasStatus_t err, string message)
{
		switch(err)
		{
			case ::CUBLAS_STATUS_SUCCESS:
			{
				std::cout << message << " ::CUBLAS_STATUS_SUCCESS"  << std::endl;
				break;
			}
			case ::CUBLAS_STATUS_NOT_INITIALIZED:
			{
				std::cout<< message << " ::CUBLAS_STATUS_NOT_INITIALIZED"  << std::endl;
				break;
			}
			case ::CUBLAS_STATUS_ALLOC_FAILED:
			{
				std::cout<< message << " ::CUBLAS_STATUS_ALLOC_FAILED"  << std::endl;
				break;
			}

			default:
			{
				std::cout<< message << " error_code: "  << err << std::endl;
				break;
			}
		}
}

void CCUDA_AX_B_SolverWrapper::throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

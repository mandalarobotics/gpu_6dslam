#include "fallback_allocator.cuh"

//#include <boost/chrono.hpp>
//#include <boost/log/common.hpp>
//#include <boost/log/expressions.hpp>
//#include <boost/log/utility/setup/file.hpp>
//#include <boost/log/utility/setup/console.hpp>
//#include <boost/log/utility/setup/common_attributes.hpp>
//#include <boost/log/attributes/timer.hpp>
//#include <boost/log/attributes/named_scope.hpp>
//#include <boost/log/sources/logger.hpp>
//#include <boost/log/core.hpp>
//#include <boost/log/expressions.hpp>

char *fallback_allocator::allocate(std::ptrdiff_t n){
	//boost::log::sources::logger lg;

	char *result = 0;

	      size_t mfree;
	      size_t mtotal;

	      //BOOST_LOG(lg) << "trying allocate:";

	      if(cudaMemGetInfo 	( &mfree, &mtotal) ==  cudaSuccess){
	    	  //BOOST_LOG(lg) << "memoryfree: " << mfree << " of memorytotal " << mtotal;

	    	  if(mfree < n){
	    		  //BOOST_LOG(lg) << "no GPU memory...";
	    	  }
	      }

	      // attempt to allocate device memory
	      if(cudaMalloc(&result, n) == cudaSuccess)
	      {
	    	  //BOOST_LOG(lg) << "allocated " << n << " bytes of device memory";

	        if(cudaMemGetInfo 	( &mfree, &mtotal) ==  cudaSuccess){
	        	//BOOST_LOG(lg) << "AFTER cudaMalloc memoryfree: " << mfree << " of memorytotal " << mtotal;
		    }
	      }
	      else
	      {
	        // reset the last CUDA error
	        cudaGetLastError();

	        // attempt to allocate pinned host memory
	        void *h_ptr = 0;
	        if(cudaMallocHost(&h_ptr, n) == cudaSuccess)
	        {
	          // attempt to map host pointer into device memory space
	          if(cudaHostGetDevicePointer(&result, h_ptr, 0) == cudaSuccess)
	          {
	        	  //BOOST_LOG(lg) << "allocated " << n << " bytes of pinned host memory (fallback successful)";
	          }
	          else
	          {
	            // reset the last CUDA error
	            cudaGetLastError();

	            // attempt to deallocate buffer
	            //BOOST_LOG(lg) << "failed to map host memory into device address space (fallback failed)";
	            cudaFreeHost(h_ptr);

	            throw std::bad_alloc();
	          }
	        }
	        else
	        {
	          // reset the last CUDA error
	          cudaGetLastError();

	          //BOOST_LOG(lg) << "failed to allocate " << n << " bytes of memory (fallback failed)";

	          throw std::bad_alloc();
	        }
	      }
	      return result;
}

void fallback_allocator::deallocate(char *ptr, size_t n)
{
	void *raw_ptr = thrust::raw_pointer_cast(ptr);

	// determine where memory resides
	cudaPointerAttributes	attributes;

	if(cudaPointerGetAttributes(&attributes, raw_ptr) == cudaSuccess){
		// free the memory in the appropriate way
		if(attributes.memoryType == cudaMemoryTypeHost){
			cudaFreeHost(raw_ptr);
		}
		else{
			cudaFree(raw_ptr);
		}
	}
}



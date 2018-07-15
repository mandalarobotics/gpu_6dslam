#ifndef ARRAY_CUH
#define ARRAY_CUH

#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <utility>

#include "fallback_allocator.cuh"
#include "CudaException.h"

//struct RangeStart {
//	bool isRangeStart;
//	int index;
//};

//template <typename E>
//struct ArrCuda {
//	E * data;
//	int size;
//};


template <typename E>
struct DataBufferCuda {
	E * data;
	int size;
};

template <typename E, typename ALLOC_TYPE=fallback_allocator>
class DataBuffer {
public:

	E * data;
	size_t size;
	ALLOC_TYPE alloc;

	DataBuffer(const ALLOC_TYPE & alloc_= ALLOC_TYPE()) : data(NULL), size(0) {}

	DataBuffer(const DataBuffer<E> & other, const ALLOC_TYPE & alloc_= ALLOC_TYPE()) : data(NULL), size(0) {
		copyFromArr(other);
	}

	//Arr(size_t size_, const ALLOC_TYPE & alloc_= ALLOC_TYPE()) {
	//	std::cout << "gg" << std::endl;
	//	init(size_);
	//}

	void init(size_t size_) {

		if(data != NULL){
			dispose();
		}
		//size = size_;
		//cudaMalloc(&data, size * sizeof(E));
		this->size = size_;
		this->data = reinterpret_cast<E*>(alloc.allocate(size_ * sizeof(E)));
	}

	void dispose() {

		if (data != NULL) {
			alloc.deallocate(reinterpret_cast<char*>(this->data), this->size * sizeof(E));
		//	cudaFree(data);
			data = NULL;
			size = 0;
		}
	}

	~DataBuffer() {
		dispose();
	}

	//ArrCuda<E> getArrCuda() {
	//	ArrCuda<E> arr;
	//	arr.data = data;
	//	arr.size = size;
	//	return arr;
	//}

	void copyFromHostToDevice(std::vector<E> & v) {
		copyFromHostToDevice(&(v[0]), v.size());
	}

	template <typename K>
	void copyFromHostToDevice(K * v, std::size_t size_) {
		//ToDo sizeof(K) == sizeof(E)
			if (size != size_)
			{
				dispose();
				init(size_);
			}
			cudaError_t errCUDA = cudaMemcpy(data, v, size_ * sizeof(E), cudaMemcpyHostToDevice) ;
			throw_cuda_error(errCUDA, __FILE__, __LINE__);
		}
	void copyFromDeviceToDevice(E * v_from, std::size_t size_) {
		if (size != size_)
		{
			dispose();
			init(size_);
		}
		cudaError_t errCUDA = cudaMemcpy(data, v_from, size_ * sizeof(E), cudaMemcpyDeviceToDevice) ;
		throw_cuda_error(errCUDA, __FILE__, __LINE__);
	}

	void copyFromDeviceToDevice(const DataBuffer<E> & other) {
		copyFromDeviceToDevice(other.data, other.size);
	}

	void copyFromDeviceToHost(std::vector<E> & v) {
		if (v.size() != size) {
			v.resize(size);
		}
		cudaError_t errCUDA = cudaMemcpy(&(v[0]), data, size * sizeof(E), cudaMemcpyDeviceToHost) ;
		throw_cuda_error(errCUDA, __FILE__, __LINE__);
	}

	template <typename K>
	void copyFromDeviceToHost(K * v) {
		//ToDo sizeof(K) == sizeof(E)
		cudaError_t errCUDA = cudaMemcpy(v, data, size * sizeof(E), cudaMemcpyDeviceToHost) ;
		throw_cuda_error(errCUDA, __FILE__, __LINE__);
	}
};


//template <typename T, typename CT>
//void sort(ArrCuda<T> & arr, const CT & compare) {
//	thrust::device_ptr<T> tempPtr(arr.data);
//	thrust::sort(tempPtr, tempPtr + arr.size, compare);
//}


template <typename T, typename CT, typename ALLOC_TYPE>
void sort(DataBuffer<T> & arr, const CT & compare, ALLOC_TYPE alloc) {
	thrust::device_ptr<T> tempPtr(arr.data);
	thrust::sort(thrust::cuda::par(alloc), tempPtr, tempPtr + arr.size, compare);
}

template <typename T, typename PRED, typename ALLOC_TYPE=fallback_allocator>
__host__ DataBufferCuda<T> remove_if(DataBuffer<T> & arr, PRED pred) {
	ALLOC_TYPE alloc;
	thrust::device_ptr<T> tempPtr(arr.data);
	auto newEndIt = thrust::remove_if(thrust::cuda::par(alloc), tempPtr, tempPtr + arr.size, pred);
	int count = thrust::distance(tempPtr, newEndIt);
	DataBufferCuda<T> newArr;
	newArr.data = arr.data;
	newArr.size = count;
	return newArr;
}




//template <typename T, typename CT>
//void sort(ArrCuda<T> & arr, const CT & compare) {
//	thrust::device_ptr<T> tempPtr(arr.data);
//	fallback_allocator alloc;
//	thrust::sort(thrust::cuda::par(alloc), tempPtr, tempPtr + arr.size, compare);
//}



/*template <typename T, typename PRED, typename ALLOC_TYPE=fallback_allocator>
__host__ ArrCuda<T> remove_if(Arr<T> & arr, PRED pred) {
	ALLOC_TYPE alloc;
	thrust::device_ptr<T> tempPtr(arr.data);
	auto newEndIt = thrust::remove_if(thrust::cuda::par(alloc), tempPtr, tempPtr + arr.size, pred);
	int count = thrust::distance(tempPtr, newEndIt);
	ArrCuda<T> newArr;
	newArr.data = arr.data;
	newArr.size = count;
	return newArr;
}*/

/*template <typename T, typename PRED, typename ALLOC_TYPE>
__host__ ArrCuda<T> remove_if(Arr<T> & arr, PRED pred) {
	ALLOC_TYPE alloc;
	thrust::device_ptr<T> tempPtr(arr.data);
	auto newEndIt = thrust::remove_if(thrust::cuda::par(alloc), tempPtr, tempPtr + arr.size, pred);
	int count = thrust::distance(tempPtr, newEndIt);
	ArrCuda<T> newArr;
	newArr.data = arr.data;
	newArr.size = count;
	return newArr;
}*/


#endif // !ARRAY_CUH

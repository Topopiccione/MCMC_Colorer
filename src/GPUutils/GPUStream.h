#pragma once
#include <cinttypes>

#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

class GPUStream {

public:

	GPUStream( uint32_t n );
	~GPUStream();

	uint32_t		numThreads;
	cudaStream_t	*	streams;

};

#pragma once

#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

class GPUStream {

public:

	GPUStream(int n);
	~GPUStream();

	int numThreads;
	cudaStream_t * streams;

};

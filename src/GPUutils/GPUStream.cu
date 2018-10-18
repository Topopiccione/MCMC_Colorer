// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

#include "GPUStream.h"

GPUStream::GPUStream( uint32_t n ) : numThreads( n ) {

    streams = new cudaStream_t[numThreads];

    for (uint32_t i = 0; i < numThreads; i++)
        cudaStreamCreate(&streams[i]);
		//cudaStreamCreateWithFlags( &streams[i], cudaStreamNonBlocking	);
}

GPUStream::~GPUStream() {

    for (uint32_t i = 0; i < numThreads; i++)
        cudaStreamDestroy(streams[i]);

    delete[] streams;
}

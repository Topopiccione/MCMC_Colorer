#pragma once
#include <iostream>

#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

// The first thing a main program should do is to create a (static?) instance of
// CudaCleaner. Its destructor will be the last function that will be invoked,
// therefore invoking the cudaDeviceReset function
// (this is not true if there are static objects, since their destruction and
// construction order are compiler dependant: this CudaCleaner stuff would
// certainly do fancy stuff in this case...) 

class CudaCleaner {
public:
    CudaCleaner();
    ~CudaCleaner();
};

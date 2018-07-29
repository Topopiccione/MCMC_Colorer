#pragma once
#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <random>
#include <curand.h>
#include <curand_kernel.h>

/**
 * Cumulative distribution function
 */
struct discreteDistribution_st {
	float* prob;
	unsigned int length;
	float normFactor;
};
typedef struct discreteDistribution_st* discreteDistribution;

/**
 * Exponential cumulative distribution function with param lambda
 */
struct expDiscreteDistribution_st {
	discreteDistribution_st CDF;
	float lambda;
};
typedef struct expDiscreteDistribution_st* expDiscreteDistribution;

namespace GPURand_k {
    __global__ void initCurand(curandState*, unsigned int, unsigned int);
    __device__ int discreteSampling(curandState *, discreteDistribution);
}

namespace CPURand {
    void createExpDistribution(expDiscreteDistribution, float, unsigned int);
    void discreteSampling(discreteDistribution, unsigned int*, unsigned int, unsigned);
}

class GPURand {
public:

	GPURand( int n, long seed );
	~GPURand();

	int						num;
	unsigned int			seed;
	cudaError_t             cuSts;
	curandStatus_t          curandSts;
	curandState         *   randStates;
	curandGenerator_t       gen;

};

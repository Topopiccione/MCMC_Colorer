#pragma once
#include <iostream>
#include <memory>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "graph/graph.h"
#include "coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"

template<typename nodeW, typename edgeW>
class ColoringMCMC : public Colorer<nodeW, edgeW> {
public:

	ColoringMCMC( Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params );
	~ColoringMCMC();

	void			run();

	Coloring	*	getColoringGPU();

protected:
	uint32_t		nnodes;
	uint32_t		numOfColors;

	std::unique_ptr<int[]> coloring_h;

	//dati del grafo
	const GraphStruct<nodeW, edgeW>	* const	graphStruct_d;
	std::unique_ptr<Coloring> outColoring_d;

	// Strutture della colorazione [da uniformare a luby]
	col_sz			nCol;
	col				*C, *C1_d, *C2_d;
	col				*colNeigh;

	uint32_t		threadId;

	ColoringMCMCParams param;
	expDiscreteDistribution_st dist;

	cudaError_t		cuSts;
	cudaEvent_t		start, stop;
	uint32_t		numThreads;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
	curandState *	randStates;
	curandState *	states;		// da eliminare

	void			printgraph();

	void			convert_to_standard_notation();

};


namespace ColoringMCMC_k {
__global__ void initColoring(curandState*, expDiscreteDistribution_st *, col*, col_sz, node_sz);
template<typename nodeW, typename edgeW>
__global__ void drawNewColoring(unsigned int*, curandState*, const GraphStruct<nodeW, edgeW>*, col*, col*, col*);
__inline__ __host__ __device__ node_sz checkConflicts(node, node_sz, node*, col*);
__inline__  __device__ col newFreeColor(curandState*, node, node_sz, node*, col*, col*);
__inline__  __device__ col fixColor(curandState*, node*, col);
__device__ int discreteSampling(curandState *,discreteDistribution_st *);
template<typename nodeW, typename edgeW>
__global__ void print_graph_k( uint32_t nnodes, const node_sz * const cumulSize, const node * const neighs );
}

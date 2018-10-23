#pragma once
#include <iostream>
#include <memory>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#include "graph/graph.h"
#include "coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"

template<typename nodeW, typename edgeW>
class ColoringMCMC : public Colorer<nodeW, edgeW> {
public:

	ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params);
	~ColoringMCMC();

	void			run();

protected:
	uint32_t		nnodes;
	uint32_t		nedges;
	uint32_t		numOfColors;

	ColoringMCMCParams param;
	uint32_t		rip = 0;
	float			divider;

	//dati del grafo
	const GraphStruct<nodeW, edgeW>	* const	graphStruct_d;

	int				counter;
	uint32_t	*	counter_h; // lo spazio occupato può essere grande quanto blocksPerGrid_half_edges se si usa solo la somma parallela
	uint32_t	*	counter_d;

	int				newCounter;
	uint32_t	*	newCounter_h;
	uint32_t	*	newCounter_d;

	float		result, random;

	uint32_t	*	temp;
	uint32_t	*	coloring_d;			// each element denotes a color
	uint32_t	*	newColoring_d;		// each element denotes a new color

	float		probColoring;
	float	*	probColoring_h;
	float	*	probColoring_d;			// each element denotes a probability for a color

	float		probNewColoring;
	float	*	probNewColoring_h;
	float	*	probNewColoring_d;		// each element denotes a probability for a new color

	bool	*	colorsChecker_d;
	uint32_t *	orderedColors_d;

	uint32_t	*	statsFreeColors_h;
	uint32_t	*	statsFreeColors_d;
	uint32_t	statsFreeColors_max, statsFreeColors_min, statsFreeColors_avg;

	uint32_t		threadId;

	cudaError_t		cuSts;
	uint32_t		numThreads;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
	dim3			blocksPerGrid_edges;
	dim3			blocksPerGrid_half_edges;
	curandState *	randStates;
};


namespace ColoringMCMC_k {
	__global__ void initColoring(uint32_t nnodes, uint32_t * coloring_d, float divider, curandState * states);
	__global__ void conflictChecker(uint32_t nedges, uint32_t * counter_d, uint32_t * coloring_d, node_sz * edges);
	//template <uint32_t blockSize> 
	__global__ void sumReduction(uint32_t nedges, uint32_t * counter_d);
	//template <uint32_t blockSize> 
	__device__ void warpReduction(volatile int *sdata, uint32_t tid, uint32_t blockSize);
	__global__ void selectNewColoring(uint32_t nedges, uint32_t * newColoring_d, float * probNewColoring, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * orderedColors_d, curandState * states, float epsilon, uint32_t * statsFreeColors_d);
	__global__ void lookOldColoring(uint32_t nedges, float * probColoring_d, col_sz nCol, uint32_t * newColoring_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon);
}

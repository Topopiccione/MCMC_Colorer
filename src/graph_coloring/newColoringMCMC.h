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
class NewColoringMCMC : public Colorer<nodeW, edgeW> {
public:

	NewColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params);
	~NewColoringMCMC();

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
	uint32_t	*	counter_h;
	uint32_t	*	counter_d;

	int				newCounter;
	uint32_t	*	newCounter_h;
	uint32_t	*	newCounter_d;

	float		result, random;

	uint32_t	*	coloring_d;			// each element denotes a color
	uint32_t	*	newColoring_d;		// each element denotes a new color

	float		probColoring;
	float	*	probColoring_h;
	float	*	probColoring_d;			// each element denotes a probability for a color

	float		probNewColoring;
	float	*	probNewColoring_h;
	float	*	probNewColoring_d;		// each element denotes a probability for a new color

	uint32_t		threadId;

	cudaError_t		cuSts;
	uint32_t		numThreads;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
	dim3			blocksPerGrid_edges;
	curandState *	randStates;
};


namespace NewColoringMCMC_k {
	__global__ void initColoring(uint32_t nnodes, uint32_t * coloring_d, float divider, curandState * states);
	__global__ void conflictCounter(uint32_t nedges, uint32_t * counter_d, uint32_t * coloring_d, const node_sz * const edges);
	__global__ void selectNewColoring(uint32_t nedges, uint32_t * newColoring_d, float * probNewColoring, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, curandState * states, float epsilon);
	__global__ void lookOldColoring(uint32_t nedges, float * probColoring_d, col_sz nCol, uint32_t * newColoring_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, float epsilon);
	__global__ void changeColoring(uint32_t nnodes, uint32_t * newColoring_d, uint32_t * coloring_d);
}

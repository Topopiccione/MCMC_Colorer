#pragma once
#include <iostream>
#include <fstream>
#include <ctime>
#include <memory>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "graph/graph.h"
#include "coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"
#include "easyloggingpp/easylogging++.h"

// #define PRINTHISTOGRAM

#define TABOO
#define TAIL_CUTTING

/**
* choose one to indicate how to initialize the colors
*/
#define STANDARD_INIT
//#define DISTRIBUTION_LINE_INIT
//#define DISTRIBUTION_EXP_INIT

/**
* choose one to indicate the desired colorer
*/
// #define STANDARD
// #define COLOR_DECREASE_LINE
// #define COLOR_DECREASE_EXP
// #define COLOR_BALANCE_LINE
// #define COLOR_BALANCE_EXP
#define COLOR_BALANCE_DYNAMIC_DISTR

//#define HASTINGS

template<typename nodeW, typename edgeW>
class ColoringMCMC : public Colorer<nodeW, edgeW> {
public:

	ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params);
	~ColoringMCMC();

	void			run(int iteration);
	void setDirectoryPath(std::string directory) { this->directory = directory; }

protected:
	uint32_t		nnodes;
	float			prob;
	uint32_t		numOfColors;

	ColoringMCMCParams param;
	uint32_t		rip = 0;
	bool			maxIterReached = false;

	//dati del grafo
	const GraphStruct<nodeW, edgeW>	* const	graphStruct_d;

	int				conflictCounter;
	int				conflictCounterStar;

	float			stdDev;
	float			stdDevStar;
	void			calcStdDev(float &std, uint32_t	* col_d);

	//int *ret;

	uint32_t	*	conflictCounter_h;
	uint32_t	*	conflictCounter_d;


	float			result, random;

	uint32_t	*	coloring_d;			// each element denotes a color
	uint32_t	*	starColoring_d;		// each element denotes a new color
	uint32_t	*	switchPointer;

	uint32_t	*	taboo_d;			// each element denotes a color

	float			p;
	float		*	q_h;
	float		*	q_d;				// each element denotes a probability for a color

	float			pStar;
	float		*	qStar_h;
	float		*	qStar_d;			// each element denotes a probability for a new color

	bool		*	colorsChecker_d;

	float		*	probDistributionLine_d;
	float		*	probDistributionExp_d;
	float		*	probDistributionDynamic_d;

	uint32_t	*	orderedIndex_h;
	uint32_t	*	orderedIndex_d;

	// STATS
	uint32_t	*	coloring_h;			// each element denotes a color
	uint32_t	*	statsColors_h;		// used to get differents data from gpu memory
	uint32_t	*	statsColors_d;
	uint32_t	*	statsFreeColors_d;	// used to see free colors for nodes

	uint32_t		threadId;

	cudaError_t		cuSts;
	uint32_t		numThreads;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
	dim3			blocksPerGrid_nCol;
	dim3			blocksPerGrid_half;
	curandState *	randStates;

	void			calcConflicts(int &conflictCounter, uint32_t * coloring_d);
	void			getStatsFreeColors();
	void			getStatsNumColors(std::string prefix);
	void			calcProbs();

	// void			__printMemAlloc();
	void			__customPrintRun0_start(int iteration);
	void			__customPrintRun1_init();
	void			__customPrintRun2_conflicts(bool isTailCutting);
	void			__customPrintRun3_newConflicts();
	void			__customPrintRun5();
	void			__customPrintRun6_change();
	void			__customPrintRun7_end();

	std::clock_t start;
	double duration;

	std::ofstream logFile, colorsFile;
	std::string directory;
};


namespace ColoringMCMC_k {
	__global__ void initDistributionLine(float nCol, float denom, float lambda, float * probDistributionLine_d);
	__global__ void initDistributionExp(float nCol, float denom, float lambda, float * probDistributionExp_d);

	__global__ void initColoring(uint32_t nnodes, uint32_t * coloring_d, float nCol, curandState * states);
	__global__ void initColoringWithDistribution(uint32_t nnodes, uint32_t * coloring_d, float nCol, float * probDistribution_d, curandState * states);

	__global__ void selectStarColoring(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, curandState * states, float epsilon, uint32_t * statsFreeColors_d);
	__global__ void selectStarColoringDecrease(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistributionLine_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d);
	__global__ void selectStarColoringBalance(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistributionLine_d, uint32_t * orderedIndex_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d);
	__global__ void selectStarColoringBalanceDynamic(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistributionLine_d, uint32_t * orderedIndex_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d);
	__global__ void genDynamicDistribution(float * probDistributionDynamic_d, uint32_t nCol, uint32_t nnodes, uint32_t * statsColors_d);

	__global__ void tailCutting(uint32_t nnodes, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, int conflictCounter, uint32_t * conflictCounter_d, uint32_t * orderedIndex_d);
	__global__ void conflictCounter(uint32_t nnodes, uint32_t * conflictCounter_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs);
	__global__ void sumReduction(uint32_t n, float * conflictCounter_d);
	__device__ void warpReduction(volatile float *sdata, uint32_t tid, uint32_t blockSize);

	__global__ void logarithmer(uint32_t nnodes, float * values);
	__global__ void lookOldColoring(uint32_t nnodes, uint32_t * starColoring_d, float * q_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon);
}

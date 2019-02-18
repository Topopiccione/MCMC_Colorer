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
#ifdef WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "graph/graph.h"
#include "coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"

#define STATS
//#define PRINTS
#define WRITE

#define FIXED_N_COLORS
//#define DYNAMIC_N_COLORS		

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
#define STANDARD
//#define COLOR_DECREASE_LINE
//#define COLOR_DECREASE_EXP				
//#define COLOR_BALANCE_LINE
//#define COLOR_BALANCE_EXP

template<typename nodeW, typename edgeW>
class ColoringMCMC : public Colorer<nodeW, edgeW> {
public:

	ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params);
	~ColoringMCMC();

	void			run(int iteration);

#ifdef  WRITE
	void setDirectoryPath(std::string directory) { this->directory = directory; }
#endif //  WRITE


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

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE) || defined(COLOR_BALANCE_LINE)
	float		*	probDistributionLine_d;
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE || COLOR_BALANCE_LINE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP) || defined(COLOR_BALANCE_EXP)
	float		*	probDistributionExp_d;
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP || COLOR_BALANCE_EXP

#if defined(COLOR_BALANCE_EXP) || defined(TAIL_CUTTING)
	uint32_t	*	orderedIndex_h;
	uint32_t	*	orderedIndex_d;
#endif // COLOR_BALANCE_EXP || TAIL_CUTTING

	// STATS
	uint32_t	*	coloring_h;			// each element denotes a color
	uint32_t	*	statsColors_h;		// used to get differents data from gpu memory
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
	void			getStatsNumColors(char * prefix);
	void			calcProbs();

#if defined(PRINTS) || defined(WRITE)
	void			__customPrintConstructor0_start();
	void			__customPrintConstructor1_end();
	void			__customPrintRun0_start(int iteration);
	void			__customPrintRun1_init();
	void			__customPrintRun2_conflicts();
	void			__customPrintRun3_newConflicts();
	void			__customPrintRun4();
	void			__customPrintRun5();
	void			__customPrintRun6_change();
	void			__customPrintRun7_end();
#endif

	std::clock_t start;
	double duration;
#ifdef WRITE
	std::ofstream logFile, resultsFile, colorsFile;
	std::string directory;
#endif //WRITE

};


namespace ColoringMCMC_k {

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE) || defined(COLOR_BALANCE_LINE)
	__global__ void initDistributionLine(float nCol, float denom, float lambda, float * probDistributionLine_d);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE || COLOR_BALANCE_LINE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP) || defined(COLOR_BALANCE_EXP)
	__global__ void initDistributionExp(float nCol, float denom, float lambda, float * probDistributionExp_d);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP || COLOR_BALANCE_EXP

#ifdef STANDARD_INIT
	__global__ void initColoring(uint32_t nnodes, uint32_t * coloring_d, float nCol, curandState * states);
#endif // STANDARD_INIT
#if defined(DISTRIBUTION_LINE_INIT) || defined(DISTRIBUTION_EXP_INIT)
	__global__ void initColoringWithDistribution(uint32_t nnodes, uint32_t * coloring_d, float nCol, float * probDistribution_d, curandState * states);
#endif // DISTRIBUTION_LINE_INIT || DISTRIBUTION_EXP_INIT

	__global__ void logarithmer(uint32_t nnodes, float * values);

	__global__ void tailCutting(uint32_t nnodes, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, int conflictCounter, uint32_t * conflictCounter_d, uint32_t * orderedIndex_d);

	__global__ void conflictCounter(uint32_t nnodes, uint32_t * conflictCounter_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs);
	__global__ void sumReduction(uint32_t n, float * conflictCounter_d);
	__device__ void warpReduction(volatile float *sdata, uint32_t tid, uint32_t blockSize);

#ifdef STANDARD
	__global__ void selectStarColoring(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, curandState * states, float epsilon, uint32_t * statsFreeColors_d);
#endif // STANDARD
#if defined(COLOR_DECREASE_LINE) || defined(COLOR_DECREASE_EXP)
	__global__ void selectStarColoringDecrease(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistributionLine_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d);
#endif // COLOR_DECREASE_LINE || COLOR_DECREASE_EXP
#if defined(COLOR_BALANCE_LINE) || defined(COLOR_BALANCE_EXP)
	__global__ void selectStarColoringBalance(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistributionLine_d, uint32_t * orderedIndex_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d);
#endif // COLOR_BALANCE_LINE || COLOR_BALANCE_EXP

	__global__ void lookOldColoring(uint32_t nnodes, uint32_t * starColoring_d, float * q_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon);
}

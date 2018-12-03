// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params) :
	Colorer<nodeW, edgeW>(inGraph_d),
	graphStruct_d(inGraph_d->getStruct()),
	nnodes(inGraph_d->getStruct()->nNodes),
	nedges(inGraph_d->getStruct()->nCleanEdges),
	randStates(randStates),
	numOfColors(0),
	threadId(0),
	param(params) {

	// configuro la griglia e i blocchi
	numThreads = 32;
	threadsPerBlock = dim3(numThreads, 1, 1);
	blocksPerGrid = dim3((nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_nCol = dim3((param.nCol + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half = dim3(((nnodes / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_edges = dim3((nedges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half_edges = dim3(((nedges / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

	//https://stackoverflow.com/questions/34356768/managing-properly-an-array-of-results-that-is-larger-than-the-memory-available-a
	//colorsChecker_d e orderedColors_d

	size_t total_mem, free_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "total mem: " << total_mem << " free mem:" << free_mem << std::endl;

	int tot = nnodes * sizeof(uint32_t) * 3;
	std::cout << "nnodes * sizeof(uint32_t): " << nnodes * sizeof(uint32_t) << " X 3" << std::endl;
	tot += nnodes * sizeof(float) * 2;
	std::cout << "nnodes * sizeof(float): " << nnodes * sizeof(float) << " X 2" << std::endl;
	tot += nedges * sizeof(uint32_t);
	std::cout << "nedges * sizeof(uint32_t): " << nedges * sizeof(uint32_t) << " X 1" << std::endl;
	tot += nnodes * param.nCol * sizeof(bool);
	std::cout << "nnodes * param.nCol * sizeof(bool): " << nnodes * param.nCol * sizeof(bool) << " X 1" << std::endl;
	tot += nnodes * param.nCol * sizeof(uint32_t);
	std::cout << "nnodes * param.nCol * sizeof(uint32_t): " << nnodes * param.nCol * sizeof(uint32_t) << " X 1" << std::endl;
	std::cout << "TOTALE: " << tot << " bytes" <<std::endl;

	cuSts = cudaMalloc((void**)&coloring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&starColoring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	q_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&q_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
	qStar_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&qStar_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter_h = (uint32_t *)malloc(nedges * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&conflictCounter_d, nedges * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	cuSts = cudaMalloc((void**)&colorsChecker_d, nnodes * param.nCol * sizeof(bool));	cudaCheck(cuSts, __FILE__, __LINE__);
	//std::cout << "colorsChecker_d: " << nnodes * param.nCol * sizeof(bool) << std::endl;
#ifdef STANDARD
	cuSts = cudaMalloc((void**)&orderedColors_d, nnodes * param.nCol * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	//std::cout << "orderedColors_d:" << nnodes * param.nCol * sizeof(uint32_t) << std::endl;
#endif // STANDARD
#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE_CUMULATIVE)
	cuSts = cudaMalloc((void**)&probDistributionLine_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
	cuSts = cudaMalloc((void**)&probDistributionExp_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP_CUMULATIVE


#ifdef STATS
	coloring_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	statsColors_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&statsFreeColors_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif

	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << "total mem: " << total_mem << " free mem:" << free_mem << std::endl;

#ifdef PRINTS
	std::cout << std::endl << "ColoringMCMC GPU" << std::endl;
	std::cout << "numCol: " << params.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	std::cout << "startingNCol: " << params.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	std::cout << "epsilon: " << params.epsilon << std::endl;
	std::cout << "lambda: " << params.lambda << std::endl;
	std::cout << "ratioFreezed: " << params.ratioFreezed << std::endl;
	std::cout << "maxRip: " << params.maxRip << std::endl << std::endl;
#endif // PRINTS

#ifdef WRITE
	logFile.open(std::to_string(nnodes) + "-" + std::to_string(nedges) + "-logFile.txt");
	resultsFile.open(std::to_string(nnodes) + "-" + std::to_string(nedges) + "-resultsFile.txt");
	colorsFile.open(std::to_string(nnodes) + "-" + std::to_string(nedges) + "-colorsFile.txt");

	logFile << "numCol: " << params.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	logFile << "startingNCol: " << params.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	logFile << "epsilon: " << params.epsilon << std::endl;
	logFile << "lambda: " << params.lambda << std::endl;
	logFile << "ratioFreezed: " << params.ratioFreezed << std::endl;
	logFile << "maxRip: " << params.maxRip << std::endl << std::endl;

	resultsFile << "numCol " << params.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	resultsFile << "startingNCol " << params.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	resultsFile << "epsilon " << params.epsilon << std::endl;
	resultsFile << "lambda " << params.lambda << std::endl;
	resultsFile << "ratioFreezed " << params.ratioFreezed << std::endl;
	resultsFile << "maxRip " << params.maxRip << std::endl;
#endif // WRITE
}

template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::~ColoringMCMC() {
	cuSts = cudaFree(coloring_d); 					cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(starColoring_d); 				cudaCheck(cuSts, __FILE__, __LINE__);

	cuSts = cudaFree(colorsChecker_d); 				cudaCheck(cuSts, __FILE__, __LINE__);
#ifdef STANDARD
	cuSts = cudaFree(orderedColors_d); 				cudaCheck(cuSts, __FILE__, __LINE__);
#endif // STANDARD
#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE_CUMULATIVE)
	cuSts = cudaFree(probDistributionLine_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
	cuSts = cudaFree(probDistributionExp_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP_CUMULATIVE

	cuSts = cudaFree(conflictCounter_d); 			cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(q_d); 							cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(qStar_d);						cudaCheck(cuSts, __FILE__, __LINE__);

#ifdef STATS
	free(coloring_h);
#endif

	free(conflictCounter_h);
	free(q_h);
	free(qStar_h);

#ifdef STATS
	free(statsColors_h);
	cuSts = cudaFree(statsFreeColors_d);			cudaCheck(cuSts, __FILE__, __LINE__);
#endif // STATS

#ifdef WRITE
	logFile.close();
	resultsFile.close();
	colorsFile.close();
#endif // WRITE
}

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE_CUMULATIVE)
__global__ void ColoringMCMC_k::initDistributionLine(float nCol, float denom, float lambda, float * probDistributionLine_d) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nCol)
		return;

	probDistributionLine_d[idx] = (float)(nCol - lambda * idx) / denom;
	//probDistributionLine_d[idx] = (float)(lambda * idx) / denom;
}
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE

#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
__global__ void ColoringMCMC_k::initDistributionExp(float nCol, float denom, float lambda, float * probDistributionExp_d) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nCol)
		return;

	probDistributionExp_d[idx] = exp(-lambda * idx) / denom;
}
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP_CUMULATIVE

/**
* Set coloring_d with random colors
*/
#ifdef STANDARD_INIT
__global__ void ColoringMCMC_k::initColoring(uint32_t nnodes, uint32_t * coloring_d, float nCol, curandState * states) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	float randnum = curand_uniform(&states[idx]);

	int color = (int)(randnum * nCol);

	coloring_d[idx] = color;
	//coloring_d[idx] = 0;
}
#endif // STANDARD_INIT

/**
* Set coloring_d with random colors
*/
#if defined(DISTRIBUTION_LINE_INIT) || defined(DISTRIBUTION_EXP_INIT)
__global__ void ColoringMCMC_k::initColoringWithDistribution(uint32_t nnodes, uint32_t * coloring_d, float nCol, float * probDistribution_d, curandState * states) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	float randnum = curand_uniform(&states[idx]);

	int color = 0;
	float threshold = 0;
	while (threshold < randnum)
	{
		threshold += probDistribution_d[color];
		color++;
	}

	/*if (idx == 0) {
		float a = 0;
		for (int i = 0; i < nCol; i++)
		{
			a += probDistribution_d[i];
			printf("parziale : %f\n", probDistribution_d[i]);
		}
		printf("totale : %f\n", a);
	}*/

	coloring_d[idx] = color - 1;
}
#endif // DISTRIBUTION_LINE_INIT

/**
* Apply logarithm to all values
*/
__global__ void ColoringMCMC_k::logarithmer(uint32_t nnodes, float * values) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	values[idx] = log(values[idx]);
}


/**
* For all the edges of the graph, set the value of conflictCounter_d to 0 or 1 if the nodes of the edge have the same color
*/
__global__ void ColoringMCMC_k::conflictChecker(uint32_t nedges, uint32_t * conflictCounter_d, uint32_t * coloring_d, node_sz * edges) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nedges)
		return;

	uint32_t idx0 = idx * 2;
	uint32_t idx1 = idx0 + 1;

	uint32_t node0 = edges[idx0];
	uint32_t node1 = edges[idx1];

	uint32_t col0 = coloring_d[node0];
	uint32_t col1 = coloring_d[node1];

	conflictCounter_d[idx] = col0 == col1;
}

/**
* Parallel sum reduction inside a single warp
*/
__device__ void ColoringMCMC_k::warpReduction(volatile float *sdata, uint32_t tid, uint32_t blockSize) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

/*
* Parallel sum reduction inside a block and write the partial result in conflictCounter_d.
* At the end, conflictCounter_d have n partial results for the first n positions where n is the number of blocks called.

* refs: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
__global__ void ColoringMCMC_k::sumReduction(uint32_t nedges, float * conflictCounter_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nedges)
		return;

	extern	__shared__ float sdata[];

	uint32_t tid = threadIdx.x;
	uint32_t blockSize = blockDim.x;
	uint32_t i = (blockSize * 2) * blockIdx.x + tid;

	sdata[tid] = conflictCounter_d[i] + conflictCounter_d[i + blockSize];

	/*uint32_t gridSize = (blockSize * 2) * gridDim.x;
	sdata[tid] = 0;
	while (i < nedges) {
		sdata[tid] += conflictCounter_d[i] + conflictCounter_d[i + blockSize];
		i += gridSize;
	}*/
	__syncthreads();

	//useless for blocks of dim <= 64
	if (blockSize >= 512)
	{
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32)
		//ColoringMCMC_k::warpReduction<blockSize>(sdata, tid);
		ColoringMCMC_k::warpReduction(sdata, tid, blockSize);

	if (tid == 0)
		conflictCounter_d[blockIdx.x] = sdata[0];
}

/**
* For every node, look at neighbors and select a new color.
* This will be write in starColoring_d and the probability of the chosen color will be write in qStar_d
*/
#ifdef STANDARD
__global__ void ColoringMCMC_k::selectStarColoring(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * orderedColors_d, curandState * states, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set to 1 or 0 the colors occupied from the neighbors
	for (int i = 0; i < nneighs; i++)
		colorsChecker[coloring_d[neighs[index + i]]] = 1;

	uint32_t * orderedColors = &(orderedColors_d[idx * nCol]);	//array containing previously occupied colors and then free ones
	uint32_t Zp = nCol, Zn = 0;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
	{
		orderedColors[Zn] += i * (1 - (1 - colorsChecker[i]));
		orderedColors[Zp - 1] += i * (1 - colorsChecker[i]);
		Zn += colorsChecker[i];
		Zp -= 1 - colorsChecker[i];
	}
	Zp = nCol - Zn;

#ifdef STATS
	statsFreeColors_d[idx] = Zp;
#endif // STATS

	if (!Zp)													//manage exception of no free colors
	{
#ifdef FIXED_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // DYNAMIC_N_COLORS
		return;
	}

	float randnum = curand_uniform(&states[idx]);				//random number

	float threshold;
	uint32_t selectedIndex = 0;									//selected index for orderedColors to select the new color
	if (colorsChecker[nodeCol])									//if node color is used by neighbors
	{
		threshold = 1 - epsilon * Zn;							//threshold used to randomly determine whether to extract a free color or a busy one
		if (randnum < threshold)
		{
			selectedIndex = ((randnum * Zp) / threshold) + Zn;	//get the selected index
			qStar_d[idx] = (1 - epsilon * Zn) / Zp;				//save the probability of the color chosen
		}
		else
		{
			selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);	//get the selected index
			qStar_d[idx] = epsilon;								//save the probability of the color chosen
		}
		starColoring_d[idx] = orderedColors[selectedIndex];		//save the new color
	}
	else
	{
		threshold = 1 - epsilon * (nCol - 1);					//threshold used to randomly determine whether to extract a occupied color
																//or keep the same
		if (randnum < threshold)
		{
			starColoring_d[idx] = nodeCol;						//keep the same color
			qStar_d[idx] = 1 - ((nCol - 1) * epsilon);	//save the probability of the color chosen
		}
		else
		{
			selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);	//get the selected index
			starColoring_d[idx] = orderedColors[selectedIndex];	//save the new color
			qStar_d[idx] = epsilon;						//save the probability of the color chosen
		}
	}
}
#endif // STANDARD

#ifdef COLOR_BALANCE_ON_NODE_CUMULATIVE
__global__ void ColoringMCMC_k::selectStarColoringBalanceOnNode_cumulative(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, curandState * states, float partition, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to count how many times a color is used from the neighbors
	for (int i = 0; i < nneighs; i++) {
		colorsChecker[coloring_d[neighs[index + i]]]++;
	}

	if (colorsChecker[nodeCol] > 0) {
		float randnum = curand_uniform(&states[idx]);				//random number

		uint32_t Zp = 0;											//number of free colors (p) and occupied colors (n)
		for (int i = 0; i < nCol; i++)
		{
			//Zn += colorsChecker[i] != 0;
			if (colorsChecker[i] == 0)
				Zp++;
		}

		float threshold = 0;
		float q;
		int i = 0;
		do {
			q = (1 - ((float)colorsChecker[i] / (float)nneighs)) / ((float)nCol - 1);
			q /= partition;
			if (colorsChecker[i] == 0)
				q += ((partition - 1) / partition) / Zp;

			threshold += q;
			i++;
		} while (threshold < randnum);

		qStar_d[idx] = q;											//save the probability of the color chosen
		starColoring_d[idx] = i - 1;
	}
	else {
		qStar_d[idx] = (1 - ((float)colorsChecker[nodeCol] / (float)nneighs)) / ((float)nCol - 1);
		starColoring_d[idx] = nodeCol;
	}
}
#endif // COLOR_BALANCE_ON_NODE_CUMULATIVE

#if defined(COLOR_DECREASE_LINE_CUMULATIVE) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
__global__ void ColoringMCMC_k::selectStarColoringDecrease_cumulative(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float * probDistribution_d, curandState * states, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set if a color is used from the neighbors
	for (int i = 0; i < nneighs; i++) {
		colorsChecker[coloring_d[neighs[index + i]]] = 1;
	}

	float reminder = 0;
	uint32_t Zn = 0, Zp = nCol;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
	{
		Zn += colorsChecker[i] != 0;
		reminder += (colorsChecker[i] != 0) * (probDistribution_d[i] - epsilon);
	}
	Zp -= Zn;
	reminder /= Zp;

	if (!Zp)													//manage exception of no free colors
	{
#ifdef FIXED_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // DYNAMIC_N_COLORS
		return;
	}

	int i = 0;
	float q;
	float threshold = 0;
	float randnum = curand_uniform(&states[idx]);				//random number
	if (colorsChecker[nodeCol])									//if node color is used by neighbors
	{
		do {
			q = (probDistribution_d[i] + reminder) * (!colorsChecker[i]) + (epsilon) * (colorsChecker[i]);
			threshold += q;
			i++;
		} while (threshold < randnum);
	}
	else
	{
		do {
			q = (1.0f - (nCol - 1) * epsilon) * (nodeCol == i) + (epsilon) * (nodeCol != i);
			threshold += q;
			i++;
		} while (threshold < randnum);
	}
	qStar_d[idx] = q;											//save the probability of the color chosen
	if ((i - 1) >= nCol)										//TEMP
		i = nCol;
	starColoring_d[idx] = i - 1;
}
#endif // COLOR_DECREASE_LINE_CUMULATIVE || COLOR_DECREASE_EXP_CUMULATIVE

/**
* For every node, look at neighbors.
* The probability of the old color will be write in probColoring_d
*/
__global__ void ColoringMCMC_k::lookOldColoring(uint32_t nnodes, float * q_d, col_sz nCol, uint32_t * starColoring_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color
	uint32_t nodeStarCol = starColoring_d[idx];					//node new color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set to 1 or 0 the colors occupied from the neighbors
	for (int i = 0; i < nneighs; i++)
		colorsChecker[starColoring_d[neighs[index + i]]] = 1;

	uint32_t Zp = nCol, Zn = 0;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
		Zn += colorsChecker[i];
	Zp = nCol - Zn;

	if (!Zp)													//manage exception of no free colors
	{
		q_d[idx] = 1;
		return;
	}

	if (colorsChecker[nodeStarCol])								//if node color is used by neighbors
	{
		if (!colorsChecker[nodeCol])
			q_d[idx] = (1 - epsilon * Zn) / Zp;			//save the probability of the old color
		else
			q_d[idx] = epsilon;							//save the probability of the old color
	}
	else
	{
		if (nodeStarCol == nodeCol)
			q_d[idx] = 1 - ((nCol - 1) * epsilon);		//save the probability of the old color
		else
			q_d[idx] = epsilon;							//save the probability of the old color
	}
}

/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::run() {
	start = std::clock();

	cuSts = cudaMemset(coloring_d, 0, nnodes * sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE_CUMULATIVE)
	float denomL = 0;
	for (int i = 0; i < param.nCol; i++)
	{
		denomL += param.nCol - param.lambda * i;
		//denomL += param.lambda * i;
	}
	ColoringMCMC_k::initDistributionLine << < blocksPerGrid_nCol, threadsPerBlock >> > (param.nCol, denomL, param.lambda, probDistributionLine_d);
	cudaDeviceSynchronize();
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
#ifdef FIXED_N_COLORS
	float denomE = 0;
	for (int i = 0; i < param.nCol; i++)
	{
		denomE += exp(-param.lambda * i);
	}
	ColoringMCMC_k::initDistributionExp << < blocksPerGrid_nCol, threadsPerBlock >> > (param.nCol, denomE, param.lambda, probDistributionExp_d);
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
	float denomE = 0;
	for (int i = 0; i < param.startingNCol; i++)
	{
		denomE += exp(-param.lambda * i);
	}
	ColoringMCMC_k::initDistributionExp << < blocksPerGrid_nCol, threadsPerBlock >> > (param.startingNCol, denomE, param.lambda, probDistributionExp_d);
#endif // DYNAMIC_N_COLORS

	cudaDeviceSynchronize();
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE

#ifdef STANDARD_INIT
#ifdef FIXED_N_COLORS
	ColoringMCMC_k::initColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, randStates);
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
	ColoringMCMC_k::initColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.startingNCol, randStates);
#endif // DYNAMIC_N_COLORS
#endif // STANDARD_INIT

#ifdef DISTRIBUTION_LINE_INIT
	ColoringMCMC_k::initColoringWithDistribution << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, probDistributionLine_d, randStates);
#endif // DISTRIBUTION_LINE_INIT

#ifdef DISTRIBUTION_EXP_INIT
#ifdef FIXED_N_COLORS
	ColoringMCMC_k::initColoringWithDistribution << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, probDistributionExp_d, randStates);
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
	ColoringMCMC_k::initColoringWithDistribution << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.startingNCol, probDistributionExp_d, randStates);
#endif // DYNAMIC_N_COLORS
#endif // DISTRIBUTION_EXP_INIT
	cudaDeviceSynchronize();

#if defined(STATS) && (defined(PRINTS) || defined(WRITE))
#ifdef PRINTS
	std::cout << "COLORAZIONE INIZIALE" << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "COLORAZIONE INIZIALE" << std::endl;
#endif // WRITE

	getStatsNumColors("start_");

#ifdef PRINTS
	std::cout << std::endl << "end colorazione iniziale -------------------------------------------------------------------" << std::endl << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << std::endl << "end colorazione iniziale -------------------------------------------------------------------" << std::endl << std::endl;
#endif // WRITE
#endif // STATS && ( PRINTS || WRITE )

	do {

		rip++;

		calcConflicts(conflictCounter, coloring_d);

		if (conflictCounter == 0)
			break;

#ifdef PRINTS
		std::cout << "***** Tentativo numero: " << rip << std::endl;
		std::cout << "conflitti rilevati: " << conflictCounter << std::endl;
#endif // PRINTS
#ifdef WRITE
		logFile << "***** Tentativo numero: " << rip << std::endl;
		logFile << "conflitti rilevati: " << conflictCounter << std::endl;

		resultsFile << "iteration " << rip << std::endl;
		resultsFile << "iteration_" << rip << "_conflicts " << conflictCounter << std::endl;
#endif // WRITE

		cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));

#ifdef STANDARD
		cudaMemset(orderedColors_d, 0, nnodes * param.nCol * sizeof(uint32_t));
#ifdef FIXED_N_COLORS
		ColoringMCMC_k::selectStarColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, orderedColors_d, randStates, param.epsilon, statsFreeColors_d);
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
		ColoringMCMC_k::selectStarColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.startingNCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, orderedColors_d, randStates, param.epsilon, statsFreeColors_d);
		cudaDeviceSynchronize();
		//cuSts = cudaMemcpy(coloring_h, starColoring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		cuSts = cudaMemcpy(qStar_h, qStar_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		for (uint32_t i = 0; i < nnodes && param.startingNCol < param.nCol; i++)
		{


			//if (coloring_h[i] == param.startingNCol)
			if (qStar_h[i] == 1)
			{
				//param.startingNCol++;
				param.startingNCol += 1;
				i = nnodes;
			}

		}
		std::cout << "startingNCol = " << param.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
#endif // STANDARD

#ifdef COLOR_BALANCE_ON_NODE_CUMULATIVE
		float partition = 15;
		ColoringMCMC_k::selectStarColoringBalanceOnNode_cumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, randStates, partition, param.epsilon, statsFreeColors_d);
#endif // COLOR_BALANCE_ON_NODE_CUMULATIVE

#ifdef COLOR_DECREASE_LINE_CUMULATIVE
		ColoringMCMC_k::selectStarColoringDecrease_cumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, probDistributionLine_d, randStates, param.epsilon, statsFreeColors_d);
#endif // COLOR_DECREASE_LINE_CUMULATIVE

#ifdef COLOR_DECREASE_EXP_CUMULATIVE
#ifdef FIXED_N_COLORS
		ColoringMCMC_k::selectStarColoringDecrease_cumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, probDistributionExp_d, randStates, param.epsilon, statsFreeColors_d);
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
		ColoringMCMC_k::selectStarColoringDecrease_cumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.startingNCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, probDistributionExp_d, randStates, param.epsilon, statsFreeColors_d);
		cuSts = cudaMemcpy(qStar_h, qStar_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		for (uint32_t i = 0; i < nnodes && param.startingNCol < param.nCol; i++)
		{
			//if (coloring_h[i] == param.startingNCol)
			if (qStar_h[i] == 1)
			{
				//param.startingNCol++;
				param.startingNCol += 1;
				float denomE = 0;
				for (int i = 0; i < param.startingNCol; i++)
				{
					denomE += exp(-param.lambda * i);
				}
				ColoringMCMC_k::initDistributionExp << < blocksPerGrid_nCol, threadsPerBlock >> > (param.startingNCol, denomE, param.lambda, probDistributionExp_d);
				i = nnodes;
			}

		}
		std::cout << "startingNCol = " << param.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
#endif // COLOR_DECREASE_LINE_CUMULATIVE

		cudaDeviceSynchronize();

		/*cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
		ColoringMCMC_k::lookOldColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, q_d, param.nCol, starColoring_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, param.epsilon);
		cudaDeviceSynchronize();*/

#if defined(PRINTS) && defined(STATS)
		getStatsFreeColors();
#endif // PRINTS && ( STATS || WRITE )

		calcConflicts(conflictCounterStar, starColoring_d);

#ifdef PRINTS
		std::cout << "nuovi conflitti rilevati: " << conflictCounterStar << std::endl;
#endif // PRINTS
#ifdef WRITE
		logFile << "nuovi conflitti rilevati: " << conflictCounterStar << std::endl;
#endif // WRITE

#ifdef PRINTS
		/*cuSts = cudaMemcpy(qStar_h, qStar_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		cuSts = cudaMemcpy(q_h, q_d, blocksPerGrid_half.x * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		int numberOfEpsilonStar = 0, numberOfChangeColorStar = 0, numberOfSameColorStar = 0;
		int numberOfEpsilon = 0, numberOfChangeColor = 0, numberOfSameColor = 0;
		for (int i = 0; i < nnodes; i++)
		{
			if (qStar_h[i] == param.epsilon) {
				numberOfEpsilonStar++;
			}
			else if (qStar_h[i] == (1 - (param.nCol - 1) * param.epsilon)) {
				numberOfSameColorStar++;
			}
			else {
				numberOfChangeColorStar++;
			}

			if (q_h[i] == param.epsilon) {
				numberOfEpsilon++;
			}
			else if (q_h[i] == (1 - (param.nCol - 1) * param.epsilon)) {
				numberOfSameColor++;
			}
			else {
				numberOfChangeColor++;
			}
		}
		std::cout << "numberOfEpsilonStar: " << numberOfEpsilonStar << " numberOfChangeColorStar: " << numberOfChangeColorStar << " numberOfSameColorStar: " << numberOfSameColorStar << std::endl;
		std::cout << "numberOfEpsilon: " << numberOfEpsilon << " numberOfChangeColor: " << numberOfChangeColor << " numberOfSameColor: " << numberOfSameColor << std::endl;*/
#endif // PRINTS

		//calcProbs();

		//param.lambda = -numberOfChangeColorStar * log(param.epsilon);

		//result = param.lambda * (conflictCounter - conflictCounterStar) + p - pStar;
		//result = exp(result);

		//random = ((float)rand() / (float)RAND_MAX);

#ifdef PRINTS
		/*std::cout << "lambda: " << param.lambda << std::endl;
		std::cout << "probs star: " << pStar << " old:" << p << std::endl;
		std::cout << "left: " << param.lambda * (conflictCounter - conflictCounterStar) << " right:" << p - pStar << std::endl;
		std::cout << "result: " << result << std::endl;*/
		//std::cout << "random: " << random << std::endl;
#endif // PRINTS

		//if (random < result) {
#ifdef PRINTS
		std::cout << "CHANGE" << std::endl;
#endif // PRINTS
#ifdef WRITE
		logFile << "CHANGE" << std::endl;
#endif // WRITE

		switchPointer = coloring_d;
		coloring_d = starColoring_d;
		starColoring_d = switchPointer;
		//}

#if defined(STATS) && (defined(PRINTS) || defined(WRITE))
		//getStatsNumColors(true);
#endif // STATS && ( PRINTS || WRITE )

	} while (rip < param.maxRip);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

#if defined(STATS) && (defined(PRINTS) || defined(WRITE))
#ifdef PRINTS
	std::cout << "COLORAZIONE FINALE" << std::endl;
	std::cout << "Time " << duration << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "COLORAZIONE FINALE" << std::endl;
	logFile << "Time " << duration << std::endl;

	resultsFile << "time " << duration << std::endl;
#endif // WRITE

	getStatsNumColors("end_");

#ifdef PRINTS
	std::cout << std::endl << "end colorazione finale -------------------------------------------------------------------" << std::endl << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << std::endl << "end colorazione finale -------------------------------------------------------------------" << std::endl << std::endl;
#endif // WRITE
#endif // STATS && ( PRINTS || WRITE )
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::calcConflicts(int &conflictCounter, uint32_t * coloring_d) {
	ColoringMCMC_k::conflictChecker << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, conflictCounter_d, coloring_d, graphStruct_d->edges);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, (float*)conflictCounter_d);
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy(conflictCounter_h, conflictCounter_d, blocksPerGrid_half_edges.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter = 0;
	for (int i = 0; i < blocksPerGrid_half_edges.x; i++)
		conflictCounter += conflictCounter_h[i];
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::getStatsFreeColors() {
	cuSts = cudaMemcpy(statsColors_h, statsFreeColors_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
	statsFreeColors_max = statsFreeColors_avg = 0;
	statsFreeColors_min = param.nCol + 1;
	for (uint32_t i = 0; i < nnodes; i++) {
		uint32_t freeColors = statsColors_h[i];
		statsFreeColors_avg += freeColors;
		statsFreeColors_max = (freeColors > statsFreeColors_max) ? freeColors : statsFreeColors_max;
		statsFreeColors_min = (freeColors < statsFreeColors_min) ? freeColors : statsFreeColors_min;
	}
	statsFreeColors_avg /= (float)nnodes;
	std::cout << "Max Free Colors: " << statsFreeColors_max << " - Min Free Colors: " << statsFreeColors_min << " - AVG Free Colors: " << statsFreeColors_avg << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::getStatsNumColors(char * prefix) {

	cuSts = cudaMemcpy(coloring_h, coloring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
	memset(statsColors_h, 0, nnodes * sizeof(uint32_t));
	for (int i = 0; i < nnodes; i++)
	{
		statsColors_h[coloring_h[i]]++;
		//std::cout << i << " " << coloring_h[i] << std::endl;
	}
	int counter = 0;
	int max_i = 0, min_i = nnodes;
	int max_c = 0, min_c = nnodes;

#ifdef FIXED_N_COLORS
	int numberOfCol = param.nCol;
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
	int numberOfCol = param.startingNCol;
#endif // DYNAMIC_N_COLORS

	float average = 0, variance = 0, standardDeviation;
	//float cAverage = 0, cVariance = 0, cStandardDeviation;

	for (int i = 0; i < numberOfCol; i++)
	{
		if (statsColors_h[i] > 0) {
			counter++;
			if (statsColors_h[i] > max_c) {
				max_i = i;
				max_c = statsColors_h[i];
			}
			if (statsColors_h[i] < min_c) {
				min_i = i;
				min_c = statsColors_h[i];
			}
		}

		//cAverage += i * statsColors_h[i];
	}
	average = (float)nnodes / numberOfCol;
	//cAverage /= nnodes;
	for (int i = 0; i < numberOfCol; i++) {
		variance += pow((statsColors_h[i] - average), 2.f);
		//cVariance += i * pow((statsColors_h[i] - cAverage), 2.f);
	}
	variance /= numberOfCol;
	//cVariance /= nnodes;
	standardDeviation = sqrt(variance);
	//cStandardDeviation = sqrt(cVariance);

	int divider = (max_c / (param.nCol / 3) > 0) ? max_c / (param.nCol / 3) : 1;

#ifdef PRINTS
	//for (int i = 0; i < numberOfCol; i++)
		//std::cout << "Color " << i << " used " << statsColors_h[i] << " times" << std::endl;
	for (int i = 0; i < numberOfCol; i++)
	{
		std::cout << "Color " << i << " ";
		for (int j = 0; j < statsColors_h[i] / divider; j++)
		{
			std::cout << "*";
		}
		std::cout << std::endl;
	}
	std::cout << "Every * is " << divider << " nodes" << std::endl;
	std::cout << std::endl;

	std::cout << "Number of used colors is " << counter << " on " << numberOfCol << " available" << std::endl;
	std::cout << "Most used colors is " << max_i << " used " << max_c << " times" << std::endl;
	std::cout << "Least used colors is " << min_i << " used " << min_c << " times" << std::endl;
	std::cout << std::endl;
	std::cout << "Average " << average << std::endl;
	std::cout << "Variance " << variance << std::endl;
	std::cout << "StandardDeviation " << standardDeviation << std::endl;
	//std::cout << std::endl;
	//std::cout << "Colors average " << cAverage << std::endl;
	//std::cout << "Colors variance " << cVariance << std::endl;
	//std::cout << "Colors standardDeviation " << cStandardDeviation << std::endl;
	std::cout << std::endl;
#endif // PRINTS

#ifdef WRITE
	for (int i = 0; i < numberOfCol; i++)
		colorsFile << i << " " << coloring_h[i] << std::endl;


	for (int i = 0; i < numberOfCol; i++)
	{
		logFile << "Color " << i << " ";
		for (int j = 0; j < statsColors_h[i] / divider; j++)
		{
			logFile << "*";
		}
		logFile << std::endl;
	}
	logFile << "Every * is " << divider << " nodes" << std::endl;
	logFile << std::endl;

	logFile << "Number of used colors is " << counter << " on " << numberOfCol << " available" << std::endl;
	logFile << "Most used colors is " << max_i << " used " << max_c << " times" << std::endl;
	logFile << "Least used colors is " << min_i << " used " << min_c << " times" << std::endl;
	logFile << std::endl;
	logFile << "Average " << average << std::endl;
	logFile << "Variance " << variance << std::endl;
	logFile << "StandardDeviation " << standardDeviation << std::endl;
	//logFile << std::endl;
	//logFile << "Colors average " << cAverage << std::endl;
	//logFile << "Colors variance " << cVariance << std::endl;
	//logFile << "Colors standardDeviation " << cStandardDeviation << std::endl;
	logFile << std::endl;

	resultsFile << prefix << "used_colors " << counter << std::endl;
	resultsFile << prefix << "available_colors " << numberOfCol << std::endl;
	resultsFile << prefix << "most_used_colors " << max_i << std::endl;
	resultsFile << prefix << "most_used_colors_n_times " << max_c << std::endl;
	resultsFile << prefix << "least_used_colors " << min_i << std::endl;
	resultsFile << prefix << "least_used_colors_n_times " << min_c << std::endl;
	resultsFile << prefix << "average " << average << std::endl;
	resultsFile << prefix << "variance " << variance << std::endl;
	resultsFile << prefix << "standard_deviation " << standardDeviation << std::endl;
	//logFile << "Colors average " << cAverage << std::endl;
	//logFile << "Colors variance " << cVariance << std::endl;
	//logFile << "Colors standardDeviation " << cStandardDeviation << std::endl;
#endif // WRITE
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::calcProbs() {
	ColoringMCMC_k::logarithmer << < blocksPerGrid, threadsPerBlock >> > (nnodes, qStar_d);
	ColoringMCMC_k::logarithmer << < blocksPerGrid, threadsPerBlock >> > (nnodes, q_d);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(float) >> > (nedges, qStar_d);
	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(float) >> > (nedges, q_d);
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy(qStar_h, qStar_d, blocksPerGrid_half.x * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMemcpy(q_h, q_d, blocksPerGrid_half.x * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

	pStar = 0;
	p = 0;
	for (int i = 0; i < blocksPerGrid_half.x; i++)
	{
		pStar += qStar_h[i];
		p += q_h[i];
	}
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

// Original Counter
/*ColoringMCMC_k::conflictChecker << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, starCounter_d, starColoring_d, graphStruct_d->edges);
cudaDeviceSynchronize();

cuSts = cudaMemcpy(starCounter_h, starCounter_d, nedges * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

conflictCounterStar = 0;
for (int i = 0; i < nedges; i++)
	conflictCounterStar += starCounter_h[i];
*/
// End Original

// Original Prob Calc
/*
cuSts = cudaMemcpy(qStar_h, qStar_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
cuSts = cudaMemcpy(q_h, q_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

pStar = 0;
p = 0;
for (int i = 0; i < nnodes; i++)
{
	pStar += log(qStar_h[i]);
	p += log(q_h[i]);
}

std::cout << "q star: " << pStar << " old:" << p << std::endl;
*/

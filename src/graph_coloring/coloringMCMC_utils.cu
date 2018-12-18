// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

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

#ifdef COUNTER_WITH_NODES
__global__ void ColoringMCMC_k::conflictCounter(uint32_t nnodes, uint32_t * conflictCounter_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	uint32_t conflicts = 0;		//array used to set to 1 or 0 the colors occupied from the neighbors
	for (int i = 0; i < nneighs; i++)
		conflicts += (coloring_d[neighs[index + i]] == nodeCol) && (idx < neighs[index + i]);

	conflictCounter_d[idx] = conflicts;
}
#endif // COUNTER_WITH_NODES

/**
* For all the edges of the graph, set the value of conflictCounter_d to 0 or 1 if the nodes of the edge have the same color
*/
#ifdef COUNTER_WITH_EDGES
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
#endif // COUNTER_WITH_EDGES

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
__global__ void ColoringMCMC_k::sumReduction(uint32_t n, float * conflictCounter_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n)
		return;

	extern	__shared__ float sdata[];

	uint32_t tid = threadIdx.x;
	uint32_t blockSize = blockDim.x;
	uint32_t i = (blockSize * 2) * blockIdx.x + tid;

	sdata[tid] = conflictCounter_d[i] + conflictCounter_d[i + blockSize];

	/*uint32_t gridSize = (blockSize * 2) * gridDim.x;
	sdata[tid] = 0;
	while (i < n) {
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

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::calcConflicts(int &conflictCounter, uint32_t * coloring_d) {

#ifdef COUNTER_WITH_NODES
	ColoringMCMC_k::conflictCounter << < blocksPerGrid, threadsPerBlock >> > (nnodes, conflictCounter_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nnodes, (float*)conflictCounter_d);
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy(conflictCounter_h, conflictCounter_d, blocksPerGrid_half.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter = 0;

	for (int i = 0; i < blocksPerGrid_half.x; i++)
		conflictCounter += conflictCounter_h[i];
#endif // COUNTER_WITH_NODES
#ifdef COUNTER_WITH_EDGES
	ColoringMCMC_k::conflictChecker << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, conflictCounter_d, coloring_d, graphStruct_d->edges);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, (float*)conflictCounter_d);
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy(conflictCounter_h, conflictCounter_d, blocksPerGrid_half_edges.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter = 0;

	for (int i = 0; i < blocksPerGrid_half_edges.x; i++)
		conflictCounter += conflictCounter_h[i];
#endif // COUNTER_WITH_EDGES
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
#ifdef PRINTS
	std::cout << "Max Free Colors: " << statsFreeColors_max << " - Min Free Colors: " << statsFreeColors_min << " - AVG Free Colors: " << statsFreeColors_avg << std::endl;
#endif
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

	for (int i = 0; i < nnodes; i++)
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

	for (int i = 0; i < numberOfCol; i++)
	{
		resultsFile << prefix << "cluster_color_" << i << " " << statsColors_h[i] << std::endl;
	}
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

//****************************************************************************************
//									PRINTS & WRITES
//****************************************************************************************

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintConstructor0_start() {
	//https://stackoverflow.com/questions/34356768/managing-properly-an-array-of-results-that-is-larger-than-the-memory-available-a
	//colorsChecker_d e orderedColors_d

	//size_t total_mem, free_mem;
	//cudaMemGetInfo(&free_mem, &total_mem);
	//std::cout << "total mem: " << total_mem << " free mem:" << free_mem << std::endl;

	//int tot = nnodes * sizeof(uint32_t) * 3;
	//std::cout << "nnodes * sizeof(uint32_t): " << nnodes * sizeof(uint32_t) << " X 3" << std::endl;
	//tot += nnodes * sizeof(float) * 2;
	//std::cout << "nnodes * sizeof(float): " << nnodes * sizeof(float) << " X 2" << std::endl;
	//tot += nedges * sizeof(uint32_t);
	//std::cout << "nedges * sizeof(uint32_t): " << nedges * sizeof(uint32_t) << " X 1" << std::endl;
	//tot += nnodes * param.nCol * sizeof(bool);t
	//std::cout << "nnodes * param.nCol * sizeof(bool): " << nnodes * param.nCol * sizeof(bool) << " X 1" << std::endl;
	//tot += nnodes * param.nCol * sizeof(uint32_t);
	//std::cout << "nnodes * param.nCol * sizeof(uint32_t): " << nnodes * param.nCol * sizeof(uint32_t) << " X 1" << std::endl;
	//std::cout << "TOTALE: " << tot << " bytes" << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintConstructor1_end() {
	//size_t total_mem, free_mem;
	//cudaMemGetInfo(&free_mem, &total_mem);
	//std::cout << "total mem: " << total_mem << " free mem:" << free_mem << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun0_start(int iteration) {

#ifdef PRINTS
	std::cout << std::endl << "ColoringMCMC GPU" << std::endl;
	std::cout << "numCol: " << param.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	std::cout << "startingNCol: " << param.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	std::cout << "epsilon: " << param.epsilon << std::endl;
	std::cout << "lambda: " << param.lambda << std::endl;
	std::cout << "ratioFreezed: " << param.ratioFreezed << std::endl;
	std::cout << "maxRip: " << param.maxRip << std::endl << std::endl;
#endif // PRINTS

#ifdef WRITE

	std::string directory = std::to_string(nnodes) + "-" + std::to_string(prob) + "-results";

	logFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-logFile-" + std::to_string(iteration) + ".txt");
	resultsFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-resultsFile-" + std::to_string(iteration) + ".txt");
	colorsFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-colorsFile-" + std::to_string(iteration) + ".txt");

	size_t total_mem, free_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	logFile << "total memory: " << total_mem << " free memory:" << free_mem << std::endl;
	resultsFile << "total_memory " << total_mem << std::endl;
	resultsFile << "free_memory " << free_mem << std::endl;

	logFile << "numCol: " << param.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	logFile << "startingNCol: " << param.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	logFile << "epsilon: " << param.epsilon << std::endl;
	logFile << "lambda: " << param.lambda << std::endl;
	logFile << "ratioFreezed: " << param.ratioFreezed << std::endl;
	logFile << "maxRip: " << param.maxRip << std::endl << std::endl;

	resultsFile << "numCol " << param.nCol << std::endl;
#ifdef DYNAMIC_N_COLORS
	resultsFile << "startingNCol " << param.startingNCol << std::endl;
#endif // DYNAMIC_N_COLORS
	resultsFile << "epsilon " << param.epsilon << std::endl;
	resultsFile << "lambda " << param.lambda << std::endl;
	resultsFile << "ratioFreezed " << param.ratioFreezed << std::endl;
	resultsFile << "maxRip " << param.maxRip << std::endl;
#endif // WRITE
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun1_init() {
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
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun2_conflicts() {
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
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun3_newConflicts() {
#ifdef PRINTS
	std::cout << "nuovi conflitti rilevati: " << conflictCounterStar << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "nuovi conflitti rilevati: " << conflictCounterStar << std::endl;
#endif // WRITE

#if defined(STATS) && (defined(PRINTS) || defined(WRITE))
	getStatsFreeColors();
#endif // STATS && ( PRINTS || WRITE )
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun4() {

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
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun5() {
#ifdef PRINTS
	/*std::cout << "lambda: " << param.lambda << std::endl;
	std::cout << "probs star: " << pStar << " old:" << p << std::endl;
	std::cout << "left: " << param.lambda * (conflictCounter - conflictCounterStar) << " right:" << p - pStar << std::endl;
	std::cout << "result: " << result << std::endl;*/
	//std::cout << "random: " << random << std::endl;
#endif // PRINTS
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun6_change() {
#ifdef PRINTS
	std::cout << "CHANGE" << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "CHANGE" << std::endl;
#endif // WRITE
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun7_end() {
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

#ifdef WRITE
	logFile.close();
	resultsFile.close();
	colorsFile.close();
#endif // WRITE
}

//template<typename nodeW, typename edgeW>
//void ColoringMCMC<nodeW, edgeW>::__customPrintRun0() {
//}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

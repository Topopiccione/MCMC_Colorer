// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE)
__global__ void ColoringMCMC_k::initDistributionLine(float nCol, float denom, float lambda, float * probDistributionLine_d) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nCol)
		return;

	probDistributionLine_d[idx] = (float)(nCol - lambda * idx) / denom;
	//probDistributionLine_d[idx] = (float)(lambda * idx) / denom;
}
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE

#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP)
__global__ void ColoringMCMC_k::initDistributionExp(float nCol, float denom, float lambda, float * probDistributionExp_d) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nCol)
		return;

	probDistributionExp_d[idx] = exp(-lambda * idx) / denom;
}
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP

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
	ColoringMCMC_k::conflictCounter << < blocksPerGrid, threadsPerBlock >> > (nnodes, conflictCounter_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nnodes, (float*)conflictCounter_d);
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy(conflictCounter_h, conflictCounter_d, blocksPerGrid_half.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter = 0;

	for (int i = 0; i < blocksPerGrid_half.x; i++)
		conflictCounter += conflictCounter_h[i];
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::calcProbs() {

	ColoringMCMC_k::logarithmer << < blocksPerGrid, threadsPerBlock >> > (nnodes, qStar_d);
	ColoringMCMC_k::logarithmer << < blocksPerGrid, threadsPerBlock >> > (nnodes, q_d);
	cudaDeviceSynchronize();

	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(float) >> > (nnodes, qStar_d);
	ColoringMCMC_k::sumReduction << < blocksPerGrid_half, threadsPerBlock, threadsPerBlock.x * sizeof(float) >> > (nnodes, q_d);
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

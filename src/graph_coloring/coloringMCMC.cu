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
	blocksPerGrid_edges = dim3((nedges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half_edges = dim3(((nedges / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

	cuSts = cudaMalloc((void**)&coloring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&newColoring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	probColoring_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&probColoring_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
	probNewColoring_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&probNewColoring_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);

	counter_h = (uint32_t *)malloc(nedges * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&counter_d, nedges * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	newCounter_h = (uint32_t *)malloc(nedges * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&newCounter_d, nedges * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	cuSts = cudaMalloc((void**)&colorsChecker_d, nnodes * param.nCol * sizeof(bool));	cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&orderedColors_d, nnodes * param.nCol * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	divider = 1 / ((float)params.nCol);
}

// eliminare tutto quello creato nel construttore per evitare memory leak
template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::~ColoringMCMC() {
	cuSts = cudaFree(coloring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(newColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(probColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(probNewColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(counter_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(newCounter_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(colorsChecker_d); 		cudaCheck(cuSts, __FILE__, __LINE__);

	free(counter_h);
	free(newCounter_h);
	free(probColoring_h);
	free(probNewColoring_h);
}

__global__ void ColoringMCMC_k::initColoring(uint32_t nnodes, uint32_t * coloring_d, float divider, curandState * states) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	float randnum = curand_uniform(&states[idx]);

	int color = (int)(randnum / divider);

	coloring_d[idx] = color;
	//coloring_d[idx] = 0;

	//printf("nodo: %d, colore: %d\n", idx, coloring_d[idx]);
}

__global__ void ColoringMCMC_k::conflictCounter(uint32_t nedges, uint32_t * counter_d, uint32_t * coloring_d, node_sz * edges) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nedges)
		return;

	uint32_t idx0 = idx * 2;
	uint32_t idx1 = idx0 + 1;

	uint32_t node0 = edges[idx0];
	uint32_t node1 = edges[idx1];

	uint32_t col0 = coloring_d[node0];
	uint32_t col1 = coloring_d[node1];

	counter_d[idx] = col0 == col1;

	//printf("arco: %d, col0: %d, col1: %d, res: %d\n", idx, col0, col1, counter_d[idx]);
}

//template <uint32_t blockSize>
__device__ void ColoringMCMC_k::warpReduction(volatile int *sdata, uint32_t tid, uint32_t blockSize) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

//template <uint32_t blockSize>
__global__ void ColoringMCMC_k::sumReduction(uint32_t nedges, uint32_t * counter_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nedges)
		return;

	uint32_t tid = threadIdx.x;
	uint32_t blockSize = blockDim.x;
	uint32_t i = (blockSize * 2) * blockIdx.x + tid;
	uint32_t gridSize = (blockSize * 2) * gridDim.x;

	extern	__shared__ int sdata[];

	sdata[tid] = 0;
	while (i < nedges) {
		sdata[tid] += counter_d[i] + counter_d[i + blockSize];
		i += gridSize;
	}
	__syncthreads();

	//useless for blocks of dim <= 64
	// refs: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	if (blockSize >= 512) {
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}

	if (tid < 32)
		//ColoringMCMC_k::warpReduction<blockSize>(sdata, tid);
		ColoringMCMC_k::warpReduction(sdata, tid, blockSize);

	if (tid == 0) {
		counter_d[blockIdx.x] = sdata[0];
	}
}

__global__ void ColoringMCMC_k::selectNewColoring(uint32_t nnodes, uint32_t * newColoring_d, float * probNewColoring_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * orderedColors_d, curandState * states, float epsilon) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];					//indice in neighs del nodo
	uint32_t nneighs = cumulDegs[idx + 1] - index;		//numero di vicini

	uint32_t nodeCol = coloring_d[idx];					//il colore del nodo

	//bool * colorsChecker = new bool[nCol];				//indicatore di colore usato dai vicini del nodo
	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);
	//for (int i = 0; i < 80; i++)
		//colorsChecker[i] = 0;
	//memset(colorsChecker, 0, nCol * sizeof(bool));
	for (int i = 0; i < nneighs; i++)
	{
		colorsChecker[coloring_d[neighs[index + i]]] = 1;
	}

	//uint32_t * orderedColors = new uint32_t[nCol];				//contiene prima i colori usati e poi i liberi
	uint32_t * orderedColors = &(orderedColors_d[idx * nCol]);
	//for (int i = 0; i < 80; i++)
		//orderedColors[i] = 0;
	//memset(orderedColors, 0, nCol * sizeof(uint32_t));
	uint32_t Zp = nCol, Zn = 0;								//numero colori liberi (p) e occupati (n)
	for (int i = 0; i < nCol; i++)
	{
		orderedColors[Zn] += i * (1 - (1 - colorsChecker[i]));
		orderedColors[Zp - 1] += i * (1 - colorsChecker[i]);
		Zn += colorsChecker[i];
		Zp -= 1 - colorsChecker[i];
	}
	Zp = nCol - Zn;

	if (!Zp) {
		newColoring_d[idx] = nodeCol;
		probNewColoring_d[idx] = 1;				//??????????????
		//delete colorsChecker;
		//delete orderedColors;
		return;
	}

	float randnum = curand_uniform(&states[idx]);

	float threshold;
	uint32_t selectedIndex = 0;
	if (colorsChecker[nodeCol]) {						//se il colore del nodo � usato dai vicini
		threshold = 1 - epsilon * Zn;					//soglia per decidere se scegliere a caso tra un colore libero o uno occupato
		if (randnum < threshold)
		{
			selectedIndex = ((randnum * Zp) / threshold) + Zn;
			probNewColoring_d[idx] = (1 - epsilon * Zn) / Zp;
		}
		else
		{
			selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);
			probNewColoring_d[idx] = epsilon;
		}
		newColoring_d[idx] = orderedColors[selectedIndex];
	}
	else {
		threshold = 1 - epsilon * (nCol - 1);			//soglia per decidere se scegliere a caso se mantenere lo stesso colore
		if (randnum < threshold)
		{
			newColoring_d[idx] = nodeCol;
			probNewColoring_d[idx] = 1 - ((nCol - 1) * epsilon);
		}
		else
		{
			selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);
			newColoring_d[idx] = orderedColors[selectedIndex];
			probNewColoring_d[idx] = epsilon;
		}
	}

	//delete colorsChecker;
	//delete orderedColors;

	/*if (idx == 0) {
		printf("nodo: %d, numero vicini: %d\n", idx, nneighs);
		printf("colore nodo: %d\n", nodeCol);
		for (int i = 0; i < nneighs; i++)
		{
			printf("vicino %d color: %d\n", i, coloring_d[neighs[index + i]]);
		}
		printf("colorsChecker [ ");
		for (int i = 0; i < nCol; i++)
		{
			printf("%d ", colorsChecker[i]);
		}
		printf("]\n");
		printf("orderedColors [ ");
		for (int i = 0; i < nCol; i++)
		{
			printf("%d ", orderedColors[i]);
		}
		printf("]\n");
		printf("epsilon: %f\n", epsilon);
		printf("numero colori: %d, colori liberi: %d, colori occupati: %d\n", nCol, Zp, Zn);
		printf("numero random:%f, soglia:%f\n", randnum, threshold);
		printf("selectedIndex:%d, selectedColor:%d\n", selectedIndex, selectedColor);
	}*/
	//printf("nodo: %d, nuovo colore: %d, probabilit�: %f\n", idx, newColoring_d[idx], probNewColoring_d[idx]);
}

__global__ void ColoringMCMC_k::lookOldColoring(uint32_t nnodes, float * probColoring_d, col_sz nCol, uint32_t * newColoring_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];					//indice in neighs del nodo
	uint32_t nneighs = cumulDegs[idx + 1] - index;		//numero di vicini

	uint32_t nodeCol = coloring_d[idx];					//il colore del nodo
	uint32_t nodeNewCol = newColoring_d[idx];			//il colore del nuovo nodo

	//bool * colorsChecker = new bool[nCol];	//indicatore di colore usato dai vicini del nodo
	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);
	//for (int i = 0; i < 80; i++)
		//colorsChecker[i] = 0;
	//memset(colorsChecker, 0, nCol * sizeof(bool));

	for (int i = 0; i < nneighs; i++)
	{
		colorsChecker[newColoring_d[neighs[index + i]]] = 1;
	}

	uint32_t Zp = nCol, Zn = 0;							//numero colori liberi (p) e occupati (n)
	for (int i = 0; i < nCol; i++)
	{
		Zn += colorsChecker[i];
	}
	Zp = nCol - Zn;

	if (!Zp) {
		probColoring_d[idx] = 1;				//??????????????
		//delete colorsChecker;
		return;
	}

	if (colorsChecker[nodeNewCol]) {			//se il colore del nuovo nodo � usato dai nuovi vicini
		if (!colorsChecker[nodeCol]) {
			probColoring_d[idx] = (1 - epsilon * Zn) / Zp;
		}
		else {
			probColoring_d[idx] = epsilon;
		}
	}
	else {
		if (nodeNewCol == nodeCol) {
			probColoring_d[idx] = 1 - ((nCol - 1) * epsilon);
		}
		else {
			probColoring_d[idx] = epsilon;
		}
	}

	//delete colorsChecker;

	//printf("nodo: %d, probabilit�: %f\n", idx, probColoring_d[idx]);
}

/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::run() {

	cuSts = cudaMemset(coloring_d, 0, nnodes * sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);

	ColoringMCMC_k::initColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, divider, randStates);
	cudaDeviceSynchronize();


	do {

		rip++;

		/*ColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, counter_d, coloring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(counter_h, counter_d, nedges * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		counter = 0;
		for (int i = 0; i < nedges; i++)
			counter += counter_h[i];*/

		ColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, counter_d, coloring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		//ColoringMCMC_k::sumReduction <32><< < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, counter_d);
		ColoringMCMC_k::sumReduction << < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, counter_d);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(counter_h, counter_d, blocksPerGrid_half_edges.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		counter = 0;
		for (int i = 0; i < blocksPerGrid_half_edges.x; i++)
			counter += counter_h[i];

		if (counter == 0)
			break;

		std::cout << "***** Tentativo numero: " << rip << std::endl;
		std::cout << "conflitti rilevati: " << counter << std::endl;

		cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
		cudaMemset(orderedColors_d, 0, nnodes * param.nCol * sizeof(uint32_t));
		ColoringMCMC_k::selectNewColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, newColoring_d, probNewColoring_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, orderedColors_d, randStates, param.epsilon);
		cudaDeviceSynchronize();

		cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
		ColoringMCMC_k::lookOldColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, probColoring_d, param.nCol, newColoring_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, param.epsilon);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(probNewColoring_h, probNewColoring_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		cuSts = cudaMemcpy(probColoring_h, probColoring_d, nnodes * sizeof(float), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		probColoring = 1;
		probNewColoring = 1;
		for (int i = 0; i < nnodes; i++)
		{
			probColoring *= probColoring_h[i];
			probNewColoring *= probNewColoring_h[i];
		}

		//std::cout << "probColoring: " << probColoring << " probNewColoring: " << probNewColoring << std::endl;

		/*ColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, newCounter_d, newColoring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(newCounter_h, newCounter_d, nedges * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		newCounter = 0;
		for (int i = 0; i < nedges; i++)
			newCounter += newCounter_h[i];*/

		ColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, newCounter_d, newColoring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		//ColoringMCMC_k::sumReduction <32><< < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, newCounter_d);
		ColoringMCMC_k::sumReduction << < blocksPerGrid_half_edges, threadsPerBlock, threadsPerBlock.x * sizeof(uint32_t) >> > (nedges, newCounter_d);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(newCounter_h, newCounter_d, blocksPerGrid_half_edges.x * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		newCounter = 0;
		for (int i = 0; i < blocksPerGrid_half_edges.x; i++)
			newCounter += newCounter_h[i];

		std::cout << "nuovi conflitti rilevati: " << newCounter << std::endl;

		result = exp(-param.lambda * ((int64_t)newCounter - (int64_t)counter));		//exp(n) = e ^ (n)

		//std::cout << "result:" << result << " lambda:" << param.lambda << " newCounter:" << newCounter << " counter:" << counter << std::endl;

		result = (result * probNewColoring) / probColoring;

		//std::cout << "result: " << result << std::endl;

		result = result > 1 ? 1 : result;

		//std::cout << "result: " << result << std::endl;

		//********************************************************************** LOG
		//probColoring = 0;
		//probNewColoring = 0;
		//for (int i = 0; i < nnodes; i++)
		//{
			//probColoring += log(probColoring_h[i]);
			//probNewColoring += log(probNewColoring_h[i]);
		//}
		//result = -param.lambda * ((int64_t)newCounter - (int64_t)counter) + probColoring - probNewColoring;
		//result = exp(result);

		//std::cout << "result log: " << result << std::endl;

		//**********************************************************************

		random = ((float)rand() / (float)RAND_MAX);

		//std::cout << "random: " << random << std::endl;

		//if (random < result) {
		std::cout << "CHANGE" << std::endl;
		temp = coloring_d;
		coloring_d = newColoring_d;
		newColoring_d = temp;
		//}

	} while (rip < param.maxRip);

}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

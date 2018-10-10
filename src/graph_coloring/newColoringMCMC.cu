// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "newColoringMCMC.h"

__constant__ float NEW_LAMBDA;
__constant__ col_sz NEW_NCOLS;
__constant__ float NEW_EPSILON;
__constant__ float NEW_CUMULSUMDIST;
__constant__ float NEW_RATIOFREEZED;


template<typename nodeW, typename edgeW>
NewColoringMCMC<nodeW, edgeW>::NewColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params) :
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

	divider = 1 / ((float)params.nCol);
}

// eliminare tutto quello creato nel construttore per evitare memory leak
template<typename nodeW, typename edgeW>
NewColoringMCMC<nodeW, edgeW>::~NewColoringMCMC() {
	cuSts = cudaFree(coloring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(newColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(probColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(probNewColoring_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(counter_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(newCounter_d); 		cudaCheck(cuSts, __FILE__, __LINE__);

	free(counter_h);
	free(newCounter_h);
	free(probColoring_h);
	free(probNewColoring_h);
}

__global__ void NewColoringMCMC_k::initColoring(uint32_t nnodes, uint32_t * coloring_d, float divider, curandState * states) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	float randnum = curand_uniform(&states[idx]);

	int color = (int)(randnum / divider);

	coloring_d[idx] = color;

	//printf("nodo: %d, colore: %d\n", idx, coloring_d[idx]);
}

__global__ void NewColoringMCMC_k::conflictCounter(uint32_t nedges, uint32_t * counter_d, uint32_t * coloring_d, const node_sz * const edges) {

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

__global__ void NewColoringMCMC_k::selectNewColoring(uint32_t nnodes, uint32_t * newColoring_d, float * probNewColoring_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, curandState * states, float epsilon) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];					//indice in neighs del nodo
	uint32_t nneighs = cumulDegs[idx + 1] - index;		//numero di vicini

	uint32_t nodeCol = coloring_d[idx];					//il colore del nodo

	//uint32_t colors[nCol];
	uint32_t colorsChecker[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};					//indicatore di colore usato dai vicini del nodo
	for (int i = 0; i < nneighs; i++)
	{
		colorsChecker[coloring_d[neighs[index + i]]] = 1;
	}

	uint32_t orderedColors[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};		//contiene prima i colori usati e poi i liberi
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

__global__ void NewColoringMCMC_k::lookOldColoring(uint32_t nnodes, float * probColoring_d, col_sz nCol, uint32_t * newColoring_d, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, float epsilon) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];					//indice in neighs del nodo
	uint32_t nneighs = cumulDegs[idx + 1] - index;		//numero di vicini

	uint32_t nodeCol = coloring_d[idx];					//il colore del nodo
	uint32_t nodeNewCol = newColoring_d[idx];			//il colore del nuovo nodo

	//uint32_t colors[nCol];
	uint32_t colorsChecker[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};					//indicatore di colore usato dai vicini del nodo
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
	//printf("nodo: %d, probabilit�: %f\n", idx, probColoring_d[idx]);
}

__global__ void NewColoringMCMC_k::changeColoring(uint32_t nnodes, uint32_t * newColoring_d, uint32_t * coloring_d) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	coloring_d[idx] = newColoring_d[idx];

}

/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void NewColoringMCMC<nodeW, edgeW>::run() {

	cuSts = cudaMemset(coloring_d, 0, nnodes * sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);

	NewColoringMCMC_k::initColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, divider, randStates);
	cudaDeviceSynchronize();

	do {

		rip++;

		std::cout << "***** Tentativo numero: " << rip << std::endl;

		NewColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, counter_d, coloring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(counter_h, counter_d, nedges * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		counter = 0;
		for (int i = 0; i < nedges; i++)
		{
			counter += counter_h[i];
		}

		if (counter == 0)
			break;

		//if (counter > 0)
		//{
		std::cout << "conflitti rilevati: " << counter << std::endl;

		NewColoringMCMC_k::selectNewColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, newColoring_d, probNewColoring_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, randStates, param.epsilon);
		cudaDeviceSynchronize();

		NewColoringMCMC_k::lookOldColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, probColoring_d, param.nCol, newColoring_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, param.epsilon);
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

		std::cout << "probColoring: " << probColoring << " probNewColoring: " << probNewColoring << std::endl;

		NewColoringMCMC_k::conflictCounter << < blocksPerGrid_edges, threadsPerBlock >> > (nedges, newCounter_d, newColoring_d, graphStruct_d->edges);
		cudaDeviceSynchronize();

		cuSts = cudaMemcpy(newCounter_h, newCounter_d, nedges * sizeof(node_sz), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);

		newCounter = 0;
		for (int i = 0; i < nedges; i++)
		{
			newCounter += newCounter_h[i];
		}

		//std::cout << "nuovi conflitti rilevati: " << newCounter << std::endl;

		result = exp(-param.lambda * (newCounter - counter));		//exp(n) = e ^ (n)

		std::cout << "result:" << result << " lambda:" << param.lambda << " newCounter:" << newCounter << " counter:" << counter << std::endl;

		result = (result * probNewColoring) / probColoring;

		//std::cout << "result: " << result << std::endl;

		result = result > 1 ? 1 : result;

		//std::cout << "result: " << result << std::endl;

		random = ((float)rand() / (float)RAND_MAX);

		std::cout << "random: " << random << std::endl;

		//if (random < result) {
		std::cout << "CHANGE" << std::endl;
		NewColoringMCMC_k::changeColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, newColoring_d, coloring_d);
		cudaDeviceSynchronize();
		//}

		/*}
		else {
			rip = param.maxRip;
		}*/

	} while (rip < param.maxRip);

}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class NewColoringMCMC<col, col>;
template class NewColoringMCMC<float, float>;

// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams param) :
	Colorer<nodeW, edgeW>(inGraph_d),
	graphStruct_d(inGraph_d->getStruct()),
	nnodes(inGraph_d->getStruct()->nNodes),
	nedges(inGraph_d->getStruct()->nCleanEdges),
	randStates(randStates),
	numOfColors(0),
	threadId(0),
	param(param) {

	// configuro la griglia e i blocchi
	numThreads = 32;
	threadsPerBlock = dim3(numThreads, 1, 1);
	blocksPerGrid = dim3((nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_nCol = dim3((param.nCol + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half = dim3(((nnodes / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
#ifdef COUNTER_WITH_EDGES
	blocksPerGrid_edges = dim3((nedges + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half_edges = dim3(((nedges / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
#endif // COUNTER_WITH_EDGES

	__customPrintConstructor0_start();

	cuSts = cudaMalloc((void**)&coloring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&starColoring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

	q_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&q_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
	qStar_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&qStar_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);


#ifdef COUNTER_WITH_NODES
	conflictCounter_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&conflictCounter_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COUNTER_WITH_NODES
#ifdef COUNTER_WITH_EDGES
	conflictCounter_h = (uint32_t *)malloc(nedges * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&conflictCounter_d, nedges * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COUNTER_WITH_EDGES


	cuSts = cudaMalloc((void**)&colorsChecker_d, nnodes * param.nCol * sizeof(bool));	cudaCheck(cuSts, __FILE__, __LINE__);
#ifdef STANDARD
	cuSts = cudaMalloc((void**)&orderedColors_d, nnodes * param.nCol * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // STANDARD
#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE_CUMULATIVE)
	cuSts = cudaMalloc((void**)&probDistributionLine_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE_CUMULATIVE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
	cuSts = cudaMalloc((void**)&probDistributionExp_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP_CUMULATIVE


#ifdef STATS
	coloring_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
#if defined(COUNTER_WITH_NODES)
	statsColors_h = conflictCounter_h;
	statsFreeColors_d = conflictCounter_d;
#endif // COUNTER_WITH_NODES
#if defined(COUNTER_WITH_EDGES)
	statsColors_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&statsFreeColors_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COUNTER_WITH_EDGES
#endif

	__customPrintConstructor1_end();
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

#if defined(STATS) && defined(COUNTER_WITH_EDGES)
	free(statsColors_h);
	cuSts = cudaFree(statsFreeColors_d);			cudaCheck(cuSts, __FILE__, __LINE__);
#endif // STATS && COUNTER_WITH_EDGES
}

/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::run(int iteration) {

	rip = 0;

	__customPrintRun0_start(iteration);

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

	__customPrintRun1_init();

	do {

		rip++;

		calcConflicts(conflictCounter, coloring_d);

		if (conflictCounter == 0)
			break;

		__customPrintRun2_conflicts();

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

#ifdef STANDARD_CUMULATIVE
		ColoringMCMC_k::selectStarColoringCumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, randStates, param.epsilon, statsFreeColors_d);
		cudaDeviceSynchronize();
#endif // STANDARD_CUMULATIVE

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
#endif // COLOR_DECREASE_EXP_CUMULATIVE

#ifdef COLOR_BALANCE_ON_NODE_CUMULATIVE
		float partition = 15;
		ColoringMCMC_k::selectStarColoringBalanceOnNode_cumulative << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, randStates, partition, param.epsilon, statsFreeColors_d);
#endif // COLOR_BALANCE_ON_NODE_CUMULATIVE

		cudaDeviceSynchronize();

		/*cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
		ColoringMCMC_k::lookOldColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, q_d, param.nCol, starColoring_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, param.epsilon);
		cudaDeviceSynchronize();*/

		calcConflicts(conflictCounterStar, starColoring_d);

		__customPrintRun3_newConflicts();

		__customPrintRun4();

		//calcProbs();

		//param.lambda = -numberOfChangeColorStar * log(param.epsilon);

		//result = param.lambda * (conflictCounter - conflictCounterStar) + p - pStar;
		//result = exp(result);

		//random = ((float)rand() / (float)RAND_MAX);

		__customPrintRun5();

		//if (random < result) {
		__customPrintRun6_change();

		switchPointer = coloring_d;
		coloring_d = starColoring_d;
		starColoring_d = switchPointer;
		//}

	} while (rip < param.maxRip);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	__customPrintRun7_end();
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

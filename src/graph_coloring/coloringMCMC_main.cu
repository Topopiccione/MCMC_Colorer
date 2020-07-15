// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::ColoringMCMC(Graph<nodeW, edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams param) :
	Colorer<nodeW, edgeW>(inGraph_d),
	graphStruct_d(inGraph_d->getStruct()),
	nnodes(inGraph_d->getStruct()->nNodes),
	prob(inGraph_d->prob),
	randStates(randStates),
	numOfColors(0),
	threadId(0),
	param(param) {

	// configuro la griglia e i blocchi
	numThreads = 64;
	threadsPerBlock = dim3(numThreads, 1, 1);
	blocksPerGrid = dim3((nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_nCol = dim3((param.nCol + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
	blocksPerGrid_half = dim3(((nnodes / 2) + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

	// __printMemAlloc();

	cuSts = cudaMalloc((void**)&coloring_d, nnodes * sizeof(uint32_t));		cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&starColoring_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);

#ifdef TABOO
	cuSts = cudaMalloc((void**)&taboo_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // TABOO

	q_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&q_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
	qStar_h = (float *)malloc(nnodes * sizeof(float));
	cuSts = cudaMalloc((void**)&qStar_d, nnodes * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);

	conflictCounter_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&conflictCounter_d, nnodes * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc((void**)&colorsChecker_d, nnodes * param.nCol * sizeof(bool));	cudaCheck(cuSts, __FILE__, __LINE__);
#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE) || defined(COLOR_BALANCE_LINE)
	cuSts = cudaMalloc((void**)&probDistributionLine_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP) || defined(COLOR_BALANCE_EXP)
	cuSts = cudaMalloc((void**)&probDistributionExp_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP || COLOR_BALANCE_EXP
#ifdef COLOR_BALANCE_DYNAMIC_DISTR
	cuSts = cudaMalloc((void**)&probDistributionDynamic_d, param.nCol * sizeof(float));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COLOR_BALANCE_DYNAMIC_DISTR

#if defined(COLOR_BALANCE_LINE) || defined(COLOR_BALANCE_EXP) || defined(COLOR_BALANCE_DYNAMIC_DISTR)
	orderedIndex_h = (uint32_t *)malloc(param.nCol * sizeof(uint32_t));
	cuSts = cudaMalloc((void**)&orderedIndex_d, param.nCol * sizeof(uint32_t));	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COLOR_BALANCE_EXP

	coloring_h = (uint32_t *)malloc(nnodes * sizeof(uint32_t));
	statsColors_h = conflictCounter_h;
	statsColors_d = conflictCounter_d;
	statsFreeColors_d = conflictCounter_d;

}

template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::~ColoringMCMC() {
	cuSts = cudaFree(coloring_d); 					cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(starColoring_d); 				cudaCheck(cuSts, __FILE__, __LINE__);

#ifdef TABOO
	cuSts = cudaFree(taboo_d); 						cudaCheck(cuSts, __FILE__, __LINE__);
#endif // TABOO

	cuSts = cudaFree(colorsChecker_d); 				cudaCheck(cuSts, __FILE__, __LINE__);
#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE) || defined(COLOR_BALANCE_LINE)
	cuSts = cudaFree(probDistributionLine_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE
#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP) || defined(COLOR_BALANCE_EXP)
	cuSts = cudaFree(probDistributionExp_d); 		cudaCheck(cuSts, __FILE__, __LINE__);
#endif // DISTRIBUTION_EXP_INIT || COLOR_DECREASE_EXP || COLOR_BALANCE_EXP
#ifdef COLOR_BALANCE_DYNAMIC_DISTR
	cuSts = cudaFree(probDistributionDynamic_d); 	cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COLOR_BALANCE_DYNAMIC_DISTR

#if defined(COLOR_BALANCE_LINE) || defined(COLOR_BALANCE_EXP) || defined(COLOR_BALANCE_DYNAMIC_DISTR)
	free(orderedIndex_h);
	cuSts = cudaFree(orderedIndex_d); 				cudaCheck(cuSts, __FILE__, __LINE__);
#endif // COLOR_BALANCE_EXP

	cuSts = cudaFree(conflictCounter_d); 			cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(q_d); 							cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaFree(qStar_d);						cudaCheck(cuSts, __FILE__, __LINE__);

	free(coloring_h);
	free(conflictCounter_h);
	free(q_h);
	free(qStar_h);
}

/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::run(int iteration) {

	rip = 0;

	__customPrintRun0_start(iteration);

	cuSts = cudaMemset(coloring_d, 0, nnodes * sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);

#ifdef TABOO
	cuSts = cudaMemset(taboo_d, 0, nnodes * sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);
#endif // TABOO

#if defined(DISTRIBUTION_LINE_INIT) || defined(COLOR_DECREASE_LINE) || defined(COLOR_BALANCE_LINE)
	float denomL = 0;
	for (int i = 0; i < param.nCol; i++)
		denomL += (param.nCol - i * param.lambda);
	ColoringMCMC_k::initDistributionLine << < blocksPerGrid_nCol, threadsPerBlock >> > (param.nCol, denomL, param.lambda, probDistributionLine_d);
	cudaDeviceSynchronize();
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE || COLOR_BALANCE_LINE

#if defined(DISTRIBUTION_EXP_INIT) || defined(COLOR_DECREASE_EXP) || defined(COLOR_BALANCE_EXP)
	float denomE = 0;
	for (int i = 0; i < param.nCol; i++)
		denomE += exp(-param.lambda * i);
	ColoringMCMC_k::initDistributionExp << < blocksPerGrid_nCol, threadsPerBlock >> > (param.nCol, denomE, param.lambda, probDistributionExp_d);
	cudaDeviceSynchronize();
#endif // DISTRIBUTION_LINE_INIT || COLOR_DECREASE_LINE || COLOR_BALANCE_EXP

#if defined(COLOR_BALANCE_DYNAMIC_DISTR)
	for (uint32_t i = 0; i < param.nCol; i++) orderedIndex_h[i] = i;
	cuSts = cudaMemcpy(orderedIndex_d, orderedIndex_h, param.nCol * sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);
#endif

#ifdef STANDARD_INIT
	ColoringMCMC_k::initColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, randStates);
#endif // STANDARD_INIT

#ifdef DISTRIBUTION_LINE_INIT
	ColoringMCMC_k::initColoringWithDistribution << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, probDistributionLine_d, randStates);
#endif // DISTRIBUTION_LINE_INIT

#ifdef DISTRIBUTION_EXP_INIT
	ColoringMCMC_k::initColoringWithDistribution << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, param.nCol, probDistributionExp_d, randStates);
#endif // DISTRIBUTION_EXP_INIT
	cudaDeviceSynchronize();

	//__customPrintRun1_init(); 		// Prints initial coloring. WHO CARES.

	// Setting the tail cutting threshold.
	if (param.tailcut) {
		z = (50 > nnodes / 2000) ? 50 : (nnodes / 2000);
		LOG(TRACE) << TXT_BIYLW << "Tailcut set to " << z << " nodes" << TXT_NORML;
	}
	else {
		z = 0;
		LOG(TRACE) << TXT_BIYLW << "Tailcut disabled (--tailcut)" << TXT_NORML;
	}

	start = std::clock();
	do {
		rip++;

		calcConflicts(conflictCounter, coloring_d);

#ifdef HASTINGS
		calcStdDev(stdDev, coloring_d);
#endif // HASTINGS

		if (conflictCounter <= z)
			break;

		__customPrintRun2_conflicts(false);

		cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));

#ifdef STANDARD
		ColoringMCMC_k::selectStarColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, taboo_d, param.tabooIteration, randStates, param.epsilon, statsFreeColors_d);
		cudaDeviceSynchronize();
#endif // STANDARD

#if defined(COLOR_DECREASE_LINE) || defined(COLOR_DECREASE_EXP)
#ifdef COLOR_DECREASE_LINE
		float * probDistribution_d = probDistributionLine_d;
#endif // COLOR_DECREASE_LINE
#ifdef COLOR_DECREASE_EXP
		float * probDistribution_d = probDistributionExp_d;
#endif // COLOR_DECREASE_EXP

		ColoringMCMC_k::selectStarColoringDecrease << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, taboo_d, param.tabooIteration, probDistribution_d, randStates, param.lambda, param.epsilon, statsFreeColors_d);
#endif // COLOR_DECREASE_LINE || COLOR_DECREASE_EXP

#if defined(COLOR_BALANCE_LINE) || defined(COLOR_BALANCE_EXP)
		cuSts = cudaMemcpy(coloring_h, coloring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		memset(statsColors_h, 0, nnodes * sizeof(uint32_t));
		for (int i = 0; i < nnodes; i++) statsColors_h[coloring_h[i]]++;
		for (uint32_t i = 0; i < param.nCol; i++) orderedIndex_h[i] = i;
		std::sort(&orderedIndex_h[0], &orderedIndex_h[param.nCol], [&](int i, int j) {return statsColors_h[i] < statsColors_h[j]; });
		cuSts = cudaMemcpy(orderedIndex_d, orderedIndex_h, param.nCol * sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);

#ifdef COLOR_BALANCE_LINE
		float * probDistribution_d = probDistributionLine_d;
#endif // COLOR_DECREASE_LINE
#ifdef COLOR_BALANCE_EXP
		float * probDistribution_d = probDistributionExp_d;
#endif // COLOR_DECREASE_EXP

		ColoringMCMC_k::selectStarColoringBalance << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, taboo_d, param.tabooIteration, probDistribution_d, orderedIndex_d, randStates, param.lambda, param.epsilon, statsFreeColors_d);
#endif // COLOR_BALANCE_LINE || COLOR_BALANCE_EXP

#if defined(COLOR_BALANCE_DYNAMIC_DISTR)
		cuSts = cudaMemcpy(coloring_h, coloring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		memset(statsColors_h, 0, nnodes * sizeof(uint32_t));
		for (int i = 0; i < nnodes; i++) statsColors_h[coloring_h[i]]++;
		cuSts = cudaMemcpy(statsColors_d, statsColors_h, param.nCol * sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);

		ColoringMCMC_k::genDynamicDistribution << < blocksPerGrid_nCol, threadsPerBlock >> > (probDistributionDynamic_d, param.nCol, nnodes, statsColors_d);

		ColoringMCMC_k::selectStarColoringBalanceDynamic << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, qStar_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, taboo_d, param.tabooIteration, probDistributionDynamic_d, orderedIndex_d, randStates, param.lambda, param.epsilon, statsColors_d);
#endif

		cudaDeviceSynchronize();

#ifdef HASTINGS
		cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
		ColoringMCMC_k::lookOldColoring << < blocksPerGrid, threadsPerBlock >> > (nnodes, starColoring_d, q_d, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, param.epsilon);
		cudaDeviceSynchronize();
#endif //HASTINGS

		calcConflicts(conflictCounterStar, starColoring_d);

		__customPrintRun3_newConflicts();

#ifdef HASTINGS
		/*calcStdDev(stdDevStar, starColoring_d);

		std::cout << "stdDev " << stdDev << std::endl;
		std::cout << "stdDevStar " << stdDevStar << std::endl;

		float conflPart = (float)(conflictCounterStar - conflictCounter) / float(conflictCounter);
		std::cout << "conflPart " << -conflPart << std::endl;

		float balancePart = (stdDevStar - stdDev) / stdDev;
		std::cout << "balancePart " << -balancePart << std::endl;

		float l = 0.3, g = 1 - l;
		result = -l * conflPart
			- g * balancePart;
		std::cout << "result " << result << std::endl;*/

		//exec the sums of logs of q and qstar
		calcProbs();
		result = -param.lambda * (conflictCounterStar - conflictCounter) + (p - pStar);
		result = exp(result);

		random = ((float)rand() / (float)RAND_MAX);
		std::cout << "random " << random << std::endl;

		__customPrintRun5();

		__customPrintRun6_change();
#endif //HASTINGS

		switchPointer = coloring_d;
		coloring_d = starColoring_d;
		starColoring_d = switchPointer;

		//getStatsNumColors("running_");

	} while (rip < param.maxRip);

	if(param.tailcut) {
		cuSts = cudaMemcpy(coloring_h, coloring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		memset(statsColors_h, 0, nnodes * sizeof(uint32_t));
		for (int i = 0; i < nnodes; i++) statsColors_h[coloring_h[i]]++;
		for (uint32_t i = 0; i < param.nCol; i++) orderedIndex_h[i] = i;
		std::sort(&orderedIndex_h[0], &orderedIndex_h[param.nCol], [&](int i, int j) {return statsColors_h[i] < statsColors_h[j]; });
		cuSts = cudaMemcpy(orderedIndex_d, orderedIndex_h, param.nCol * sizeof(uint32_t), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);

		while (conflictCounter > 0) {
			__customPrintRun2_conflicts(true);
			// set conflictCounter_d vector with 1 or 0 to indicate conflicts
			ColoringMCMC_k::conflictCounter << < blocksPerGrid, threadsPerBlock >> > (nnodes, conflictCounter_d, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs);
			// set colorsChecker_d vector values to 0
			cudaMemset(colorsChecker_d, 0, nnodes * param.nCol * sizeof(bool));
			// resolve conflicts
			ColoringMCMC_k::tailCutting << < 1, 1 >> > (nnodes, param.nCol, coloring_d, graphStruct_d->cumulDegs, graphStruct_d->neighs, colorsChecker_d, conflictCounter, conflictCounter_d, orderedIndex_d);
			calcConflicts(conflictCounter, coloring_d);
			__customPrintRun3_newConflicts();
		}
	}

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	if (rip == param.maxRip)
		maxIterReached = true;

	__customPrintRun7_end();
}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

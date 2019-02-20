// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"


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
	std::cout << "epsilon: " << param.epsilon << std::endl;
	std::cout << "lambda: " << param.lambda << std::endl;
	std::cout << "ratioFreezed: " << param.ratioFreezed << std::endl;
	std::cout << "maxRip: " << param.maxRip << std::endl << std::endl;
	std::cout << "numColorRatio: " << param.numColorRatio << std::endl;
#endif // PRINTS

#ifdef WRITE

	logFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-logFile-" + std::to_string(iteration) + ".txt");
	resultsFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-resultsFile-" + std::to_string(iteration) + ".txt");
	colorsFile.open(directory + "/" + std::to_string(nnodes) + "-" + std::to_string(prob) + "-colorsFile-" + std::to_string(iteration) + ".txt");

	size_t total_mem, free_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	logFile << "total memory: " << total_mem << " free memory:" << free_mem << std::endl;
	resultsFile << "total_memory " << total_mem << std::endl;
	resultsFile << "free_memory " << free_mem << std::endl;

	logFile << "numCol: " << param.nCol << std::endl;
	logFile << "epsilon: " << param.epsilon << std::endl;
	logFile << "lambda: " << param.lambda << std::endl;
	logFile << "ratioFreezed: " << param.ratioFreezed << std::endl;
	logFile << "maxRip: " << param.maxRip << std::endl << std::endl;
	logFile << "numColorRatio: " << param.numColorRatio << std::endl;

	resultsFile << "numCol " << param.nCol << std::endl;
	resultsFile << "epsilon " << param.epsilon << std::endl;
	resultsFile << "lambda " << param.lambda << std::endl;
	resultsFile << "ratioFreezed " << param.ratioFreezed << std::endl;
	resultsFile << "maxRip " << param.maxRip << std::endl;
	resultsFile << "numColorRatio " << param.numColorRatio << std::endl;
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
void ColoringMCMC<nodeW, edgeW>::__customPrintRun2_conflicts(bool isTailCutting) {
#ifdef PRINTS
	std::cout << "***** Tentativo numero: " << rip << std::endl;
	if (isTailCutting)
		std::cout << "---> TailCutting" << std::endl;
	std::cout << "conflitti rilevati: " << conflictCounter << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "***** Tentativo numero: " << rip << std::endl;
	if (isTailCutting)
		logFile << "---> TailCutting" << std::endl;
	logFile << "conflitti rilevati: " << conflictCounter << std::endl;

	resultsFile << "iteration " << rip << std::endl; if (isTailCutting)
		resultsFile << "---> TailCutting" << std::endl;
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
	std::cout << "lambda: " << param.lambda << std::endl;
	std::cout << "probs star: " << pStar << " old:" << p << std::endl;
	std::cout << "left: " << param.lambda * (conflictCounter - conflictCounterStar) << " right:" << p - pStar << std::endl;
	std::cout << "result: " << result << std::endl;
	std::cout << "random: " << random << std::endl;
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
	std::string maxIteration = rip < param.maxRip ? "no" : "yes";
#ifdef PRINTS
	std::cout << "COLORAZIONE FINALE" << std::endl;
	std::cout << "Time " << duration << std::endl;
	std::cout << "Max iteration reached " << maxIteration << std::endl;
#endif // PRINTS
#ifdef WRITE
	logFile << "COLORAZIONE FINALE" << std::endl;
	logFile << "Time " << duration << std::endl;
	logFile << "Max iteration reached " << maxIteration << std::endl;

	resultsFile << "time " << duration << std::endl;
	resultsFile << "max_iteration_reached " << maxIteration << std::endl;
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

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::getStatsFreeColors() {

	uint32_t		statsFreeColors_max, statsFreeColors_min, statsFreeColors_avg;

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

	int numberOfCol = param.nCol;

	float average = 0, variance = 0, standardDeviation, balancingIndex;

	average = (float)nnodes / numberOfCol;

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
			balancingIndex += pow(statsColors_h[i] - average, 2.f);
		}
	}
	balancingIndex /= nnodes * prob;
	balancingIndex = sqrtf(balancingIndex);

	for (int i = 0; i < numberOfCol; i++) {
		variance += pow((statsColors_h[i] - average), 2.f);
	}
	variance /= numberOfCol;

	standardDeviation = sqrt(variance);

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
	std::cout << "BalancingIndex " << balancingIndex << std::endl;
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
	logFile << "BalancingIndex " << balancingIndex << std::endl;
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
	resultsFile << prefix << "balancing_index " << balancingIndex << std::endl;
	//logFile << "Colors average " << cAverage << std::endl;
	//logFile << "Colors variance " << cVariance << std::endl;
	//logFile << "Colors standardDeviation " << cStandardDeviation << std::endl;
#endif // WRITE
}

//template<typename nodeW, typename edgeW>
//void ColoringMCMC<nodeW, edgeW>::__customPrintRun0() {
//}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;
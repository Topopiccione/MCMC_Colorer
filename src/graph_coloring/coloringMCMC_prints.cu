// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

/*
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__printMemAlloc() {
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
*/

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun0_start(int iteration) {

	LOG(TRACE) << std::endl << "ColoringMCMC GPU";
	LOG(TRACE) << "numCol: " << param.nCol;
	LOG(TRACE) << "epsilon: " << param.epsilon;
	LOG(TRACE) << "lambda: " << param.lambda;
	LOG(TRACE) << "ratioFreezed: " << param.ratioFreezed;
	LOG(TRACE) << "maxRip: " << param.maxRip << std::endl;
	LOG(TRACE) << "numColorRatio: " << param.numColorRatio;

	logFile.open(directory + ".log");
	colorsFile.open(directory + "-colors.txt");

	size_t total_mem, free_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	logFile << "total memory: " << total_mem << " free memory:" << free_mem << std::endl;


	logFile << "numCol: " << param.nCol << std::endl;
	logFile << "epsilon: " << param.epsilon << std::endl;
	logFile << "lambda: " << param.lambda << std::endl;
	logFile << "ratioFreezed: " << param.ratioFreezed << std::endl;
	logFile << "maxRip: " << param.maxRip << std::endl << std::endl;
	logFile << "numColorRatio: " << param.numColorRatio << std::endl;

}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun1_init() {
	LOG(TRACE) << "COLORAZIONE INIZIALE";
	logFile << "COLORAZIONE INIZIALE" << std::endl;
	getStatsNumColors("start_");
	LOG(TRACE) << std::endl << "end colorazione iniziale -------------------------------------------------------------------" << std::endl << std::endl;
	logFile << std::endl << "end colorazione iniziale -------------------------------------------------------------------" << std::endl << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun2_conflicts(bool isTailCutting) {
	LOG(TRACE) << "***** Tentativo numero: " << rip;
	if (isTailCutting)
		LOG(TRACE) << "---> TailCutting";
	LOG(TRACE) << "conflitti rilevati: " << conflictCounter;

	logFile << "***** Tentativo numero: " << rip << std::endl;
	if (isTailCutting)
		logFile << "---> TailCutting" << std::endl;
	logFile << "conflitti rilevati: " << conflictCounter << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun3_newConflicts() {
	LOG(TRACE) << "nuovi conflitti rilevati: " << conflictCounterStar;
	logFile << "nuovi conflitti rilevati: " << conflictCounterStar << std::endl;

	getStatsFreeColors();
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun5() {
	LOG(TRACE) << "lambda: " << -param.lambda;
	LOG(TRACE) << "probs p: " << p << " pStar:" << pStar;
	LOG(TRACE) << "left(no lambda): " << conflictCounterStar - conflictCounter << " right:" << p - pStar;
	LOG(TRACE) << "result: " << result;
	LOG(TRACE) << "random: " << random;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun6_change() {
	LOG(TRACE) <<"CHANGE";
	logFile << "CHANGE" << std::endl;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::__customPrintRun7_end() {
	std::string maxIteration = rip < param.maxRip ? "no" : "yes";
	LOG(TRACE) << "COLORAZIONE FINALE";
	LOG(TRACE) << "Time " << duration;
	LOG(TRACE) << "Max iteration reached " << maxIteration;

	logFile << "COLORAZIONE FINALE" << std::endl;
	logFile << "Time " << duration << std::endl;
	logFile << "Max iteration reached " << maxIteration << std::endl;

	getStatsNumColors("end_");

	LOG(TRACE) << std::endl << "end colorazione finale -------------------------------------------------------------------";
	logFile << std::endl << "end colorazione finale -------------------------------------------------------------------" << std::endl << std::endl;

	logFile.close();
	colorsFile.close();
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

	LOG(TRACE) << "Max Free Colors: " << statsFreeColors_max << " - Min Free Colors: " << statsFreeColors_min << " - AVG Free Colors: " << statsFreeColors_avg;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::getStatsNumColors(std::string prefix) {

	cuSts = cudaMemcpy(coloring_h, coloring_d, nnodes * sizeof(uint32_t), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
	memset(statsColors_h, 0, nnodes * sizeof(uint32_t));
	for (int i = 0; i < nnodes; i++)
	{
		statsColors_h[coloring_h[i]]++;
	}
	int counter = 0;
	int max_i = 0, min_i = nnodes;
	int max_c = 0, min_c = nnodes;

	int numberOfCol = param.nCol;

	float average = 0, variance = 0, standardDeviation, balancingIndex = 0;

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

	balancingIndex /= (nnodes * prob);
	balancingIndex = sqrtf(balancingIndex);

	for (int i = 0; i < numberOfCol; i++) {
		variance += pow((statsColors_h[i] - average), 2.f);
	}
	variance /= numberOfCol;

	standardDeviation = sqrt(variance);

	int divider = (max_c / (param.nCol / 3) > 0) ? max_c / (param.nCol / 3) : 1;


#ifdef PRINTHISTOGRAM
	for (int i = 0; i < numberOfCol; i++)
	{
		std::cout << "Color " << i << " ";
		std::string linea;
		for (int j = 0; j < statsColors_h[i] / divider; j++)
		{
			linea += "*";
		}
		LOG(TRACE) << linea;
	}
	LOG(TRACE) <<"Every * is " << divider << " nodes";
#endif // PRINTHISTOGRAM

	LOG(TRACE) << "Number of used colors is " << counter << " on " << numberOfCol << " available";
	LOG(TRACE) << "Most used colors is " << max_i << " used " << max_c << " times";
	LOG(TRACE) << "Least used colors is " << min_i << " used " << min_c << " times";
	LOG(TRACE) << "Average " << average;
	LOG(TRACE) << "Variance " << variance;
	LOG(TRACE) << "StandardDeviation " << standardDeviation;
	LOG(TRACE) << "BalancingIndex " << balancingIndex;
	//LOG(TRACE) << "Colors average " << cAverage << std::endl;
	//LOG(TRACE) << "Colors variance " << cVariance << std::endl;
	//LOG(TRACE) << "Colors standardDeviation " << cStandardDeviation << std::endl;

	if (prefix == "end_") {
		for (int i = 0; i < nnodes; i++)
			colorsFile << i << " " << coloring_h[i] << std::endl;
	}

#ifdef PRINTHISTOGRAM
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
#endif //PRINTHISTOGRAM

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
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

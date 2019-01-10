#include "coloringMCMC_CPU.h"
#include "utils/dbg.h"
#include "utils/miscUtils.h"

template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::show_histogram() {
	std::vector<size_t> histBins(nCol, 0);
	std::vector<size_t> histBinsOcc(nCol, 0);
	// Initial check: each value should be 0 <= val < nCol
	size_t idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { if ((val < 0) | (val >= nCol)) std::cout << "Error in color " << idx << ". Exiting..." << std::endl; idx++;} );

	// Filling histogram bins
	idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBinsOcc[val] += (Cviols[idx++] == 1);} );
	idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBins[val] += (Cviols[idx++] == 0);} );

	// finding the correct scaler, so that the maximum bin get printed 80 times (ignoring rounding errors...)
	std::vector<size_t> tempVec(nCol);
	idx = 0;
	std::for_each( std::begin(tempVec), std::end(tempVec), [&](size_t &val) { val = histBinsOcc[idx] + histBins[idx]; idx++;} );
	auto max_val = std::max_element( std::begin(tempVec), std::end(tempVec) );
	size_t scaler = *max_val / 80;

	// Printing the histogram.
	for (size_t i = 0; i < histBins.size(); i++) {
		std::cout << i << ": ";

		size_t howMany = histBins[i] / scaler;
		std::cout << TXT_BIGRN;
		for (size_t j = 0; j < howMany; j++)
			std::cout << "*";

		howMany = histBinsOcc[i] / scaler;
		std::cout << TXT_BIRED;
		for (size_t j = 0; j < howMany; j++)
			std::cout << "*";

		std::cout << TXT_NORML << std::endl;
	}
	std::cout << "Each '*' = " << scaler << " nodes" << std::endl;
}


template<typename nodeW, typename edgeW>
bool ColoringMCMC_CPU<nodeW, edgeW>::unlock_stall() {
	if (Cviol == Cstarviol) {
		for (size_t i = 0; i < C.size(); i++) {
			if (Cviols[i]) {
				//std::cout << "Node " << i << " from " << C[i];
				C[i] = rand() % (nCol - 1);
				//std::cout << " to " << C[i] << std::endl;
				// const size_t nodeDeg = str->cumulDegs[i + 1] - str->cumulDegs[i];
				// const size_t neighIdx = str->cumulDegs[i];
				// const node * neighPtr = str->neighs + neighIdx;
				// for (size_t j = 0; j < nodeDeg; j++)
				// 	C[neighPtr[j]] = rand() % (nCol - 1);

			}
		}
		return true;
	}
	return false;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::saveStats( size_t it, float duration, std::ofstream & outFile ) {
	outFile << "MCMC Colorer - CPU version - Report" << std::endl;
	outFile << "-------------------------------------------" << std::endl;
	outFile << "GRAPH INFO" << std::endl;
	outFile << "Nodes: " << nNodes << " - Edges: " << str->nEdges << std::endl;
	outFile << "Max deg: " << this->graph->getMaxNodeDeg() << " - Min deg: " << this->graph->getMinNodeDeg() <<
			" - Avg deg: " << this->graph->getMeanNodeDeg() << std::endl;
	outFile << "Edge probability (for randomly generated graphs): " << this->graph->prob << std::endl;
	outFile << "Seed: " << seed << std::endl;
	outFile << "-------------------------------------------" << std::endl;
	outFile << "EXECUTION INFO" << std::endl;
	outFile << "Iteration: " << it << std::endl;
	outFile << "Execution time: " << duration << std::endl;
	outFile << "Iteration performed: " << iter << std::endl;
	outFile << "Max iteration reached: ";
	if (maxIterReached) std::cout << "1"; else std::cout << "0"; std::cout << std::endl;
	outFile << "-------------------------------------------" << std::endl;
	outFile << "Number of colors: " << nCol << std::endl;
	outFile << "Color histogram:" << std::endl;
	std::vector<size_t> histBins(nCol, 0);
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBins[val]++;} );
	size_t idx = 0;
	size_t usedCols = 0;
	std::for_each( std::begin(histBins), std::end(histBins), [&](size_t val) {outFile << idx << ": " << histBins[idx] << std::endl; idx++; if (val) usedCols++;} );
	outFile << "Color histogram end" << std::endl;
	outFile << "-------------------------------------------" << std::endl;
	outFile << "Used colors: " << usedCols << std::endl;
	float mean = std::accumulate( std::begin(histBins), std::end(histBins), 0) / (float) nCol;
	float variance = 0;
	std::for_each( std::begin(histBins), std::end(histBins), [&](size_t val) {variance += ((val - mean) * (val - mean));} );
	variance /= (float) nCol;
	float std = sqrtf( variance );
	outFile << "Average number of nodes for each color: " << mean << std::endl;
	outFile << "Variance: " << variance << std::endl;
	outFile << "StD: " << std << std::endl;
}

/////////////////////
template class Graph<float, float>;
template class ColoringMCMC_CPU<float, float>;

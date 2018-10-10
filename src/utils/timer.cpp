// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "timer.h"

Timer::Timer() {}

double Timer::duration() {
	std::chrono::duration<double, std::milli> fp_ms = end - start;
	return fp_ms.count();
}

void Timer::startTime() {
	start = std::chrono::high_resolution_clock::now();
}

void Timer::endTime() {
	end = std::chrono::high_resolution_clock::now();
}

void Timer::saveTimeToFile( std::string timeFilename, bool append, uint32_t nNodes, uint32_t nEdges, uint32_t nClasses,
			std::string dataFilename, std::string labelFilename, uint32_t nThreads ) {
	if (timeFilename != "") {
		// Append computation time to log file, if specified by option --tttt
		if (append) {
			std::ofstream timeFile( timeFilename.c_str(), std::ios::out | std::ios::app );
			timeFile << "Datafile:" << dataFilename << " - Labelfile:" << labelFilename <<
					" - nNodes: " << nNodes << " - nEdges: " << nEdges << " - nClasses: " << nClasses <<
					" - #threads: " << nThreads << " - Computation time (s): " << duration() / 1000 << std::endl;
			timeFile.close();
		} else {

		}
	}
}

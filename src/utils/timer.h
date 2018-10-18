#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

class Timer {
public:
	Timer();

	void startTime();
	void endTime();
	double duration();
	void saveTimeToFile( std::string timeFilename, bool append, uint32_t nNodes, uint32_t nEdges, uint32_t nClasses,
				std::string dataFilename, std::string labelFilename, uint32_t nThreads );

protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
};

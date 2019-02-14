// Commandline Argument Handler
// Alessandro Petrini, 2017
#pragma once
#include <string>
#include <cinttypes>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <time.h>

// Class for processing command line arguments

class ArgHandle {
public:
	ArgHandle( int argc, char **argv );
	virtual ~ArgHandle();

	void processCommandLine();

	std::string		dataFilename;
	std::string		foldFilename;
	std::string		labelFilename;
	std::string		outFilename;
	std::string		geneOutFilename;
	std::string		statesOutFilename;
	std::string		foldsOutFilename;
	std::string		timeOutFilename;

	uint32_t		m;
	uint32_t		n;
	double			prob;
	double			numColRatio;
	uint32_t		nFolds;
	uint32_t		seed;
	uint32_t		verboseLevel;
	uint32_t		nThreads;
	uint32_t		repetitions;
	uint32_t		tabooIteration;

	bool			generateRandomFold;
	bool			simulate;

private:
	void displayHelp();
	int				argc;
	char		**	argv;

};

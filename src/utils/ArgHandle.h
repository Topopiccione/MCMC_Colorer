// MCMC Colorer - Command line argument manager
// Alessandro Petrini, 2019-20
#pragma once
#include <string>
#include <vector>
#include <cinttypes>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "utils/miscUtils.h"

class ArgHandle {
public:
	ArgHandle(int argc, char ** argv);
	virtual ~ArgHandle();

	void processCommandLine();

	std::string		graphFilename;
	std::string		outDir;

	double			prob;
	double			numColRatio;
	uint32_t		n;
	uint32_t		nCol;
	uint32_t		seed;
	uint32_t		verboseLevel;
	uint32_t		repetitions;
	uint32_t		tabooIteration;

	bool			simulate;
	bool			mcmccpu;
	bool			mcmcgpu;
	bool			lubygpu;
	bool			tailcut;
	
	bool			greedyff;		//use this to enable GreedyFF coloring

	std::string		graphName;

private:
	void displayHelp();
	void printLogo();
	void citeMe();
	int				argc;
	char		**	argv;

};

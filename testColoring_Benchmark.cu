// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include <ctime>
#include "utils/ArgHandle.h"
#include "utils/fileImporter.h"
#include "utils/dbg.h"
#include "graph/graph.h"
#include "graph/graphCPU.cpp"
#include "graph/graphGPU.cu"
#include "graph_coloring/coloring.h"
#include "graph_coloring/coloringMCMC_CPU.h"
#include "graph_coloring/coloringLuby.h"
#include "graph_coloring/coloringMCMC.h"
#include "GPUutils/GPURandomizer.h"
#include "easyloggingpp/easylogging++.h"

bool		g_traceLogEn;	// Declared in utils/miscUtils.h
dbg		*	g_debugger;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {

	////EasyLogging++
	START_EASYLOGGINGPP(argc, argv);
    el::Configurations conf("../logger.conf");
    el::Loggers::reconfigureLogger("default", conf);
    el::Loggers::reconfigureAllLoggers(conf);

	el::Configuration * loggerConf = conf.get( el::Level::Trace, el::ConfigurationType::Enabled );
	g_traceLogEn = (loggerConf->value() == "true");
	// Debugger pre-init
	g_debugger = nullptr;

	// Commandline arguments
	ArgHandle commandLine( argc, argv );
	commandLine.processCommandLine();

	//uint32_t			N				= commandLine.n;
	//uint32_t			M				= commandLine.m;
	//float				prob			= (float) commandLine.prob;
	uint32_t			seed			= commandLine.seed;
	std::string			graphFileName	= commandLine.dataFilename;
	std::string			labelsFileName	= commandLine.labelFilename;

	//seed = 10000;

	std::clock_t start;
	double duration;

	bool GPUEnabled = 1;

	fileImporter fImport( graphFileName, labelsFileName );
	Graph<float, float> test( &fImport, !GPUEnabled );
	LOG(TRACE) << "Nodi: " << test.getStruct()->nNodes << " - Archi: " << test.getStruct()->nEdges;
	LOG(TRACE) << "minDeg: " << test.getMinNodeDeg() << " - maxDeg: " << test.getMaxNodeDeg() << " - meanDeg: "
		<< test.getMeanNodeDeg();

	//// CPU greedy coloring
	// Graph<col, col> graph( N, GPUEnabled );  	// random graph
	// ColoringGeedyCPU<col,col> colGreedyCPU(&graph);
	// colGreedyCPU.run();
	// cout << "Greedy-CPU coloring elapsed time: " << colGreedyCPU.getElapsedTime() << "(sec)" << endl;
	//colGreedyCPU.print(0);

	Graph<float, float> graph_d( &test );
	//// GPU Luby coloring
	GPURand GPURandGen( test.getStruct()->nNodes, (long) commandLine.seed );
	ColoringLuby<float, float> colLuby(&graph_d, GPURandGen.randStates);
	start = std::clock();
	colLuby.run_fast();
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << TXT_NORML;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU elapsed time: " << duration << TXT_NORML;

	ColoringMCMCParams aa;
	aa.nCol = 70;//test.getMaxNodeDeg() / 2.0f;
	aa.epsilon = 1e-12;
	aa.lambda = 2.0f;
	aa.ratioFreezed = 0.1f;

	//// GPU MCMC coloring

	// ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, aa);
	// start = std::clock();
	// colMCMC.run();
	// duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	ColoringMCMC_CPU<float, float> mcmc_cpu( &test, aa, seed );
	g_debugger = new dbg( &test, &mcmc_cpu );
	start = std::clock();
	mcmc_cpu.run();
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	LOG(TRACE) << TXT_BIYLW  << "MCMC_CPU elapsed time: " << duration << TXT_NORML;

	if (g_debugger != nullptr)
		delete g_debugger;

	return EXIT_SUCCESS;
}

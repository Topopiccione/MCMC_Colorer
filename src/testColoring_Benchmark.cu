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
#include "easyloggingpp/easyloggingpp/easylogging++.h"

#define INLINE_ARGS
//argomenti tipo : --data net.txt --label lab.txt --gene gene.txt

bool		g_traceLogEn;	// Declared in utils/miscUtils.h
dbg		*	g_debugger;

INITIALIZE_EASYLOGGINGPP

void newComponents(Graph<float, float> * test) {

	// arches rappresenta una lista di coppie di nodi che consistono nei singoli archi del grafo (senza ripetizioni)
	std::vector<node_sz> tempEdges;

	for (int i = 0; i < test->getStruct()->nNodes; i++)
	{
		int index = test->getStruct()->cumulDegs[i];
		int numOfNeighbors = test->getStruct()->cumulDegs[i + 1] - index;

		for (int j = 0; j < numOfNeighbors; j++)
		{
			if (i < test->getStruct()->neighs[index + j]) {
				tempEdges.push_back(i);
				tempEdges.push_back(test->getStruct()->neighs[index + j]);
			}
		}
	}

	test->getStruct()->nCleanEdges = tempEdges.size() / 2;
	test->getStruct()->edges = new node_sz[tempEdges.size()];
	for (int i = 0; i < tempEdges.size(); i++)
	{
		test->getStruct()->edges[i] = tempEdges[i];
	}

	std::cout << "Archi singoli: " << (tempEdges.size() / 2) << std::endl;

	//node_sz	* edges = &tempEdges[0];
	//test->getStruct()->edges = edges;
	//test->getStruct()->edges = tempEdges.data();
}

int main(int argc, char *argv[]) {

	////EasyLogging++
	START_EASYLOGGINGPP(argc, argv);
	el::Configurations conf("../src/logger.conf");
	el::Loggers::reconfigureLogger("default", conf);
	el::Loggers::reconfigureAllLoggers(conf);

	el::Configuration * loggerConf = conf.get(el::Level::Trace, el::ConfigurationType::Enabled);
	g_traceLogEn = (loggerConf->value() == "true");
	// Debugger pre-init
	g_debugger = nullptr;

#ifdef INLINE_ARGS
	argc = 7;
	argv[1] = "--data";
	argv[2] = "net.txt";
	argv[3] = "--label";
	argv[4] = "lab.txt";
	argv[5] = "--gene";
	argv[6] = "gene.txt";
#endif // INLINE_ARGS

	// Commandline arguments
	ArgHandle commandLine(argc, argv);
	commandLine.processCommandLine();

	//uint32_t			N				= commandLine.n;
	//uint32_t			M				= commandLine.m;
	//float				prob			= (float) commandLine.prob;
	uint32_t			seed = commandLine.seed;
	std::string			graphFileName = commandLine.dataFilename;
	std::string			labelsFileName = commandLine.labelFilename;

	//seed = 10000;

	std::clock_t start;
	double duration;

	bool GPUEnabled = 1;

	fileImporter fImport(graphFileName, labelsFileName);
	Graph<float, float> test(&fImport, !GPUEnabled);
	LOG(TRACE) << "Nodi: " << test.getStruct()->nNodes << " - Archi: " << test.getStruct()->nEdges;
	LOG(TRACE) << "minDeg: " << test.getMinNodeDeg() << " - maxDeg: " << test.getMaxNodeDeg() << " - meanDeg: "
		<< test.getMeanNodeDeg();

	newComponents(&test);

	//// CPU greedy coloring
	//// Don't know if this still works...
	// Graph<col, col> graph( N, GPUEnabled );  	// random graph
	// ColoringGeedyCPU<col,col> colGreedyCPU(&graph);
	// colGreedyCPU.run();
	// cout << "Greedy-CPU coloring elapsed time: " << colGreedyCPU.getElapsedTime() << "(sec)" << endl;
	//colGreedyCPU.print(0);

	Graph<float, float> graph_d(&test);
	//// GPU Luby coloring
	GPURand GPURandGen(test.getStruct()->nNodes, (long)commandLine.seed);

	/*ColoringLuby<float, float> colLuby(&graph_d, GPURandGen.randStates);
	start = std::clock();
	colLuby.run_fast();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << TXT_NORML;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU elapsed time: " << duration << TXT_NORML;*/

	ColoringMCMCParams params;
	params.nCol = 80;	//test.getMaxNodeDeg() / 2.0f;
	params.epsilon = 1e-8f;
	params.lambda = 0.002f;
	params.ratioFreezed = 1e-2;
	params.maxRip = 250;

	ColoringMCMC_CPU<float, float> mcmc_cpu(&test, params, seed);
	g_debugger = new dbg(&test, &mcmc_cpu);
	start = std::clock();
	mcmc_cpu.run();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	LOG(TRACE) << TXT_BIYLW << "MCMC_CPU elapsed time: " << duration << TXT_NORML;

	ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, params);

	start = std::clock();
	colMCMC.run();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	LOG(TRACE) << TXT_BIYLW << "Elapsed time: " << duration << TXT_NORML;

	if (g_debugger != nullptr)
		delete g_debugger;

	return EXIT_SUCCESS;
}

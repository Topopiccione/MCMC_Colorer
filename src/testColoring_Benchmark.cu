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

#ifdef WIN32
#define INLINE_ARGS
#endif

#define WRITE
#ifdef WRITE
#ifdef WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif
#endif // WRITE


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
	el::Configurations conf("logger.conf");
	el::Loggers::reconfigureLogger("default", conf);
	el::Loggers::reconfigureAllLoggers(conf);

	el::Configuration * loggerConf = conf.get(el::Level::Trace, el::ConfigurationType::Enabled);
	g_traceLogEn = (loggerConf->value() == "true");
	// Debugger pre-init
	g_debugger = nullptr;

#ifdef INLINE_ARGS
	//argc = 7;
	/*argv[1] = "--data";
	argv[2] = "net50k001.txt";
	argv[3] = "--label";
	argv[4] = "lab50k001.txt";
	argv[5] = "--gene";
	argv[6] = "gene.txt";*/

	argc = 5;
	argv[1] = "--simulate";
	argv[2] = "0.001";
	argv[3] = "-n";
	argv[4] = "25000";
#endif // INLINE_ARGS

	// Commandline arguments
	ArgHandle commandLine(argc, argv);
	commandLine.processCommandLine();
	uint32_t N;
	float prob;

	if (commandLine.simulate) {
		N = commandLine.n;
		//M = commandLine.m;
		prob = (float)commandLine.prob;
	}
	uint32_t			seed = commandLine.seed;
	std::string			graphFileName = commandLine.dataFilename;
	std::string			labelsFileName = commandLine.labelFilename;
	float				numColorRatio = 1.0f / (float) commandLine.numColRatio;
	int					repet = commandLine.repetitions;

	for (int i = 0; i < repet; i++)
	{
		std::cout << "Ripetizione: " << i << std::endl;

		std::clock_t start;
		double duration;

		bool GPUEnabled = 1;
		Graph<float, float> *	test;
		fileImporter 		*	fImport;

		if (commandLine.simulate)
			test = new Graph<float, float>(N, prob, i);
		else {
			fImport = new fileImporter(graphFileName, labelsFileName);
			test = new Graph<float, float>(fImport, !GPUEnabled);
		}
		//LOG(TRACE) << "Nodi: " << test->getStruct()->nNodes << " - Archi: " << test->getStruct()->nEdges;
		//LOG(TRACE) << "minDeg: " << test->getMinNodeDeg() << " - maxDeg: " << test->getMaxNodeDeg() << " - meanDeg: "
		//<< test->getMeanNodeDeg();
		std::cout << "Nodi: " << test->getStruct()->nNodes << " - Archi: " << test->getStruct()->nEdges << std::endl;
		std::cout << "minDeg: " << test->getMinNodeDeg() << " - maxDeg: " << test->getMaxNodeDeg() << " - meanDeg: "
			<< test->getMeanNodeDeg() << std::endl;

		newComponents(test);

#ifdef WRITE
		std::string directory = std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->prob) + "-" + std::to_string(numColorRatio) + "-results";
#ifdef WIN32
		mkdir(directory.c_str());
#else
		mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
#endif // WRITE

		Graph<float, float> graph_d(test);

		GPURand GPURandGen(test->getStruct()->nNodes, (long)commandLine.seed);

		//// GPU Luby coloring
		ColoringLuby<float, float> colLuby(&graph_d, GPURandGen.randStates);
		start = std::clock();
		colLuby.run_fast();
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		LOG(TRACE) << TXT_BIYLW << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << TXT_NORML;
		LOG(TRACE) << TXT_BIYLW << "LubyGPU elapsed time: " << duration << TXT_NORML;
		std::cout << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << std::endl;
		std::cout << "LubyGPU elapsed time: " << duration << std::endl;

#ifdef WRITE
		std::ofstream lubyFile;
		lubyFile.open(directory + "/" + std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->prob) + "-LUBY-" + std::to_string(i) + ".txt");
		colLuby.saveStats(i, duration, lubyFile);
		lubyFile.close();
#endif // WRITE


		ColoringMCMCParams params;
		params.nCol = numColorRatio * ((N * prob > 0) ? N * prob : 1);
		params.numColorRatio = numColorRatio;
		//params.nCol = test->getMaxNodeDeg();
		//params.nCol = 200;
		//params.nCol = 80;
		params.startingNCol = 50; //used only with DYNAMIC_N_COLORS
		//params.startingNCol = 20;
		params.epsilon = 1e-8f;
		params.lambda = 0.1f;
		//params.lambda = test->getStruct()->nNodes * log( params.epsilon );
		params.ratioFreezed = 1e-2;
		//params.maxRip = params.nCol;
		params.maxRip = 500;
		//params.maxRip = 5000;

		ColoringMCMC_CPU<float, float> mcmc_cpu(test, params, seed);
		g_debugger = new dbg(test, &mcmc_cpu);
		start = std::clock();
		mcmc_cpu.run();
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		//mcmc_cpu.show_histogram();
		//LOG(TRACE) << TXT_BIYLW << "MCMC_CPU elapsed time: " << duration << TXT_NORML;
		std::cout << "MCMC_CPU elapsed time: " << duration << std::endl;

#ifdef WRITE
		std::ofstream cpuFile;
		cpuFile.open(directory + "/" + std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->prob) + "-MCMC_CPU-" + std::to_string(i) + ".txt");
		mcmc_cpu.saveStats(i, duration, cpuFile);
		cpuFile.close();
#endif // WRITE

		ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, params);

		start = std::clock();
		colMCMC.run(i);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		//LOG(TRACE) << TXT_BIYLW << "Elapsed time: " << duration << TXT_NORML;
		std::cout << "MCMC Elapsed time: " << duration << std::endl;
		std::cout << std::endl;

		if (g_debugger != nullptr)
			delete g_debugger;

		delete test;
		if (!commandLine.simulate)
			delete fImport;
	}

	return EXIT_SUCCESS;
}

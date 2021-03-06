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

// #define MCMC_CPU
// #define LUBY
#define MCMC_GPU


//argomenti tipo : --data net.txt --label lab.txt --gene gene.txt

bool		g_traceLogEn;	// Declared in utils/miscUtils.h
dbg		*	g_debugger;

INITIALIZE_EASYLOGGINGPP

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
	uint32_t a = 1;

	argv[a++] = "--simulate";
	argv[a++] = "0.001";
	argv[a++] = "-n";
	argv[a++] = "50000";
	argv[a++] = "--repet";
	argv[a++] = "1";
	argv[a++] = "--numColRatio";
	argv[a++] = "1.0";
	argv[a++] = "--tabooIteration";
	argv[a++] = "4";

	argc = a++;
#endif // INLINE_ARGS

	// Commandline arguments
	ArgHandle commandLine(argc, argv);
	commandLine.processCommandLine();
	uint32_t N;
	float prob;

	if (commandLine.simulate) {
		N = commandLine.n;
		prob = (float)commandLine.prob;
	}
	uint32_t			seed = commandLine.seed;
	uint32_t			nColFromC = commandLine.nCol;
	std::string			graphFileName = commandLine.dataFilename;
	std::string			labelsFileName = commandLine.labelFilename;
	float				numColorRatio = 1.0f / (float)commandLine.numColRatio;
	std::cout << "NUMCOLORRATIO +++++++++++++++++++ " << (float)commandLine.numColRatio << std::endl;
	int					repet = commandLine.repetitions;
	std::cout << "REPET +++++++++++++++++++ " << repet << std::endl;

	bool GPUEnabled = 1;
	Graph<float, float> *	test;
	fileImporter 		*	fImport;

	if (commandLine.simulate)
		test = new Graph<float, float>(N, prob, seed);
	else {
		fImport = new fileImporter(graphFileName, labelsFileName);
		test = new Graph<float, float>(fImport, !GPUEnabled);
		// Eh, fix per impostare la probabilita' nel caso in cui il grafo venga letto da file
		prob = test->getStruct()->nEdges / (float) (test->getStruct()->nNodes * test->getStruct()->nNodes);
		N = test->getStruct()->nNodes;
	}
	LOG(TRACE) << "Nodi: " << test->getStruct()->nNodes << " - Archi: " << test->getStruct()->nEdges;
	LOG(TRACE) << "minDeg: " << test->getMinNodeDeg() << " - maxDeg: " << test->getMaxNodeDeg() << " - meanDeg: "
		<< test->getMeanNodeDeg();
	std::cout << "Nodi: " << test->getStruct()->nNodes << " - Archi: " << test->getStruct()->nEdges << std::endl;
	std::cout << "minDeg: " << test->getMinNodeDeg() << " - maxDeg: " << test->getMaxNodeDeg() << " - meanDeg: "
		<< test->getMeanNodeDeg() << std::endl;

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

	for (int i = 0; i < repet; i++)
	{
		std::cout << "Ripetizione: " << i << std::endl;

		std::clock_t start;
		double duration;

#ifdef LUBY
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
#endif // LUBY

		ColoringMCMCParams params;
		//params.nCol = numColorRatio * ((N * prob > 0) ? N * prob : 1);
		params.numColorRatio = numColorRatio;
		params.nCol = (nColFromC != 0) ? nColFromC : test->getMaxNodeDeg() * numColorRatio;
		params.epsilon = 1e-8f;
		params.lambda = 1.0f;
		//params.lambda = test->getStruct()->nNodes * log( params.epsilon );
		params.ratioFreezed = 1e-2;
		params.maxRip = 250;
		params.tabooIteration = commandLine.tabooIteration;

#ifdef MCMC_CPU
		ColoringMCMC_CPU<float, float> mcmc_cpu(test, params, seed + i);
		g_debugger = new dbg(test, &mcmc_cpu);
		start = std::clock();
		mcmc_cpu.run();
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		// mcmc_cpu.show_histogram();
		LOG(TRACE) << TXT_BIYLW << "MCMC_CPU elapsed time: " << duration << TXT_NORML;
		std::cout << "MCMC_CPU elapsed time: " << duration << std::endl;

#ifdef WRITE
		std::ofstream cpuFile;
		cpuFile.open(directory + "/" + std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->prob) + "-MCMC_CPU-" + std::to_string(i) + ".txt");
		mcmc_cpu.saveStats(i, duration, cpuFile);
		cpuFile.close();
#endif // WRITE
#endif // MCMC_CPU

#ifdef MCMC_GPU
		ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, params);

#ifdef WRITE
		colMCMC.setDirectoryPath(directory);
#endif

		start = std::clock();
		colMCMC.run(i);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		LOG(TRACE) << TXT_BIYLW << "Elapsed time: " << duration << TXT_NORML;
		std::cout << "MCMC Elapsed time: " << duration << std::endl;
		std::cout << std::endl;
#endif // MCMC_GPU

		if (g_debugger != nullptr)
			delete g_debugger;
	}


	delete test;
	if (!commandLine.simulate)
		delete fImport;


	return EXIT_SUCCESS;
}

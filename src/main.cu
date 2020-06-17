// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include <ctime>
#include "utils/ArgHandle.h"
#include "utils/fileImporter.h"
#include "utils/miscUtils.h"
#include "utils/dbg.h"
#include "graph/graph.h"
#include "graph/graphCPU.cpp"
#include "graph/graphGPU.cu"
#include "graph_coloring/coloring.h"
#include "graph_coloring/coloringMCMC_CPU.h"
#include "graph_coloring/coloringLuby.h"

#include "graph_coloring/coloringGreedyFF.h"
#include "graph_coloring/coloringVFF.h"

#include "graph_coloring/coloringMCMC.h"
#include "GPUutils/GPURandomizer.h"
#include "easyloggingpp/easylogging++.h"

bool		g_traceLogEn;	// Declared in utils/miscUtils.h
dbg		*	g_debugger;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
	//cudaDeviceReset();
	////EasyLogging++
	START_EASYLOGGINGPP(argc, argv);
	checkLoggerConfFile();
	el::Configurations conf("logger.conf");
	el::Loggers::reconfigureLogger("default", conf);
	el::Loggers::reconfigureAllLoggers(conf);

	el::Configuration * loggerConf = conf.get(el::Level::Trace, el::ConfigurationType::Enabled);
	g_traceLogEn = (loggerConf->value() == "true");

	// Debugger pre-init
	g_debugger = nullptr;

	// Commandline arguments
	ArgHandle commandLine(argc, argv);
	commandLine.processCommandLine();
	uint32_t				N;
	float					prob;

	uint32_t				seed			= commandLine.seed;
	uint32_t				nColFromC		= commandLine.nCol;
	std::string				graphFileName	= commandLine.graphFilename;
	std::string				outDir			= commandLine.outDir;
	float					numColorRatio	= 1.0f / (float) commandLine.numColRatio;
	uint32_t				repet			= commandLine.repetitions;

	bool GPUEnabled = 1;
	Graph<float, float> *	test;
	fileImporter 		*	fImport;

	if (commandLine.simulate) {
		N = commandLine.n;
		prob = (float)commandLine.prob;
		test = new Graph<float, float>(N, prob, seed);
	} else {
		fImport = new fileImporter(graphFileName, "");
		test = new Graph<float, float>(fImport, !GPUEnabled);
		// Eh, fix per impostare la probabilita' nel caso in cui il grafo venga letto da file
		prob = test->getStruct()->nEdges / (float) (test->getStruct()->nNodes * test->getStruct()->nNodes);
		N = test->getStruct()->nNodes;
	}
	LOG(TRACE) << "Nodes: " << test->getStruct()->nNodes << " - Edges: " << test->getStruct()->nEdges;
	LOG(TRACE) << "Min Degree: " << test->getMinNodeDeg() << " - Max Degree: " << test->getMaxNodeDeg() << " - Mean Degree: "
		<< test->getMeanNodeDeg();
	std::cout << "Nodes: " << test->getStruct()->nNodes << " - Edges: " << test->getStruct()->nEdges << std::endl;
	std::cout << "Min Degree: " << test->getMinNodeDeg() << " - Max Degree: " << test->getMaxNodeDeg() << " - Mean Degree: "
		<< test->getMeanNodeDeg() << std::endl;

	Graph<float, float> graph_d(test);

	GPURand GPURandGen(test->getStruct()->nNodes, (long)commandLine.seed);

	for (int i = 0; i < repet; i++)
	{
		std::cout << "Repetition: " << i << std::endl;

		std::clock_t start;
		double duration;

		//// GPU Luby coloring
		if (commandLine.lubygpu) {
			ColoringLuby<float, float> colLuby(&graph_d, GPURandGen.randStates);

			start = std::clock();
			colLuby.run_fast();
			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

			LOG(TRACE) << TXT_BIYLW << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << TXT_NORML;
			LOG(TRACE) << TXT_BIYLW << "LubyGPU elapsed time: " << duration << TXT_NORML;
			std::cout << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << std::endl;
			std::cout << "LubyGPU elapsed time: " << duration << std::endl;
			// Saving log
			std::ofstream lubyFileLog, lubyFileColors;
			lubyFileLog.open(outDir + "/" + commandLine.graphName + "-LUBY-" + std::to_string(i) + ".log");
			colLuby.saveStats(i, duration, lubyFileLog);
			lubyFileLog.close();
			lubyFileColors.open(outDir + "/" + commandLine.graphName + "-LUBY-" + std::to_string(i) + "-colors.txt");
			colLuby.saveColor(lubyFileColors);
			lubyFileColors.close();
		}

		if (commandLine.greedyff) {
			ColoringGreedyFF<float, float> greedy(&graph_d);

			start = std::clock();
			greedy.run();
			duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);

			LOG(TRACE) << TXT_BIYLW << "Parallel Greedy First Fit - number of colors: " << greedy.getColoring()->nCol << TXT_NORML;
			LOG(TRACE) << TXT_BIYLW << "Parallel Greedy First Fit - elapsed time: " << duration << TXT_NORML;
			std::cout << "Parallel Greedy First Fit - number of colors: " << greedy.getColoring()->nCol << std::endl;
			std::cout << "Parallel Greedy First Fit - elapsed time: " << duration << std::endl;

			// Saving log
			// std::ofstream gffFileLog, gffFileColors;
			// gffFileLog.open(outDir + "/" + commandLine.graphName + "-GFF-" + std::to_string(i) + ".log");
			// greedyff.saveStats(i, duration, lubyFileLog);
			// gffFileLog.close();
			// lubyFileColors.open(outDir + "/" + commandLine.graphName + "-GFF-" + std::to_string(i) + "-colors.txt");
			// colLuby.saveColor(lubyFileColors);
			// lubyFileColors.close();
		}

		if (commandLine.rebalanced_greedyff) {
			ColoringVFF<float, float> balanced(&graph_d);

			start = std::clock();
			balanced.run();
			duration = (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);

			LOG(TRACE) << TXT_BIYLW << "Vertex-centric First Fit rebalancing - number of colors: " << balanced.getColoring()->nCol << TXT_NORML;
			LOG(TRACE) << TXT_BIYLW << "Vertex-centric First Fit rebalancing - elapsed time: " << duration << TXT_NORML;
			std::cout << "Vertex-centric First Fit rebalancing - number of colors: " << balanced.getColoring()->nCol << std::endl;
			std::cout << "Vertex-centric First Fit rebalancing - elapsed time: " << duration << std::endl;
		}

		// TODO: Some of these should be made user-definable from command line
		ColoringMCMCParams params;
		params.numColorRatio	= numColorRatio;
		params.nCol				= (nColFromC != 0) ? nColFromC : test->getMaxNodeDeg() * numColorRatio;
		params.epsilon			= 1e-8f;
		params.lambda			= 1.0f;
		params.ratioFreezed		= 1e-2;
		params.maxRip			= 250;
		params.tabooIteration	= commandLine.tabooIteration;
		params.tailcut			= commandLine.tailcut;

		if (commandLine.mcmccpu) {
			ColoringMCMC_CPU<float, float> mcmc_cpu(test, params, seed + i);
			g_debugger = new dbg(test, &mcmc_cpu);

			start = std::clock();
			mcmc_cpu.run();
			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

			// TODO: command line option for showing color histogram
			// mcmc_cpu.show_histogram();
			LOG(TRACE) << TXT_BIYLW << "MCMC_CPU elapsed time: " << duration << TXT_NORML;
			std::cout << "MCMC_CPU elapsed time: " << duration << std::endl;
			// Saving log
			std::ofstream cpuFileLog, cpuFileColors;
			cpuFileLog.open(outDir + "/" + commandLine.graphName + "-MCMC_CPU-" + std::to_string(i) + ".log");
			mcmc_cpu.saveStats(i, duration, cpuFileLog);
			cpuFileLog.close();
			cpuFileColors.open(outDir + "/" + commandLine.graphName + "-MCMC_CPU-" + std::to_string(i) + "-colors.txt");
			mcmc_cpu.saveColor(cpuFileColors);
			cpuFileColors.close();
		}

		if (commandLine.mcmcgpu) {
			ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, params);
			colMCMC.setDirectoryPath(outDir + "/" + commandLine.graphName + "-MCMC_GPU-" + std::to_string(i));

			start = std::clock();
			colMCMC.run(i);
			duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

			LOG(TRACE) << TXT_BIYLW << "MCMC GPU elapsed time: " << duration << TXT_NORML;
			std::cout << "MCMC GPU elapsed time: " << duration << std::endl << std::endl;
		}

		if (g_debugger != nullptr)
			delete g_debugger;
	}


	delete test;
	if (!commandLine.simulate)
		delete fImport;


	return EXIT_SUCCESS;
}

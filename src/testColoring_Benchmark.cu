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

#include <fstream>

// Nich, in dbg.h e' presente una funzione per lo split delle stringhe
vector<string> split(const std::string& str, const std::string& delim)
{
	vector<std::string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == std::string::npos) pos = str.length();
		std::string token = str.substr(prev, pos - prev);
		if (!token.empty()) tokens.push_back(token);
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());
	return tokens;
}

void combineFiles() {
	std::string line;
	std::string directory = "250000-312585727";
	std::map< std::string, float> tempMap[10];
	std::map< std::string, float> finalMap;

	std::vector<std::string> keys = {
		"numCol", "epsilon", "lambda", "ratioFreezed", "maxRip", "start_used_colors", "start_available_colors", "start_most_used_colors", "start_most_used_colors_n_times", "start_least_used_colors", "start_least_used_colors_n_times", "start_average", "start_variance", "start_standard_deviation", "time", "end_used_colors", "end_available_colors", "end_most_used_colors", "end_most_used_colors_n_times", "end_least_used_colors", "end_least_used_colors_n_times", "end_average", "end_variance", "end_standard_deviation"
	};

	for (int i = 0; i < 10; i++)
	{
		ifstream myfile(directory + "-results/" + directory + "-resultsFile-" + std::to_string(i) + ".txt");

		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				vector<string> v = split(line, " ");
				tempMap[i][v[0]] = std::stof(v[1]);
			}
			myfile.close();
		}
	}

	for (std::string key : keys)
	{
		finalMap[key] = 0.;
	}

	for (int i = 0; i < 10; i++)
	{
		for (std::string key : keys)
		{
			//std::cout << key << " " << tempMap[i][key] << std::endl;
			finalMap[key] += tempMap[i][key];
		}

		//std::cout << "*************" << std::endl;
	}

	std::ofstream finalFile;
	finalFile.open(directory + "-results/" + directory + "-FINAL" + ".txt");
	for (std::string key : keys)
	{
		finalMap[key] /= 10;
		std::cout << key << " " << finalMap[key] << std::endl;
		finalFile << key << " " << finalMap[key] << std::endl;
	}

	finalFile.close();
}

int main(int argc, char *argv[]) {

	//combineFiles();
	//return EXIT_SUCCESS;

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
	argc = 7;
	argv[1] = "--data";
	argv[2] = "net25k001.txt";
	argv[3] = "--label";
	argv[4] = "lab25k001.txt";
	argv[5] = "--gene";
	argv[6] = "gene.txt";
#endif // INLINE_ARGS

	// Commandline arguments
	ArgHandle commandLine(argc, argv);
	commandLine.processCommandLine();
	uint32_t N, M;
	float prob;

	if (commandLine.simulate) {
		N				= commandLine.n;
		M				= commandLine.m;
		prob			= (float) commandLine.prob;
	}
	uint32_t			seed			= commandLine.seed;
	std::string			graphFileName	= commandLine.dataFilename;
	std::string			labelsFileName	= commandLine.labelFilename;

	//seed = 10000;

	std::clock_t start;
	double duration;

	bool GPUEnabled = 1;
	Graph<float, float> *	test;
	fileImporter 		*	fImport;

	if (commandLine.simulate)
		test	= new Graph<float, float>(N, prob, 1235);
	else {
		fImport	= new fileImporter(graphFileName, labelsFileName);
		test	= new Graph<float, float> (fImport, !GPUEnabled);
	}
	LOG(TRACE) << "Nodi: " << test->getStruct()->nNodes << " - Archi: " << test->getStruct()->nEdges;
	LOG(TRACE) << "minDeg: " << test->getMinNodeDeg() << " - maxDeg: " << test->getMaxNodeDeg() << " - meanDeg: "
		<< test->getMeanNodeDeg();

	newComponents( test );

#ifdef WRITE
	std::string directory = std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->getStruct()->nCleanEdges) + "-results";
#ifdef WIN32
	mkdir(directory.c_str());
#else
	mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
#endif // WRITE

	//// CPU greedy coloring
	//// Don't know if this still works...
	// Graph<col, col> graph( N, GPUEnabled );  	// random graph
	// ColoringGeedyCPU<col,col> colGreedyCPU(&graph);
	// colGreedyCPU.run();
	// cout << "Greedy-CPU coloring elapsed time: " << colGreedyCPU.getElapsedTime() << "(sec)" << endl;
	// colGreedyCPU.print(0);

	Graph<float, float> graph_d( test );

	GPURand GPURandGen(test->getStruct()->nNodes, (long)commandLine.seed);

	//// GPU Luby coloring
	ColoringLuby<float, float> colLuby(&graph_d, GPURandGen.randStates);
	start = std::clock();
	colLuby.run_fast();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU - number of colors: " << colLuby.getColoringGPU()->nCol << TXT_NORML;
	LOG(TRACE) << TXT_BIYLW << "LubyGPU elapsed time: " << duration << TXT_NORML;

#ifdef WRITE
	std::ofstream lubyFile;
	lubyFile.open(directory + "/" + std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->getStruct()->nCleanEdges) + "-LUBY" + ".txt");
	lubyFile << "nCol" << " " << colLuby.getColoringGPU()->nCol << std::endl;
	lubyFile << "time" << " " << duration << std::endl;
	lubyFile.close();
#endif // WRITE


	ColoringMCMCParams params;
	params.nCol = test->getMaxNodeDeg();
	//params.nCol = 200;
	//params.nCol = 80;
	params.startingNCol = 50; //used only with DYNAMIC_N_COLORS
	//params.startingNCol = 20;
	params.epsilon = 1e-8f;
	params.lambda = 0.01f;
	//params.lambda = test->getStruct()->nNodes * log( params.epsilon );
	params.ratioFreezed = 1e-2;
	params.maxRip = 10000;
	//params.maxRip = 4;
	//params.maxRip = 5000;

	ColoringMCMC_CPU<float, float> mcmc_cpu(test, params, seed);
	g_debugger = new dbg(test, &mcmc_cpu);
	start = std::clock();
	mcmc_cpu.run();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	//mcmc_cpu.show_histogram();
	LOG(TRACE) << TXT_BIYLW << "MCMC_CPU elapsed time: " << duration << TXT_NORML;

#ifdef WRITE
	std::ofstream cpuFile;
	cpuFile.open(directory + "/" + std::to_string(test->getStruct()->nNodes) + "-" + std::to_string(test->getStruct()->nCleanEdges) + "-MCMC_CPU" + ".txt");
	cpuFile << "time" << " " << duration << std::endl;
	cpuFile.close();
#endif // WRITE

	ColoringMCMC<float, float> colMCMC(&graph_d, GPURandGen.randStates, params);

	for (int i = 0; i < 10; i++)
	{
		std::cout << "Iterazione: " << i << std::endl;

		start = std::clock();
		colMCMC.run(i);
		//colMCMC.run(0);
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		LOG(TRACE) << TXT_BIYLW << "Elapsed time: " << duration << TXT_NORML;
		std::cout << std::endl;
	}

	if (g_debugger != nullptr)
		delete g_debugger;

	delete test;
	if (!commandLine.simulate)
		delete fImport;

	return EXIT_SUCCESS;
}

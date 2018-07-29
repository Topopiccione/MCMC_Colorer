// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include <cstdio>
#include <ctime>
#include "utils/ArgHandle.h"
#include "utils/fileImporter.h"
#include "graph/graph.h"
#include "graph/graphCPU.cpp"
#include "graph/graphGPU.cu"
#include "graph_coloring/coloring.h"
#include "GPUutils/GPURandomizer.h"

#define BENCHMARK 1

using namespace std;

int main(int argc, char *argv[]) {

	ArgHandle commandLine( argc, argv );
	commandLine.processCommandLine();

	uint32_t			N				= commandLine.n;
	uint32_t			M				= commandLine.m;
	float				prob			= (float) commandLine.prob;
	std::string			graphFileName	= commandLine.dataFilename;
	std::string			labelsFileName	= commandLine.labelFilename;

	std::clock_t start;
	double mean=0.0;
	double duration;

	bool GPUEnabled = 1;

/*	// CPU greedy coloring
	Graph<col, col> graph( N, GPUEnabled );  	// random graph
	ColoringGeedyCPU<col,col> colGreedyCPU(&graph);
	colGreedyCPU.run();
	cout << "Greedy-CPU coloring elapsed time: " << colGreedyCPU.getElapsedTime() << "(sec)" << endl;
	//colGreedyCPU.print(0);
	*/

	fileImporter fImport( graphFileName, labelsFileName );
	Graph<float, float> test( &fImport, GPUEnabled );
	std::cout << "Nodi: " << test.getStruct()->nNodes << " - Archi: " << test.getStruct()->nEdges << std::endl;

	/*for (auto it = fImport.inverseGeneMap.begin(); it != fImport.inverseGeneMap.end(); ++it) {
		std::cout << it->first << " " << it->second << std::endl;
	}
	std::cout << std::endl << std::endl;
*/
/*	fImport.getNextLabelling();
	for (uint32_t i = 0; i < fImport.nLabelsFromFile; i++)
		std::cout << fImport.labelsFromFile[i] << " " << fImport.inverseGeneMap[fImport.labelsFromFile[i]] << std::endl;
	std::cout << std::endl << std::endl;

	fImport.getNextLabelling();
	for (uint32_t i = 0; i < fImport.nLabelsFromFile; i++)
		std::cout << fImport.labelsFromFile[i] << " " << fImport.inverseGeneMap[fImport.labelsFromFile[i]] << std::endl;
	std::cout << std::endl << std::endl;

	fImport.getNextLabelling();
	for (uint32_t i = 0; i < fImport.nLabelsFromFile; i++)
		std::cout << fImport.labelsFromFile[i] << " " << fImport.inverseGeneMap[fImport.labelsFromFile[i]] << std::endl;
	std::cout << std::endl << std::endl;
*/

	/*for (uint32_t i = 0; i < BENCHMARK; i++) {
		std::cout << "Ciclo: " << i << std::endl;

		default_random_engine eng{i}; // fixed seed
		GPURand GPURandGen( N, (long) i );

		Graph<col,col> graph(N,GPUEnabled);  	// random graph
		graph.randGraphUnweighted(prob,eng);
		//graph.print(1);  // print from CPU

		ColoringLuby<col,col> colLuby(&graph, GPURandGen.randStates);
		start = std::clock();
		colLuby.run();
		duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

		mean+=duration;
		std::cout << "Elapsed time: " << duration << std::endl;

		int numNod=graph.getStruct()->nNodes;

		std::cout << "nNodes: " << numNod << std::endl;
		//cudaMalloc( (void**)&weight_d, nnodes * sizeof(  ) );
		//cudaMalloc( (void**)&threshold_d, nnodes * sizeof( int ) );
		//std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
		//std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
		//for(int j=0; j<)
		//std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
	}*/

	mean /= BENCHMARK;
	std::cout << "Mean time: " << mean << std::endl;

	return EXIT_SUCCESS;
}

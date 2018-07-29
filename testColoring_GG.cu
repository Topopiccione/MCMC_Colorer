// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include "graph/graph.h"
#include "graph/graphCPU.cpp"
#include "graph/graphGPU.cu"
#include "graph_coloring/coloring.h"
#include "GPUutils/GPURandomizer.h"

using namespace std;

int main(int argc, char *argv[]) {

	node_sz N = 10;				      // number of nodes for random graphs
	float prob = .2;				  // density (percentage) for random graphs
	default_random_engine eng{}; // fixed seed
	//curandState *state;
	GPURand GPURandGen( N, 123456 );

	if (argc >= 2)
		N = atoi(argv[1]);
	unsigned int p;
	if (argc >= 3) {
		p = atoi(argv[2]);
		prob = (float) p / 100.0f;
	}
	bool GPUEnabled = 1;
	Graph<col,col> graph(N,GPUEnabled);  	// random graph
	graph.randGraphUnweighted(prob,eng);
	graph.print(1);  // print from CPU

	// CPU greedy coloring
	ColoringGeedyCPU colGreedyCPU(&graph);
	colGreedyCPU.run();
	cout << "Greedy-CPU coloring elapsed time: " << colGreedyCPU.getElapsedTime() << "(sec)" << endl;
	//colGreedyCPU.print(0);

	// GPU MCMCM coloring
	/*ColoringMCMCGPU colMCMC(&graph);
	colMCMC.setLambda(1);
	colMCMC.setEpsilon(.0001f/(float)N);
	colMCMC.setFreezed(1);
	colMCMC.run();
	cout << "MCMC-GPU coloring elapsed time: " << colMCMC.getElapsedTime() << "(sec)" << endl;

	colMCMC.print(1);*/

	ColoringLuby colLuby(&graph, GPURandGen.randStates);
	colLuby.run();

	return EXIT_SUCCESS;
}

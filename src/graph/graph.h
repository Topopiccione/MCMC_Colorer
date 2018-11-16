#pragma once
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cinttypes>
#include "utils/fileImporter.h"
#include "GPUutils/GPURandomizer.h"

typedef uint32_t node;     // graph node
typedef uint32_t node_sz;

typedef float Prob;

// This shold be moved somewhere else
enum GPUMEMINITTYPES {
	GPUINIT_NODES = 1,
	GPUINIT_EDGES = 2,
	GPUINIT_NODEW = 3,
	GPUINIT_EDGEW = 4,
	GPUINIT_NODET = 5,
	GPUINIT_CEDGES = 6
};

/**
 * Base structure (array 1D format) of a graph with weighted nodes/edges
 */
template<typename nodeW, typename edgeW> struct GraphStruct {
	node			nNodes{ 0 };				// num of graph nodes
	node_sz			nEdges{ 0 };				// num of graph edges
	node_sz			nCleanEdges{ 0 };				// num of graph clean edges
	node_sz		*	cumulDegs{ nullptr };		// cumsum of node degrees
	node		*	neighs{ nullptr };			// list of neighbors for all nodes (edges)
	node_sz		*	edges{ nullptr };			// list of edges
	nodeW		*	nodeWeights{ nullptr };		// list of weights for all nodes
	edgeW		*	edgeWeights{ nullptr };		// list of weights for all edges
	nodeW		*	nodeThresholds{ nullptr };

	~GraphStruct() {
		if (neighs != nullptr)			delete[] neighs;
		if (cumulDegs != nullptr)		delete[] cumulDegs;
		if (edges != nullptr)			delete[] edges;
		if (nodeWeights != nullptr)		delete[] nodeWeights;
		if (edgeWeights != nullptr)		delete[] edgeWeights;
		if (nodeThresholds != nullptr)	delete[] nodeThresholds;
	}

	class Invalid {};

	bool is_valid() {
		for (uint32_t i = 0; i < nEdges; i++)
			if (neighs[i] > nNodes - 1)   // inconsistent neighbor index
				return false;
		if (cumulDegs[nNodes] != nEdges)  // inconsistent number of edges
			return false;
		return true;
	};

	/// return the degree of node i
	inline node_sz deg(node i) {
		return ( cumulDegs[i + 1] - cumulDegs[i] );
	}

	/// check whether node i is a neighbor of node j
	bool areNeighbor(node i, node j) {
		for (uint32_t k = 0; k < deg(j); k++) {
			if (neighs[cumulDegs[j]+k] == i)
				return true;
		}
		return false;
	}

};

/**
 * It manages a graph for CPU & GPU
 */
template<typename nodeW, typename edgeW> class Graph {
	float density{ 0.0f };	               /// Probability of an edge (Erdos graph)
	GraphStruct<nodeW, edgeW>* str{};      /// graph structure
	node maxDeg{ 0 };
	node minDeg{ 0 };
	float meanDeg{ 0.0f };
	bool connected{ true };
	bool GPUEnabled{ false };

public:
	Graph( node nn, bool GPUEnb ) : GPUEnabled{ GPUEnb } { setup(nn); }
	Graph( fileImporter * imp, bool GPUEnb );
	Graph( const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
		GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F,
		const float * const thresholds, bool GPUEnb );
	Graph( Graph<nodeW, edgeW> * const fullGraph ); // takes a graph from CPU and copy to GPU
	~Graph() {if (GPUEnabled) deleteMemGPU(); else delete str;};

	void setup(node);
	void setupImporter();
	void setupRedux( const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
		GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F, const float * const thresholds );

	void setupImporterGPU();
	void setupReduxGPU( const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
		GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F, const float * const thresholds );

	void randGraphUnweighted(float prob, std::default_random_engine&);  /// generate an Erdos random graph
	void print(bool);
	void print_d(bool);
	GraphStruct<nodeW,edgeW>* getStruct() {return str;}
	GraphStruct<nodeW,edgeW>* getStruct() const {return str;}
	void setMemGPU(node_sz nn, int mode);
	void deleteMemGPU();
	bool isGPUEnabled() { return GPUEnabled; }
	node getMaxNodeDeg() { return maxDeg; };
	node getMinNodeDeg() { return minDeg; };
	float getMeanNodeDeg() { return meanDeg; };
	void efficiency();

private:
	fileImporter		*	fImport;
};

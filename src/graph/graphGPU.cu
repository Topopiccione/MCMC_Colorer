// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <stdio.h>
#include <type_traits>
#include "graph.h"
#include "GPUutils/GPUutils.h"

using namespace std;

namespace Graph_k {
	template<typename nodeW, typename edgeW> __global__ void print_d(GraphStruct<nodeW, edgeW>*, bool);
};


/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */

 // TODO: sistemare le dimensioni in modo che siano passate come primo argomento
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setMemGPU(node_sz nn, int mode) {

	cudaError cuSts;
	if (mode == GPUINIT_NODES) {
		str = new GraphStruct<nodeW, edgeW>();
		cuSts = cudaMalloc(&(str->cumulDegs), (nn + 1) * sizeof(node_sz)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphStructSize = sizeof(GraphStruct<nodeW,edgeW>);
		//GPUMemTracker::graphDegsSize   = (nn+1)*sizeof(node_sz);
	}
	else if (mode == GPUINIT_EDGES) {
		cuSts = cudaMalloc(&(str->neighs), str->nEdges * sizeof(node)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphNeighsSize = str->nEdges*sizeof(node);
	}
	else if (mode == GPUINIT_CEDGES) {
		cuSts = cudaMalloc(&(str->edges), str->nCleanEdges * 2 * sizeof(node)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphNeighsSize = str->nEdges*sizeof(node);
	}
	else if (mode == GPUINIT_NODEW) {
		cuSts = cudaMalloc(&(str->nodeWeights), str->nEdges * sizeof(nodeW)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphNodeWSize = str->nEdges*sizeof(nodeW);
	}
	else if (mode == GPUINIT_EDGEW) {
		cuSts = cudaMalloc(&(str->edgeWeights), str->nEdges * sizeof(edgeW)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphEdgeWSize = str->nEdges*sizeof(edgeW);
	}
	else if (mode == GPUINIT_NODET) {
		cuSts = cudaMalloc(&(str->nodeThresholds), str->nNodes * sizeof(nodeW)); cudaCheck(cuSts, __FILE__, __LINE__);
		//GPUMemTracker::graphNodeTSize = str->nNodes*sizeof(nodeW);
	}
}

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupImporterGPU() {

	uint32_t nn = fImport->nNodes;
	setMemGPU(nn, GPUINIT_NODES);
	str->nNodes = nn;
	std::unique_ptr<node_sz[]> temp_cumulDegs(new node_sz[nn + 1]);

#ifdef VERBOSEGRAPH
	std::cout << "Creazione liste temporanee..." << std::endl;
#endif
	std::list<uint32_t>	** tempN = new std::list<uint32_t>*[nn];
	std::list<edgeW>	** tempW = new std::list<edgeW>*[nn];
	for (uint32_t i = 0; i < nn; i++) {
		tempN[i] = new std::list<uint32_t>;
		tempW[i] = new std::list<edgeW>;
	}

	std::cout << "a" << std::endl;

	// Leggo gli archi dal file del grafo
	fImport->fRewind();
	while (fImport->getNextEdge()) {
		if (fImport->edgeIsValid) {
			tempN[fImport->srcIdx]->push_back(fImport->dstIdx);
			tempW[fImport->srcIdx]->push_back((edgeW)fImport->edgeWgh);
			str->nEdges++;
			// anche l'arco di ritorno!
			tempN[fImport->dstIdx]->push_back(fImport->srcIdx);
			tempW[fImport->dstIdx]->push_back((edgeW)fImport->edgeWgh);
			str->nEdges++;
		}
	}

	std::cout << "b" << std::endl;

	// Ora in tempN e tempW ho tutto quello che serve per costruire il grafo
	// Inizio con i cumulDegs
	std::fill(temp_cumulDegs.get(), temp_cumulDegs.get() + (nn + 1), 0);
	for (uint32_t i = 1; i < (nn + 1); i++)
		temp_cumulDegs[i] += (temp_cumulDegs[i - 1] + (uint32_t)(tempN[i - 1]->size()));

	std::cout << "c" << std::endl;

	setMemGPU(str->nEdges, GPUINIT_EDGES);
	setMemGPU(str->nEdges, GPUINIT_EDGEW);
	setMemGPU(nn, GPUINIT_NODET);
	std::unique_ptr<node[]>  temp_neighs(new node[str->nEdges]);
	std::unique_ptr<edgeW[]> temp_edgeWeights(new edgeW[str->nEdges]);
	std::unique_ptr<nodeW[]> temp_nodeThresholds(new nodeW[str->nNodes]);

	std::cout << "d" << std::endl;

	for (uint32_t i = 0; i < nn; i++) {
		uint32_t j = 0;
		for (auto it = tempN[i]->begin(); it != tempN[i]->end(); ++it) {
			temp_neighs[temp_cumulDegs[i] + j] = *it;
			j++;
		}
		j = 0;
		for (auto it = tempW[i]->begin(); it != tempW[i]->end(); ++it) {
			temp_edgeWeights[temp_cumulDegs[i] + j] = *it;
			j++;
		}
	}

	std::cout << "e" << std::endl;

	// max, min, mean deg
	maxDeg = 0;
	minDeg = nn;
	std::cout << "nn: " << nn << std::endl;
	for (uint32_t i = 0; i < nn; i++) {
		std::cout << maxDeg << " ";
		if ((temp_cumulDegs[i + 1] - temp_cumulDegs[i]) > maxDeg)
			maxDeg = (uint32_t)str->deg(i);
		if ((temp_cumulDegs[i + 1] - temp_cumulDegs[i]) < minDeg)
			minDeg = (uint32_t)str->deg(i);
	}
	std::cout << "f" << std::endl;
	density = (float)str->nEdges / (float)(nn * (nn - 1) / 2);
	meanDeg = (float)str->nEdges / (float)nn;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;



	// Copio su GPU
	cudaMemcpy(str->cumulDegs, temp_cumulDegs.get(), (str->nNodes + 1) * sizeof(node_sz), cudaMemcpyHostToDevice);
	cudaMemcpy(str->neighs, temp_neighs.get(), str->nEdges * sizeof(node), cudaMemcpyHostToDevice);
	cudaMemcpy(str->edgeWeights, temp_edgeWeights.get(), str->nEdges * sizeof(edgeW), cudaMemcpyHostToDevice);
	cudaMemcpy(str->nodeThresholds, temp_nodeThresholds.get(), str->nNodes * sizeof(nodeW), cudaMemcpyHostToDevice);

	std::cout << "g" << std::endl;

	// elimino le strutture temporanee
	for (uint32_t i = 0; i < nn; i++) {
		delete tempW[i];
		delete tempN[i];
	}
	delete[] tempW;
	delete[] tempN;

	std::cout << "h" << std::endl;
}


// Questo setup è su con lo sputo. E' un miracolo se funziona.
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupReduxGPU(const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
	GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F, const float * const thresholds) {


	setMemGPU(unlabSize, GPUINIT_NODES);
	str->nNodes = unlabSize;
	str->nEdges = 0;

	std::unique_ptr<node_sz[]> temp_cumulDegs(new node_sz[unlabSize + 1]);

	std::fill(temp_cumulDegs.get(), temp_cumulDegs.get() + str->nNodes + 1, 0);

	for (uint32_t i = 0; i < unlabSize; i++) {
		uint32_t nodeInFullGraph = unlabelled[i];
		uint32_t nodeInFullGraphDeg = fullGraphStruct->deg(nodeInFullGraph);
		uint32_t neighIdxInFullGraphStruct = fullGraphStruct->cumulDegs[nodeInFullGraph];

		// Nuova valutazione dei gradi del grafo redux
		uint32_t tempDeg = 0;
		for (uint32_t j = 0; j < nodeInFullGraphDeg; j++) {
			// se label del vicino � == 0...
			if (!labels[fullGraphStruct->neighs[neighIdxInFullGraphStruct + j]])
				tempDeg++;
		}
		temp_cumulDegs[i + 1] += (tempDeg + temp_cumulDegs[i]);
	}

	// Ora posso allocare le restanti strutture del grafo ridotto
	str->nEdges = temp_cumulDegs[str->nNodes];

	setMemGPU(str->nEdges, GPUINIT_EDGES);
	setMemGPU(str->nEdges, GPUINIT_EDGEW);
	setMemGPU(str->nNodes, GPUINIT_NODET);
	std::unique_ptr<node[]>  temp_neighs(new node[str->nEdges]);
	std::unique_ptr<edgeW[]> temp_edgeWeights(new edgeW[str->nEdges]);
	std::unique_ptr<nodeW[]> temp_nodeThresholds(new nodeW[str->nNodes]);


	// Altro ciclo per riempire la lista dei vicini e dei pesi degli archi associati
	for (uint32_t i = 0; i < unlabSize; i++) {
		uint32_t nodeInFullGraph = unlabelled[i];
		uint32_t nodeInFullGraphDeg = fullGraphStruct->deg(nodeInFullGraph);
		uint32_t neighIdxInFullGraphStruct = fullGraphStruct->cumulDegs[nodeInFullGraph];

		uint32_t tempNeighIdx = temp_cumulDegs[i];
		for (uint32_t j = 0; j < nodeInFullGraphDeg; j++) {
			uint32_t neighInFullGraph = fullGraphStruct->neighs[neighIdxInFullGraphStruct + j];
			if (!labels[neighInFullGraph]) {
				temp_neighs[tempNeighIdx] = f2R[neighInFullGraph];
				temp_edgeWeights[tempNeighIdx] = fullGraphStruct->edgeWeights[neighIdxInFullGraphStruct + j];
				tempNeighIdx++;
			}
		}

		// Ora, la soglia presa dal vettore locale ad ogni thread
		// Occhio, thresholds è già rimappato full-->>redux
		temp_nodeThresholds[i] = thresholds[i];
	}

	// Copio su GPU
	cudaMemcpy(str->cumulDegs, temp_cumulDegs.get(), (str->nNodes + 1) * sizeof(node_sz), cudaMemcpyHostToDevice);
	cudaMemcpy(str->neighs, temp_neighs.get(), str->nEdges * sizeof(node), cudaMemcpyHostToDevice);
	cudaMemcpy(str->edgeWeights, temp_edgeWeights.get(), str->nEdges * sizeof(edgeW), cudaMemcpyHostToDevice);
	cudaMemcpy(str->nodeThresholds, temp_nodeThresholds.get(), str->nNodes * sizeof(nodeW), cudaMemcpyHostToDevice);

	// Devo fare altro? Altrimenti...
	return;
}

template<typename nodeW, typename edgeW>
Graph<nodeW, edgeW>::Graph(Graph<nodeW, edgeW> * const graph_h) {
	GraphStruct<nodeW, edgeW> * const graphStruct_h = graph_h->getStruct();
	setMemGPU(graphStruct_h->nNodes, GPUINIT_NODES);
	str->nNodes = graphStruct_h->nNodes;
	str->nEdges = graphStruct_h->nEdges;
	str->nCleanEdges = graphStruct_h->nCleanEdges;
	setMemGPU(graphStruct_h->nNodes, GPUINIT_EDGES);
	cudaMemcpy(str->cumulDegs, graphStruct_h->cumulDegs, (str->nNodes + 1) * sizeof(node_sz), cudaMemcpyHostToDevice);
	cudaMemcpy(str->neighs, graphStruct_h->neighs, str->nEdges * sizeof(node), cudaMemcpyHostToDevice);
	setMemGPU(graphStruct_h->nNodes, GPUINIT_CEDGES);
	cudaMemcpy(str->edges, graphStruct_h->edges, str->nCleanEdges * 2 * sizeof(node_sz), cudaMemcpyHostToDevice);
	maxDeg = graph_h->maxDeg;
	minDeg = graph_h->minDeg;
	meanDeg = graph_h->meanDeg;
	density = graph_h->density;
	connected = graph_h->connected;
	GPUEnabled = true;
}



/**
 * Invoke the kernel to print the graph on device
 * @param verbose print details
 */
template<typename nodeW, typename edgeW> void Graph<nodeW, edgeW>::print_d(bool verbose) {
	Graph_k::print_d << <1, 1 >> > (str, verbose);
	cudaDeviceSynchronize();
}

/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
template<typename nodeW, typename edgeW>
__global__ void Graph_k::print_d(GraphStruct<nodeW, edgeW>* str, bool verbose) {
	printf("** Graph (num node: %d, num edges: %d)\n", str->nNodes, str->nEdges);

	if (verbose) {
		for (int i = 0; i < str->nNodes; i++) {
			printf("  node(%d)[%d]-> ", i, str->cumulDegs[i + 1] - str->cumulDegs[i]);
			for (int j = 0; j < str->cumulDegs[i + 1] - str->cumulDegs[i]; j++) {
				printf("%d ", str->neighs[str->cumulDegs[i] + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}


template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::deleteMemGPU() {
	if (str->neighs != nullptr) {
		cudaFree(str->neighs);
		str->neighs = nullptr;
	}
	if (str->cumulDegs != nullptr) {
		cudaFree(str->cumulDegs);
		str->cumulDegs = nullptr;
	}
	if (str->edges != nullptr) {
		cudaFree(str->edges);
		str->edges = nullptr;
	}
	if (str->nodeWeights != nullptr) {
		cudaFree(str->nodeWeights);
		str->nodeWeights = nullptr;
	}
	if (str->edgeWeights != nullptr) {
		cudaFree(str->edgeWeights);
		str->edgeWeights = nullptr;
	}
	if (str->nodeThresholds != nullptr) {
		cudaFree(str->nodeThresholds);
		str->nodeThresholds = nullptr;
	}
	if (str != nullptr) {
		delete str;
		str = nullptr;
	}
}

// This sucks... we need to fix template declarations
#ifdef WIN32
	template class Graph<float, float>;
#endif

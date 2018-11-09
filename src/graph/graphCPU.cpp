// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "graph.h"

using namespace std;

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setup(node nn) {
	if (GPUEnabled)
		setMemGPU(nn, GPUINIT_NODES);
	else {
		str = new GraphStruct<nodeW, edgeW>;
		str->cumulDegs = new node_sz[nn + 1]{};
	}
	str->nNodes = nn;
}

template<typename nodeW, typename edgeW>
Graph<nodeW, edgeW>::Graph( fileImporter * imp, bool GPUEnb ) : GPUEnabled{ GPUEnb }, fImport{ imp } {
	if (!GPUEnabled)
		setupImporter();
	else
		setupImporterGPU();
}


template<typename nodeW, typename edgeW>
Graph<nodeW, edgeW>::Graph( const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
		GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F,
		const float * const thresholds, bool GPUEnb ) : GPUEnabled{ GPUEnb } {
	if (!GPUEnabled)
		setupRedux( unlabelled, unlabSize, labels, fullGraphStruct, f2R, r2F, thresholds );
	else
		setupReduxGPU( unlabelled, unlabSize, labels, fullGraphStruct, f2R, r2F, thresholds );
}


template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupImporter() {

	uint32_t nn = fImport->nNodes;
	str = new GraphStruct<nodeW, edgeW>;
	str->cumulDegs = new node_sz[nn + 1];
	str->nNodes = nn;

#ifdef VERBOSEGRAPH
	std::cout << "Creazione liste temporanee..." << std::endl;
#endif
	std::list<uint32_t>	** tempN = new std::list<uint32_t>*[nn];
	std::list<edgeW>	** tempW = new std::list<edgeW>*[nn];
	for (uint32_t i = 0; i < nn; i++) {
		tempN[i] = new std::list<uint32_t>;
		tempW[i] = new std::list<edgeW>;
	}

	fImport->fRewind();
	while (fImport->getNextEdge()) {
		if (fImport->edgeIsValid) {
			tempN[fImport->srcIdx]->push_back( fImport->dstIdx );
			tempW[fImport->srcIdx]->push_back( (edgeW)fImport->edgeWgh );
			str->nEdges++;
			// anche l'arco di ritorno!
			tempN[fImport->dstIdx]->push_back( fImport->srcIdx );
			tempW[fImport->dstIdx]->push_back( (edgeW)fImport->edgeWgh );
			str->nEdges++;
		}
	}

	// Ora in tempN e tempW ho tutto quello che serve per costruire il grafo
	// Inizio con i cumulDegs
	std::fill( str->cumulDegs, str->cumulDegs + (nn + 1), 0);
	for (uint32_t i = 1; i < (nn + 1); i++)
		str->cumulDegs[i] += ( str->cumulDegs[i - 1] + (node_sz)(tempN[i - 1]->size()) );

	str->neighs = new node[str->nEdges];
	str->edgeWeights = new edgeW[str->nEdges];
	str->nodeThresholds = new nodeW[str->nNodes];

	for (uint32_t i = 0; i < nn; i++) {
		uint32_t j = 0;
		for (auto it = tempN[i]->begin(); it != tempN[i]->end(); ++it) {
			str->neighs[str->cumulDegs[i] + j] = *it;
			j++;
		}
		j = 0;
		for (auto it = tempW[i]->begin(); it != tempW[i]->end(); ++it) {
			str->edgeWeights[str->cumulDegs[i] + j] = *it;
			j++;
		}
	}

	// max, min, mean deg
	maxDeg = 0;
	minDeg = nn;
	for (uint32_t i = 0; i < nn; i++) {
		if (str->deg( i ) > maxDeg)
			maxDeg = (uint32_t)str->deg( i );
		if (str->deg( i ) < minDeg)
			minDeg = (uint32_t)str->deg( i );
	}
	density = (float) str->nEdges / (float) (nn * (nn - 1) / 2);
	meanDeg = (float) str->nEdges / (float) nn;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// elimino le strutture temporanee
	for (uint32_t i = 0; i < nn; i++) {
		delete tempW[i];
		delete tempN[i];
	}
	delete[] tempW;
	delete[] tempN;
}


// Questo setup è su con lo sputo. E' un miracolo se funziona.
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupRedux( const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
	GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F, const float * const thresholds ) {

	str = new GraphStruct<nodeW, edgeW>;
	str->nNodes = unlabSize;
	str->nEdges = 0;
	str->cumulDegs = new node_sz[str->nNodes + 1];


	std::fill( str->cumulDegs, str->cumulDegs + str->nNodes + 1, 0 );

	for (uint32_t i = 0; i < unlabSize; i++) {
		uint32_t nodeInFullGraph = unlabelled[i];
		uint32_t nodeInFullGraphDeg = fullGraphStruct->deg( nodeInFullGraph );
		uint32_t neighIdxInFullGraphStruct = fullGraphStruct->cumulDegs[nodeInFullGraph];

		// Nuova valutazione dei gradi del grafo redux
		uint32_t tempDeg = 0;
		for (uint32_t j = 0; j < nodeInFullGraphDeg; j++) {
			// se label del vicino � == 0...
			if (!labels[fullGraphStruct->neighs[neighIdxInFullGraphStruct + j]])
				tempDeg++;
		}
		str->cumulDegs[i + 1] += (tempDeg + str->cumulDegs[i]);
	}

	// Ora posso allocare le restanti strutture del grafo ridotto
	str->nEdges = str->cumulDegs[str->nNodes];
	str->neighs = new node[str->nEdges];
	str->edgeWeights = new edgeW[str->nEdges];
	str->nodeThresholds = new nodeW[str->nNodes];

	// Altro ciclo per riempire la lista dei vicini e dei pesi degli archi associati
	for (uint32_t i = 0; i < unlabSize; i++) {
		uint32_t nodeInFullGraph = unlabelled[i];
		uint32_t nodeInFullGraphDeg = fullGraphStruct->deg( nodeInFullGraph );
		uint32_t neighIdxInFullGraphStruct = fullGraphStruct->cumulDegs[nodeInFullGraph];

		uint32_t tempNeighIdx = str->cumulDegs[i];
		for (uint32_t j = 0; j < nodeInFullGraphDeg; j++) {
			uint32_t neighInFullGraph = fullGraphStruct->neighs[neighIdxInFullGraphStruct + j];
			if (!labels[neighInFullGraph]) {
				str->neighs[tempNeighIdx] = f2R[neighInFullGraph];
				str->edgeWeights[tempNeighIdx] = fullGraphStruct->edgeWeights[neighIdxInFullGraphStruct + j];
				tempNeighIdx++;
			}
		}

		// Ora, la soglia presa dal vettore locale ad ogni thread
		// Occhio, thresholds è già rimappato full-->>redux
		str->nodeThresholds[i] = thresholds[i];
	}

	// Devo fare altro? Altrimenti...
	return;
}


/**
 * Generate a new random graph
 * @param eng seed
 */
 // TODO: adeguare alla rimozione della Unified Memory
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::randGraphUnweighted(float prob, std::default_random_engine & eng) {
	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n");
	}
	uniform_real_distribution<> randR(0.0, 1.0);
	node n = str->nNodes;

	// gen edges
	vector<int>* edges = new vector<int>[n];
	for (uint32_t i = 0; i < n - 1; i++) {
		for (uint32_t j = i + 1; j < n; j++)
			if (randR(eng) < prob) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				str->cumulDegs[i + 1]++;
				str->cumulDegs[j + 1]++;
				str->nEdges += 2;
			}
	}
	str->cumulDegs[0] = 0;
	for (uint32_t i = 0; i < n; i++)
		str->cumulDegs[i + 1] += str->cumulDegs[i];

	// max, min, mean deg
	maxDeg = 0;
	minDeg = n;
	for (uint32_t i = 0; i < n; i++) {
		if (str->deg(i) > maxDeg)
			maxDeg = (uint32_t) str->deg(i);
		if (str->deg(i) < minDeg)
			minDeg = (uint32_t) str->deg(i);
	}
	density = (float) str->nEdges / (float) (n * (n - 1)/2);
	meanDeg = (float) str->nEdges / (float) n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// manage memory for edges with CUDA Unified Memory
	if (GPUEnabled)
		setMemGPU(str->nEdges, GPUINIT_EDGES);
	else
		str->neighs = new node[str->nEdges] {};

	for (uint32_t i = 0; i < n; i++)
		memcpy((str->neighs + str->cumulDegs[i]), edges[i].data(), sizeof(int) * edges[i].size());

	// free resources
	delete[] edges;
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::print(bool verbose) {
	node n = str->nNodes;
	cout << "\n** Graph " << endl;
	cout << "   - num node: " << n << endl;
	cout << "   - num edges: " << str->nEdges << " (density: " << density << ")" << endl;
	cout << "   - min deg: " << minDeg << endl;
	cout << "   - max deg: " << maxDeg << endl;
	cout << "   - mean deg: " << meanDeg << endl;
	cout << "   - connected: " << connected << endl;
	if (verbose) {
		for (uint32_t i = 0; i < n; i++) {
			cout << "   node(" << i << ")" << "[" << str->cumulDegs[i + 1] - str->cumulDegs[i] << "]-> ";
			for (uint32_t j = 0; j < str->cumulDegs[i + 1] - str->cumulDegs[i]; j++) {
				cout << str->neighs[str->cumulDegs[i] + j] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}

template class Graph<float, float>;
//template class Graph<uint32_t, uint32_t>;

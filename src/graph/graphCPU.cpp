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
Graph<nodeW, edgeW>::Graph(fileImporter * imp, bool GPUEnb) : GPUEnabled{ GPUEnb }, fImport{ imp } {
	if (!GPUEnabled)
		setupImporterNew();
	else
		setupImporterGPU();
	// Eh, fix per impostare la probabilita' nel caso in cui il grafo venga letto da file
	prob = str->nEdges / (float) (str->nNodes * str->nNodes);
}

template<typename nodeW, typename edgeW>
Graph<nodeW, edgeW>::Graph( node nn, float prob, uint32_t seed ) : GPUEnabled{ false }, prob{ prob } {
	setupRnd2( nn, prob, seed );
}

template<typename nodeW, typename edgeW>
Graph<nodeW, edgeW>::Graph(const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
	GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F,
	const float * const thresholds, bool GPUEnb) : GPUEnabled{ GPUEnb } {
	if (!GPUEnabled)
		setupRedux(unlabelled, unlabSize, labels, fullGraphStruct, f2R, r2F, thresholds);
	else
		setupReduxGPU(unlabelled, unlabSize, labels, fullGraphStruct, f2R, r2F, thresholds);
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
			// Remove self loops
			if (fImport->srcIdx != fImport->dstIdx) {
				tempN[fImport->srcIdx]->push_back(fImport->dstIdx);
				tempW[fImport->srcIdx]->push_back((edgeW)fImport->edgeWgh);
				str->nEdges++;
				// anche l'arco di ritorno!
				tempN[fImport->dstIdx]->push_back(fImport->srcIdx);
				tempW[fImport->dstIdx]->push_back((edgeW)fImport->edgeWgh);
				str->nEdges++;
			}
		}
	}

	// Ora in tempN e tempW ho tutto quello che serve per costruire il grafo
	// Inizio con i cumulDegs
	std::fill(str->cumulDegs, str->cumulDegs + (nn + 1), 0);
	for (uint32_t i = 1; i < (nn + 1); i++)
		str->cumulDegs[i] += (str->cumulDegs[i - 1] + (node_sz)(tempN[i - 1]->size()));

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

	doStats();

	// elimino le strutture temporanee
	for (uint32_t i = 0; i < nn; i++) {
		delete tempW[i];
		delete tempN[i];
	}
	delete[] tempW;
	delete[] tempN;
}

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupImporterNew() {
	uint32_t nn = fImport->nNodes;
	str = new GraphStruct<nodeW, edgeW>;
	str->cumulDegs = new node_sz[nn + 1];
	std::fill(str->cumulDegs, str->cumulDegs + (nn + 1), 0);
	str->nNodes = nn;

	// Filling cumulDegs
	std::cout << "Filling cumulDegs..." << std::endl;
	fImport->fRewind();
	while (fImport->getNextEdge()) {
		if (fImport->edgeIsValid) {
			// Remove self loops
			if (fImport->srcIdx != fImport->dstIdx) {
				str->cumulDegs[fImport->srcIdx + 1]++;
				str->nEdges++;
				// anche l'arco di ritorno!
				str->cumulDegs[fImport->dstIdx + 1]++;
				str->nEdges++;
			}
		}
	}

	// Cumulating cumulDegs
	std::cout << "Cumulating cumulDegs..." << std::endl;
	for (uint32_t i = 1; i < (nn + 1); i++)
		str->cumulDegs[i] += str->cumulDegs[i - 1];

	// Filling neighs and weights reading il file again
	str->neighs = new node[str->nEdges];
	str->edgeWeights = new edgeW[str->nEdges];
	str->nodeThresholds = new nodeW[str->nNodes];

	std::vector<size_t> tempDegs( nn, 0 );
	size_t neighIdx;

	std::cout << "Filling neighs and weights..." << std::endl;
	fImport->fRewind();
	while (fImport->getNextEdge()) {
		if (fImport->edgeIsValid) {
			// Remove self loops
			if (fImport->srcIdx != fImport->dstIdx) {
				neighIdx = str->cumulDegs[fImport->srcIdx] + tempDegs[fImport->srcIdx];
				str->neighs[neighIdx] = fImport->dstIdx;
				str->edgeWeights[neighIdx] = fImport->edgeWgh;
				tempDegs[fImport->srcIdx]++;
				// anche l'arco di ritorno!
				neighIdx = str->cumulDegs[fImport->dstIdx] + tempDegs[fImport->dstIdx];
				str->neighs[neighIdx] = fImport->srcIdx;
				str->edgeWeights[neighIdx] = fImport->edgeWgh;
				tempDegs[fImport->dstIdx]++;
			}
		}
	}

	std::cout << "Calculating statistics..." << std::endl;
	doStats();
}


// Questo setup è su con lo sputo. E' un miracolo se funziona.
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupRedux(const uint32_t * const unlabelled, const uint32_t unlabSize, const int32_t * const labels,
	GraphStruct<nodeW, edgeW> * const fullGraphStruct, const uint32_t * const f2R, const uint32_t * const r2F, const float * const thresholds) {

	str = new GraphStruct<nodeW, edgeW>;
	str->nNodes = unlabSize;
	str->nEdges = 0;
	str->cumulDegs = new node_sz[str->nNodes + 1];


	std::fill(str->cumulDegs, str->cumulDegs + str->nNodes + 1, 0);

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
		uint32_t nodeInFullGraphDeg = fullGraphStruct->deg(nodeInFullGraph);
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

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupRnd( node nn, float prob, uint32_t seed ) {
	str = new GraphStruct<nodeW, edgeW>;
	str->cumulDegs = new node_sz[nn + 1];
	std::fill(str->cumulDegs, str->cumulDegs + (nn + 1), 0);
	str->nNodes = nn;
	str->nEdges = 0;

	if (prob < 0 || prob > 1) {
		std::cout << TXT_BIYLW << "Invalid probability value. Setting default [p = 0.5]" << TXT_NORML << std::endl;
		prob = 0.5f;
	}

	// Mersienne twister RNG
	// Performance comparison of C++ PRNG:
	// https://stackoverflow.com/questions/35358501/what-is-performance-wise-the-best-way-to-generate-random-bools
	std::mt19937 MtRng( seed );
	uint32_t lim = std::numeric_limits<uint32_t>::max() * prob;

	vector<uint32_t>* edges = new vector<uint32_t>[nn];
	// filling edge lists
	for (uint32_t i = 0; i < nn - 1; i++) {
		for (uint32_t j = i + 1; j < nn; j++) {
			if (MtRng() < lim) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				str->cumulDegs[i + 1]++;
				str->cumulDegs[j + 1]++;
				str->nEdges += 2;
			}
		}
	}

	std::cout << "Cumulating cumulDegs..." << std::endl;
	for (uint32_t i = 1; i < (nn + 1); i++)
		str->cumulDegs[i] += str->cumulDegs[i - 1];

	str->neighs = new node[str->nEdges];
	str->edgeWeights = new edgeW[str->nEdges];
	str->nodeThresholds = new nodeW[str->nNodes];

	std::cout << "Copying neighs..." << std::endl;
	for (size_t i = 0; i < nn; i++)
		memcpy((str->neighs + str->cumulDegs[i]), edges[i].data(), sizeof(int) * edges[i].size());

	std::cout << "Generating weights..." << std::endl;
	for (size_t i = 0; i < str->nEdges; i++)
		str->edgeWeights[i] = (float) MtRng() / std::numeric_limits<uint32_t>::max();

	std::cout << "Calculating statistics..." << std::endl;
	doStats();
#ifdef CHECKRANDGRAPH
	checkRandGraph();
#endif

	delete[] edges;
}

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupRnd2( node n, float prob, uint32_t seed ) {

	Timer rndTime;
	size_t nn = static_cast<size_t>( n );
	size_t vecSize = nn * (nn + 1) / 2;
	std::vector<bool> boolGraph( vecSize );
	// Filing the progress bar values
	std::vector<size_t> gaugeVals( 40 );
	size_t idx = 1;
	size_t aaa = vecSize / 40;
	std::for_each(std::begin(gaugeVals), std::end(gaugeVals), [&](size_t &val) {val = aaa * idx++ - 1;} );
	idx = 0;

	std::cout << "Generating... " << std::endl;
	std::cout << "|----------------------------------------|" << std::endl << "\033[F|";
	rndTime.startTime();
	for (size_t i = 0; i < vecSize; i++) {
		boolGraph[i] = ((double)rand() / (RAND_MAX)) >= prob ? 0 : 1;
		if (i == gaugeVals[idx]) {
		 	std::cout << "#" << std::flush;
			idx++;
		}
	}
	rndTime.endTime();
	std::cout << std::endl << rndTime.duration() / 1000 << " s" << std::endl;

	size_t a = 0;
	std::for_each( std::begin(boolGraph), std::end(boolGraph), [&](bool val) {a += val;} );
	std::cout << "Vecsize: " << vecSize << " - tot pos: " << a << std::endl;

	str = new GraphStruct<nodeW, edgeW>;
	str->cumulDegs = new node_sz[nn + 1];
	std::fill(str->cumulDegs, str->cumulDegs + (nn + 1), 0);
	str->nNodes = nn;
	str->nEdges = 0;

	size_t i = 0;	// col
	size_t j = 0;	// row
	idx = 0;		// progress bar
	std::cout << "Calculating degs... " << std::endl;
	std::cout << "|----------------------------------------|" << std::endl << "\033[F|";
	rndTime.startTime();
	for (size_t k = 0; k < vecSize; k++) {
		if (j == i)
			boolGraph[k] = 0;

		if (boolGraph[k]) {
			str->cumulDegs[i + 1]++;
			str->cumulDegs[j + 1]++;
			str->nEdges += 2;
		}

		i++;
		if (i == nn) {
			j++;
			i = j;
		}

		if (k == gaugeVals[idx]) {
		 	std::cout << "#" << std::flush;
			idx++;
		}
	}
	rndTime.endTime();
	std::cout << std::endl << rndTime.duration() / 1000 << " s" << std::endl;

	std::cout << "Cumulating cumulDegs... " << std::flush;
	rndTime.startTime();
	for (uint32_t i = 1; i < (nn + 1); i++)
		str->cumulDegs[i] += str->cumulDegs[i - 1];
	rndTime.endTime();
	std::cout << std::endl << rndTime.duration() / 1000 << " s" << std::endl;

	str->neighs = new node[str->nEdges];
	str->edgeWeights = new edgeW[str->nEdges];
	str->nodeThresholds = new nodeW[str->nNodes];

	std::vector<size_t> tempDegs( nn, 0 );
	size_t neighIdx;
	idx = i = j = 0;
	std::cout << "Filling neighs and weights... " << std::endl;
	std::cout << "|----------------------------------------|" << std::endl << "\033[F|";
	rndTime.startTime();
	for (size_t k = 0; k < vecSize; k++) {
		if (boolGraph[k]) {
			neighIdx = str->cumulDegs[j] + tempDegs[j];
			str->neighs[neighIdx] = i;
			//str->edgeWeights[neighIdx] = fImport->edgeWgh;
			tempDegs[j]++;
			// anche l'arco di ritorno!
			neighIdx = str->cumulDegs[i] + tempDegs[i];
			str->neighs[neighIdx] = j;
			//str->edgeWeights[neighIdx] = fImport->edgeWgh;
			tempDegs[i]++;
		}
		i++;
		if (i == nn) {
			j++;
			i = j;
		}

		if (k == gaugeVals[idx]) {
		 	std::cout << "#" << std::flush;
			idx++;
		}
	}
	rndTime.endTime();
	std::cout << std::endl << rndTime.duration() / 1000 << " s" << std::endl;

	doStats();
#ifdef CHECKRANDGRAPH
	checkRandGraph();
#endif
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

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::doStats() {
	// max, min, mean deg
	size_t nn = str->nNodes;
	maxDeg = 0;
	minDeg = nn;
	for (uint32_t i = 0; i < nn; i++) {
		if (str->deg(i) > maxDeg)
			maxDeg = (uint32_t)str->deg(i);
		if (str->deg(i) < minDeg)
			minDeg = (uint32_t)str->deg(i);
	}
	density = (float)str->nEdges / (float)(nn * (nn - 1) / 2);
	meanDeg = (float)str->nEdges / (float)nn;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;
}

template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::checkRandGraph() {
	size_t nn = str->nNodes;
	std::vector<size_t> gaugeVals( 40 );
	size_t idx = 1;
	size_t aaa = nn / 40;
	std::for_each(std::begin(gaugeVals), std::end(gaugeVals), [&](size_t &val) {val = aaa * idx++ - 1;} );
	idx = 0;

	std::cout << "Checking for duplicate edges in neighbor lists..." << std::endl;
	std::cout << "|----------------------------------------|" << std::endl << "\033[F|";
	for (size_t i = 0; i < nn; i++) {
		size_t nodeDeg = str->cumulDegs[i + 1] - str->cumulDegs[i];
		auto neighIdx = str->neighs + str->cumulDegs[i];
		auto aa = std::unique(neighIdx, neighIdx + nodeDeg);
		if (aa != (str->neighs + str->cumulDegs[i + 1])) {
			std::cout << "Aborting: duplicate in edge list of node " << i << std::endl;
			exit( -1 );
		}
		if (i == gaugeVals[idx]) {
		 	std::cout << "#" << std::flush;
			idx++;
		}
	}

	idx = 0;
	std::cout << std::endl << "Checking for edge bidirectionality in neighbor lists..." << std::endl;
	std::cout << "|----------------------------------------|" << std::endl << "\033[F|";
	for (size_t i = 0; i < nn; i++) {
		size_t nodeDeg = str->cumulDegs[i + 1] - str->cumulDegs[i];
		auto neighIdx = str->neighs + str->cumulDegs[i];
		for (size_t j = 0; j < nodeDeg; j++) {
			auto neighNode = neighIdx[j];
			auto neighNodeDeg = str->cumulDegs[neighNode + 1] - str->cumulDegs[neighNode];
			auto neighNodeIdx = str->neighs + str->cumulDegs[neighNode];
			// look for i in the neigh list of neighNode. There should be none;
			auto aa = std::find(neighNodeIdx, neighNodeIdx + neighNodeDeg, i);
			if (aa == neighNodeIdx + neighNodeDeg) {
				std::cout << "Aborting: bidirectional edge between nodes " << i << " and " << neighNode << " not found" << std::endl;
				std::cout << i << " neighs list: ";
				std::for_each(neighIdx, neighIdx + nodeDeg, [&](uint32_t val) {std::cout << val << " "; });
				std::cout << std::endl << neighNode << " neighs list: ";
				std::for_each(neighNodeIdx, neighNodeIdx + neighNodeDeg, [&](uint32_t val) {std::cout << val << " "; });
				exit( -1 );
			}
		}
		if (i == gaugeVals[idx]) {
		 	std::cout << "#" << std::flush;
			idx++;
		}
	}
	std::cout << std::endl;
}

// This sucks... we need to fix template declarations
#ifdef WIN32
template class Graph<float, float>;
#endif
//template class Graph<uint32_t, uint32_t>;

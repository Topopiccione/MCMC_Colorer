/*
Costruttori e funzioni necessarie al funzionamento della classe graph:
- costruttore con input numero di nodi: inizializza grafo vuoto [ci serve?]
- costruttore con input fileImporter: costruisce grafo da file [necessario]
- costruttore grafo casuale [anche questo, forse non necessario x' esiste datasetGen che fa la stessa cosa]
- costruttore grafo ridotto da lista di nodi con etichetta positiva [necessario per Hopfield]

Molte delle versioni devono allocare dati sulla memoria host, device o entrambe.
Trovare schema di allocazione intelligente che non stravolga i colorer e Hopfield
Importante mantenere separati i contesti
*/

// Import a graph from file
// The strategy here is to build temporary structures for data processing and move them
// into the appropriate structures if host allocation is requested, otherwise copy them on GPU memory
// and deallocate the host counterpart.
// str is a pointer to a GraphStruct that gets always allocated on host side.
// If not GPUenabled: structures inside str get allocated on host memory and
// filled with the relevant data. if GPUenabled, structures are allocated on device
// memory, and all the temporary arrays allocated on the host are removed.
template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::setupImporter( bool isBidirectional, bool GPUenabled ) {

	uint32_t nn = fImport->nNodes;
	// If working on the GPU, str gets allocated there, otherwise str stays on host ram
	if (GPUenabled)
		setMemGPU(nn, GPUINIT_NODES);
	else
		str = new GraphStruct<nodeW, edgeW>;

	str->nNodes = nn;
	node_sz * temp_cumulDegs = new[nn + 1];

	// Creating temporary lists
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
			// Add the back edge only if not explicitly present in the graph file
			if (!isBidirectional) {
				tempN[fImport->dstIdx]->push_back( fImport->srcIdx );
				tempW[fImport->dstIdx]->push_back( (edgeW)fImport->edgeWgh );
				str->nEdges++;
			}
		}
	}

	// TempN and TempW contain the whole graph; now assembling the compressed structures
	// starting with the cumulative degrees array
	std::fill( temp_cumulDegs, temp_cumulDegs + nn + 1, 0);
	for (uint32_t i = 1; i < (nn + 1); i++)
		temp_cumulDegs[i] += ( temp_cumulDegs[i - 1] + (uint32_t)(tempN[i - 1]->size()) );

	// Allocation of relevant structures in GPU memory, if needed
	if (GPUenabled) {
		setMemGPU( str->nEdges, GPUINIT_EDGES );
		setMemGPU( str->nEdges, GPUINIT_EDGEW );
		setMemGPU( nn, GPUINIT_NODET );
	}
	// And temporary arrays on host side
	node  * temp_neighs = new node[str->nEdges];
	edgeW * temp_edgeWeights = new edgeW[str->nEdges];
	nodeW * temp_nodeThresholds = new nodeW[str->nNodes];

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

	// max, min, mean deg
	maxDeg = 0;
	minDeg = nn;
	for (uint32_t i = 0; i < nn; i++) {
		if ((temp_cumulDegs[i + 1] - temp_cumulDegs[i]) > maxDeg)
			maxDeg = (uint32_t)str->deg( i );
		if ((temp_cumulDegs[i + 1] - temp_cumulDegs[i]) < minDeg)
			minDeg = (uint32_t)str->deg( i );
	}
	density = (float) str->nEdges / (float) (nn * (nn - 1) / 2);
	meanDeg = (float) str->nEdges / (float) nn;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	if (GPUenabled) {
		cudaMemcpy( str->cumulDegs,      temp_cumulDegs.get(),      (str->nNodes + 1) * sizeof(node_sz),	cudaMemcpyHostToDevice );
		cudaMemcpy( str->neighs,         temp_neighs.get(),         str->nEdges * sizeof(node),				cudaMemcpyHostToDevice );
		cudaMemcpy( str->edgeWeights,    temp_edgeWeights.get(),    str->nEdges * sizeof(edgeW),			cudaMemcpyHostToDevice );
		cudaMemcpy( str->nodeThresholds, temp_nodeThresholds.get(), str->nNodes * sizeof(nodeW),			cudaMemcpyHostToDevice );
		// No longer needed on host memory
		delete[] temp_nodeThresholds;
		delete[] temp_edgeWeights;
		delete[] temp_neighs;
		delete[] temp_cumulDegs;

	} else {
		str->cumulDegs		= temp_cumulDegs;
		str->neighs			= temp_neighs;
		str->edgeWeights	= temp_edgeWeights;
		str->nodeThresholds	= temp_nodeThresholds;
	}

	// Removing the temporary structures
	for (uint32_t i = 0; i < nn; i++) {
		delete tempW[i];
		delete tempN[i];
	}
	delete[] tempW;
	delete[] tempN;
}

// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

__constant__ float LAMBDA;
__constant__ col_sz NCOLS;
__constant__ float EPSILON;
__constant__ float CUMULSUMDIST;
__constant__ float RATIOFREEZED;


template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW,edgeW>::ColoringMCMC( Graph<nodeW,edgeW> * inGraph_d, curandState * randStates, ColoringMCMCParams params ) :
	Colorer<nodeW,edgeW>( inGraph_d ),
	graphStruct_d( inGraph_d->getStruct() ),
	nnodes( inGraph_d->getStruct()->nNodes ),
	randStates( randStates ),
	numOfColors( 0 ),
	threadId( 0 ),
	param( params ) {

	coloring_h = std::unique_ptr<int[]>( new int[nnodes] );

	// pointer per il Coloring in output
	outColoring_d = std::unique_ptr<Coloring>( new Coloring );

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// configuro la griglia e i blocchi
	numThreads = 32;
	threadsPerBlock = dim3( numThreads, 1, 1 );
	blocksPerGrid = dim3( (nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );


	// init curand
	// TODO: abbiamo già una classe che si occupa di gestire curand, megio usare quella! [cfr. lubyGPU]
	cuSts = cudaMalloc((void **) &states, nnodes * sizeof(curandState));
	cudaCheck(cuSts, __FILE__, __LINE__);
	long seed = 0;
	GPURand_k::initCurand<<<blocksPerGrid, threadsPerBlock>>>(states, seed, nnodes);
	cudaDeviceSynchronize();

	// coloring
	// TODO: adeguare le strutture usate nella colorazione in modo che siano come in lubyGPU
	// TODO: no malloc, sì new
	/*col_sz*/ nCol = inGraph_d->getMaxNodeDeg() + 1;
	//col *C, *C1_d, *C2_d;
	C = (col*)malloc(nnodes * sizeof(col));
	cuSts = cudaMalloc(&C1_d, nnodes * sizeof(col)); cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc(&C2_d, nnodes * sizeof(col)); cudaCheck(cuSts, __FILE__, __LINE__);

	// neighborhood coloring
	//col* colNeigh;
	cuSts = cudaMalloc(&colNeigh, graphStruct_d->cumulDegs[nnodes]*sizeof(col));	// Questo non funzionerà: cumulDegs è su device mem
	cudaCheck(cuSts, __FILE__, __LINE__);

	// MCMC priori distribution parameters
	param.nCol = nCol;
	param.lambda = param.lambda/(float)nCol;
	dist.lambda = param.lambda;
	CPURand::createExpDistribution(&dist, dist.lambda, nCol); // form: exp(-lambda*c/nCol), c = 1,2...

}

// TODO: eliminare tutto quello creato nel construttore per evitare memory leak
template<typename nodeW, typename edgeW>
ColoringMCMC<nodeW, edgeW>::~ColoringMCMC() {
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	if (outColoring_d->colClass != nullptr) {
		cudaFree( outColoring_d->colClass );
		outColoring_d->colClass = nullptr;
	}
	if (outColoring_d->colClass != nullptr) {
		cudaFree( outColoring_d->cumulSize );
		outColoring_d->cumulSize = nullptr;
	}
}

template<typename nodeW, typename edgeW>
Coloring* ColoringMCMC<nodeW,edgeW>::getColoringGPU() {
	return outColoring_d.get();
}

template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::printgraph() {
	ColoringMCMC_k::print_graph_k<nodeW, edgeW> <<< 1, 1 >>> (nnodes, graphStruct_d->cumulDegs, graphStruct_d->neighs);
}


template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::convert_to_standard_notation(){
	uint32_t idx;

	uint32_t *	colClass =  new uint32_t[nnodes] ;
	uint32_t *	cumulSize = new uint32_t[numOfColors+1] ;

	idx=0;
	memset( colClass, 0, nnodes*sizeof(uint32_t) );
	memset( cumulSize, 0, (numOfColors+1)*sizeof(uint32_t) );

	// Ciclo sui colori
	for(uint32_t c=0; c<numOfColors; c++){
		// NB: i colori in luby vanno da 1 a numOfColors

		// Ciclo sui nodi
		for(uint32_t i=0; i<nnodes; i++){
			if(coloring_h[i]==(c+1)){
				colClass[idx]=i;
				idx++;
			}
		}

		cumulSize[c+1]=idx;
	}
	/*
	for (uint32_t i = 0; i < nnodes; i++)
		std::cout << coloring_h[i] << " ";
	std::cout << std::endl;

	for (uint32_t i = 0; i < numOfColors + 1; i++)
		std::cout << cumulSize[i] << " ";
	std::cout << std::endl;

	for (uint32_t i = 0; i < numOfColors; i++) {
		uint32_t ISoffs = cumulSize[i];
		uint32_t ISsize = cumulSize[i + 1] - cumulSize[i];
		std::cout << "colore " << i + 1 << ": ";
		for (uint32_t j = ISoffs; j < ISoffs + ISsize; j++) {
			std::cout << colClass[j] << " ";
		}
		std::cout << std::endl;
	}
	*/
#ifdef TESTCOLORINGCORRECTNESS
	std::cout << "Test colorazione attivato!" << std::endl;
	std::unique_ptr<node_sz[]> temp_cumulDegs( new node_sz[graphStruct_d->nNodes + 1]);
	std::unique_ptr<node[]>  temp_neighs( new node[graphStruct_d->nEdges] );
	cuSts = cudaMemcpy( temp_cumulDegs.get(), graphStruct_d->cumulDegs, (graphStruct_d->nNodes + 1) * sizeof(node_sz), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( temp_neighs.get(),    graphStruct_d->neighs,    graphStruct_d->nEdges * sizeof(node_sz), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );

	for (uint32_t i = 0; i < numOfColors; i++) {
		uint32_t ISsize = cumulSize[i + 1] - cumulSize[i];
		uint32_t ISoffs = cumulSize[i];
		for (uint32_t j = 0; j < ISsize; j++) {
			const uint32_t nodoCorrente = colClass[ISoffs + j];
			const node_sz degNodoCorrn = graphStruct_d->cumulDegs[nodoCorrente + 1] - graphStruct_d->cumulDegs[nodoCorrente];
			const node_sz offNodoCorrn = graphStruct_d->cumulDegs[nodoCorrente];
			for (uint32_t k = 0; k < ISsize; k++) {
				if (std::find( &(graphStruct_d->neighs[offNodoCorrn]), &(graphStruct_d->neighs[offNodoCorrn + degNodoCorrn]), colClass[ISoffs + k] ) !=
					&(graphStruct_d->neighs[offNodoCorrn + degNodoCorrn])) {
					std::cout << "NO! In colore " << i + 1 << ", il nodo " << nodoCorrente << " ha come vicino " << colClass[i + k] << std::endl;
					abort();
				}
			}
		}
	}
#endif

	outColoring_d->nCol = numOfColors;
	cuSts = cudaMalloc( (void**)&(outColoring_d->colClass), nnodes*sizeof(uint32_t) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( outColoring_d->colClass, colClass, nnodes*sizeof(uint32_t), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );

	cuSts = cudaMalloc( (void**)&(outColoring_d->cumulSize), (numOfColors+1)*sizeof(uint32_t) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( outColoring_d->cumulSize, cumulSize, (numOfColors+1)*sizeof(uint32_t), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );


#ifdef PRINT_COLORING
	printf( "\nStampa convertita in formato standard GPU colorer\n" );
	uint32_t temp, size;
	temp=0;
	for (uint32_t i = 0; i < numOfColors; i++) {
		printf( "Colore %d: ", i );
		size=cumulSize[i+1]-cumulSize[i];
		for (uint32_t j = 0; j < size; j++){
			printf( "%d ", colClass[temp] );
			temp++;
		}
		printf( "\n" );
	}
#endif

	delete[] colClass;
	delete[] cumulSize;
}




/**
 * Samples a discrete distribution
 * @param states curand state
 * @param x the element sampled in the range [0,n-1]
 * @param dist the probability mass function (not normalized)
 * @param n the distribution size
 */
__device__ int ColoringMCMC_k::discreteSampling(curandState* states, discreteDistribution_st * dist) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int n = dist->length;
	printf("n = %d\n", n);
	unsigned int l = 0, r = n - 1;
	float u = curand_uniform(&states[tid]);
	float bin = dist->prob[n - 1] * u;
	printf("bin = %f\n", bin);
	while (l < r) {
		unsigned int m = floorf((l + r) / 2);
		if (bin < dist->prob[m])  // RIFARE
			r = m;
		else
			l = m + 1;
	}
	return r;
}

__global__ void ColoringMCMC_k::initColoring(curandState* state,
		expDiscreteDistribution_st * dist, col* C, col_sz nCol,
		node_sz n) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		discreteDistribution_st d = dist->CDF;
		printf("length = %d\n", d.length);
		// C[i] = ColoringMCMC_k::discreteSampling(state, &d);
		printf("init_col[%d] = %d\n", i, C[i]);
	}
}

/**
 * Counts the number of conflicts for a node with respect a coloring
 * @param nodeID the node
 * @param deg the node degree
 * @param neighs neighborhood
 * @param C the current coloring
 * @return the number of conflicts
 */
__inline__ __host__ __device__ node_sz ColoringMCMC_k::checkConflicts(
		node nodeID,
		node_sz deg,
		node* neighs,
		col* C) {

	node_sz nConf = 0;
	col nodeCol = C[nodeID];
	for (node_sz i = 0; i < deg; i++)
		if (C[neighs[i]] == nodeCol)
			nConf++;
	return nConf;
}

/**
 * Assigns a new color to a node drawn among the free colors
 * @param state curandState
 * @param deg
 * @param neigh
 * @param C1
 * @param param
 * @return
 */
__inline__  __device__ col ColoringMCMC_k::newFreeColor(
		curandState *state,
		node nodeID,
		node_sz deg,
		node* neighs,
		col* C,
		col* colNeighs) {   // colors of neighborhood

	// conflict analysis
	col_sz k = NCOLS;
	float lambda = LAMBDA;
	float epsilon = EPSILON;

	// find num unique neigh colors
	colNeighs[0] = C[neighs[0]];
	col_sz nColNotFree = 1;
	for (node j = 1; j < deg; j++) {
		col cc = C[neighs[j]];
		bool FOUND = 0;
		for (col i = 0; i < nColNotFree; i++)
			if (colNeighs[i] == cc) {
				FOUND = 1;
				break;
			}
		if (!FOUND) {
			colNeighs[nColNotFree] = cc;
			nColNotFree++;
		}
	}


	// bubble sort neigh nodes
	for (col c = 0; c < nColNotFree - 1; c++)
		for (col g = c + 1; g < nColNotFree; g++)
			if (colNeighs[c] > colNeighs[g]) {
				col tmp = colNeighs[g];
				colNeighs[g] = colNeighs[c];
				colNeighs[c] = tmp;
			}


	// compute cumulSum probability
	float cumProb = 0;
	col j = 0;
	for (col c = 1; c <= k; c++) {
		if ((j < nColNotFree) && (c == colNeighs[j])) {
			j++;
		}
		else {
			cumProb += expf(-lambda * (float)c);
		}
	}
	col_sz nColFree = k - nColNotFree;
	float a = (1.0f - epsilon)/nColFree;
	cumProb = cumProb*a + epsilon;

	// sample the discrete distribution
	float u = cumProb * curand_uniform(state);
	cumProb = 0;
	col newCol;
	j = 0;

	bool vicinato;
	for (col c = 1; c <= k; c++) {
		if ((j < nColNotFree) && (c == colNeighs[j])) {
			cumProb += epsilon/(float)nColNotFree;
			j++;
			vicinato = 1;
			// if (nodeID == 5)
			// printf("VICINO: Prob[%d][%d] = %f\n", nodeID,c, epsilon/(float)nColNotFree);
		} else {
			cumProb += a * __expf(-lambda * (float)c);
			vicinato = 0;
			// printf("NON VICINO: Prob[%d][%d] = %f\n", nodeID,c, a * __expf(-lambda * (float)c));
		}
			// printf("cumProb = %f -- u = %f -- vicinato = %d \n", cumProb,u,vicinato);

		// choose bin
		if (cumProb >= u) {
			printf("vicinato = %d\n", vicinato);
			newCol = c;
			break;
		}
	}
	return newCol;
}

/**
 * Assigns a new color to a node drawn among the free colors
 * @param state curandState
 * @param deg
 * @param neigh
 * @param C1
 * @param param
 * @return
 */
__inline__  __device__ col ColoringMCMC_k::fixColor(
		curandState *state,
		node* nodeNeighs,
		col nodeCol) {
	float lambda = LAMBDA;
	float epsilon = EPSILON;
	unsigned k = NCOLS;

	// sampling the discrete distribution
	float u = curand_uniform(state);
	col newCol;
	float cumProb = 1 - epsilon;
	if (u < cumProb)
		newCol = nodeCol;
	else {
		float P = __expf(-lambda)*(1.0-__expf(-lambda * (float)k)) / ((1.0-__expf(-lambda))) - __expf(-lambda * (float)nodeCol);
		for (col c = 1; c <= k; c++) {
			if (c != nodeCol)
				cumProb += epsilon * __expf(-lambda * (float)c) / P;
			if (u < cumProb) {
				newCol = c;
				break;
			}
		}
	}
	return newCol;
}

/**
 * Draws a new coloring based on the current coloring
 * @param states curand state
 * @param str graph structure
 * @param C1 current coloring
 * @param C2 new coloring
 * @param colSupport support for random generation
 * @param numConflicts overall conflicts
 * @param PHI support distribution
 */
 template<typename nodeW, typename edgeW>
__global__ void ColoringMCMC_k::drawNewColoring(
		unsigned int* numConflicts_d,
		curandState* states,
		const GraphStruct<nodeW, edgeW> * str,
		col* C1_d,
		col* C2_d,
		col* colSupport) {

	node nodeID = threadIdx.x + blockDim.x * blockIdx.x;
	node_sz n = str->nNodes;
	unsigned int nConf;

	// even & odd
	bool EVEN_ODD = 0;
	if (*numConflicts_d) {
		EVEN_ODD = 1;
		*numConflicts_d = 0;
	}

	if (nodeID < n) {
		node_sz cumulDeg = str->cumulDegs[nodeID];
		node_sz deg = str->cumulDegs[nodeID + 1] - cumulDeg;
		//		printf("deg [%d] = %d\n",nodeID,deg);
		if (EVEN_ODD)
			nConf = ColoringMCMC_k::checkConflicts(nodeID, deg,	str->neighs + cumulDeg, C1_d);
		else
			nConf = ColoringMCMC_k::checkConflicts(nodeID, deg,	str->neighs + cumulDeg, C2_d);
		atomicAdd(numConflicts_d, nConf);
		printf("num conflicts[%d] = %d\n",nodeID,nConf);

		// CASE: conflicts
		if (nConf) {
			if (RATIOFREEZED > curand_uniform(&states[nodeID])) {
				if (EVEN_ODD)
					C2_d[nodeID] = newFreeColor(states + nodeID, nodeID, deg,	str->neighs + cumulDeg, C1_d, colSupport + cumulDeg);
				else
					C1_d[nodeID] = newFreeColor(states + nodeID, nodeID, deg,	str->neighs + cumulDeg, C2_d, colSupport + cumulDeg);
			// printf("new color C2[%d] = %d   -- deg = %d  -- cumulDeg = %d\n",nodeID,C2[nodeID],deg,cumulDeg);
			} else {
				if (EVEN_ODD)
					C2_d[nodeID] = C1_d[nodeID];
				else
					C1_d[nodeID] = C2_d[nodeID];
				//				printf("new color (NO conf) C2[%d] = %d\n",nodeID,C2[nodeID]);
			}

		// CASE: NO conflicts
		} else {
			if (EVEN_ODD)
				C2_d[nodeID] = fixColor(states + nodeID, str->neighs + cumulDeg, C1_d[nodeID]);
			else
				C1_d[nodeID] = fixColor(states + nodeID, str->neighs + cumulDeg, C2_d[nodeID]);
			//			printf("fix C2[%d] = %d\n",nodeID,C2[nodeID]);
		}
	}
}


/**
 * Start the coloring on the graph
 */
template<typename nodeW, typename edgeW>
void ColoringMCMC<nodeW, edgeW>::run() {


	//	float* PHI;
//	cuSts = cudaMallocManaged(&PHI, nCol * sizeof(float)); cudaCheck(cuSts, __FILE__, __LINE__);

	// normalization factors used to sample a distribution on GPU
//	float a = (1 - param.epsilon * (nCol - 1));
//	for (unsigned i = 0; i < nCol; i++) {
//		PHI[i] = a * dist.CDF.prob[i];
//		for (unsigned k = 0; k < nCol; k++) {
//			if (k != i)
//				PHI[i] += param.epsilon * dist.CDF.prob[k];
//		}
//	}

	std::cout << "lambda = " << param.lambda << std::endl;
	// fill constant memory
	cudaMemcpyToSymbol(LAMBDA, &(param.lambda), sizeof(float));
	cudaMemcpyToSymbol(NCOLS, &(param.nCol), sizeof(col_sz));
	cudaMemcpyToSymbol(EPSILON, &(param.epsilon), sizeof(float));
	cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
	cudaMemcpyToSymbol(CUMULSUMDIST, &(dist.CDF.prob[dist.CDF.length - 1]),
			sizeof(float));

	// init coloring CPU
	// TODO: quarto argomento è il seed
	CPURand::discreteSampling(&(dist.CDF), C, nnodes, 0);


	// manage conflicts
	uint32_t numConflictsOLD = 0;
	for (int i = 0; i < nnodes; i++) {
		// TODO: questo genererà dei segFault in esecuzione...
		node_sz cumulDeg = graphStruct_d->cumulDegs[i];
		node_sz deg = graphStruct_d->cumulDegs[i + 1] - cumulDeg;
		uint32_t nc = ColoringMCMC_k::checkConflicts(i, deg, graphStruct_d->neighs + cumulDeg, C);
		numConflictsOLD += nc;
	}
	std::cout << "numConflicts init = " << numConflictsOLD << std::endl;
	cuSts = cudaMemcpy(C1_d, C, nnodes*sizeof(col), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);
	uint32_t * numConflicts_d;
	cuSts = cudaMalloc((void**)&numConflicts_d, sizeof(uint32_t)); cudaCheck(cuSts, __FILE__, __LINE__);
	uint32_t numConflicts = 1;

	// start coloring
	uint32_t nRound = 0;
	bool STOP = 1;
	cudaEventRecord(start);
	while (STOP) {
		nRound++;
		std::cout << "ROUND # " << nRound << std::endl;
		cuSts = cudaMemcpy(numConflicts_d, &numConflicts, sizeof(unsigned int), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);
		ColoringMCMC_k::drawNewColoring<<<blocksPerGrid, threadsPerBlock>>>(numConflicts_d, states, graphStruct_d, C1_d, C2_d, colNeigh);
		cuSts = cudaMemcpy(&numConflicts, numConflicts_d, sizeof(unsigned int), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		cudaDeviceSynchronize();
		std::cout << "GPU num GLOBAL conflicts = " << numConflicts << std::endl;

		if (!numConflicts)
			STOP = 0;
		else {
			if (numConflicts > numConflictsOLD) {
				if (param.ratioFreezed > 0.45) {
					param.ratioFreezed = param.ratioFreezed-0.5;
					std::cout << "numConflicts = " << numConflicts << "    numConflictsOLD = " << numConflictsOLD << std::endl;
					std::cout << "freeze = " << param.ratioFreezed << std::endl;
					cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
				}
				else {
					param.lambda = param.lambda*0.9;
					param.ratioFreezed = 1;
					CPURand::createExpDistribution(&dist, dist.lambda, nCol); // form: exp(-lambda*c/nCol), c = 1,2...
					cudaMemcpyToSymbol(LAMBDA, &(param.lambda), sizeof(float));
					cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
				}
				numConflictsOLD = numConflicts;
				if (nRound % 2)
					numConflicts = 0;
				else
					numConflicts = 1;
			}
		}
//		cin.ignore();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//float elapsedTimeSec = milliseconds/1000.0f;


	// build coloring
	// TODO: penso sia l'analogo di convert_to_standard_notation()
	//buildColoring(C, nnodes);


	//cudaFree(states);
	//cudaFree(C1_d);
	//cudaFree(C2_d);
	//cudaFree(colNeigh);
//	cudaFree(numConflicts);
//	cudaDeviceReset();

}

// Stampa grafo
template<typename nodeW, typename edgeW>
__global__ void ColoringMCMC_k::print_graph_k( uint32_t nnodes, const node_sz * const cumulDegs, const node * const neighs ) {
	uint32_t deg_i, offset;

	printf( "numero nodi: %d", nnodes );
	printf( "numero nodi 2: %d", nnodes );

	for (uint32_t idx = 0; idx < nnodes; idx++) {
		offset = cumulDegs[idx];
		deg_i = cumulDegs[idx+1] - cumulDegs[idx];
		printf( "Nodo %d - Neigh: ", idx );
		for (uint32_t i = 0; i < deg_i; i++)
			printf( "%d ", neighs[offset + i] );
		printf( "\n" );
	}
	printf( "\n" );
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringMCMC<col, col>;
template class ColoringMCMC<float, float>;

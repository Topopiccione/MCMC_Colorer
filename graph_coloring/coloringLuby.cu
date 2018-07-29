// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif
#include <memory>
#include <iostream>
#include <algorithm>
#include <time.h>

#include "coloring.h"
#include "coloringLuby.h"
#include "graph/graph.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPUStream.h"


template<typename nodeW, typename edgeW>
ColoringLuby<nodeW,edgeW>::ColoringLuby( Graph<nodeW,edgeW> * inGraph_d, curandState * randStates ) :
	Colorer<nodeW,edgeW>( inGraph_d ),
	graphStruct_d( inGraph_d->getStruct() ),
	nnodes( inGraph_d->getStruct()->nNodes ),
	randStates( randStates ),
	numOfColors( 0 ),
	coloredNodesCount( 0 ),
	threadId( 0 ) {

	coloring_h = std::unique_ptr<int[]>( new int[nnodes] );

	cuSts = cudaMalloc( (void**)&coloring_d, nnodes * sizeof( int ) );	cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&is_d, nnodes * sizeof( bool ) ); 		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&i_i_d, nnodes * sizeof( bool ) ); 		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&cands_d, nnodes * sizeof( bool ) ); 	cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&nodeLeft_d, sizeof( bool ) ); 			cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&uncoloredNodes_d, sizeof( bool ) ); 	cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&numOfColors_d, sizeof( int ) ); 		cudaCheck( cuSts, __FILE__, __LINE__ );

	// pointer per il Coloring in output
	outColoring_d = std::unique_ptr<Coloring>( new Coloring );

	// configuro la griglia e i blocchi
	threadsPerBlock = dim3( 128, 1, 1 );
	blocksPerGrid = dim3( (nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );
}

template<typename nodeW, typename edgeW>
ColoringLuby<nodeW, edgeW>::~ColoringLuby() {

	// Dati allocati nel costruttore
	cuSts = cudaFree( numOfColors_d ); 		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( uncoloredNodes_d );	cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( nodeLeft_d ); 		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( cands_d ); 			cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( i_i_d );				cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( is_d ); 				cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( coloring_d ); 		cudaCheck( cuSts, __FILE__, __LINE__ );
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
Coloring* ColoringLuby<nodeW,edgeW>::getColoringGPU() {
	return outColoring_d.get();
}

template<typename nodeW, typename edgeW>
void ColoringLuby<nodeW, edgeW>::printgraph() {
	ColoringLuby_k::print_graph_k<nodeW, edgeW> <<< 1, 1 >>> (nnodes, graphStruct_d->cumulDegs, graphStruct_d->neighs);
}


template<typename nodeW, typename edgeW>
void ColoringLuby<nodeW, edgeW>::convert_to_standard_notation(){
	int idx;

	uint32_t *	colClass =  new uint32_t[nnodes] ;
	uint32_t *	cumulSize = new uint32_t[numOfColors+1] ;

	idx=0;
	memset( colClass, 0, nnodes*sizeof(uint32_t) );
	memset( cumulSize, 0, (numOfColors+1)*sizeof(uint32_t) );

	// Ciclo sui colori
	for(int c=0; c<numOfColors; c++){
		// NB: i colori in luby vanno da 1 a numOfColors

		// Ciclo sui nodi
		for(int i=0; i<nnodes; i++){
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

	for (int i = 0; i < numOfColors; i++) {
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
			for (int k = 0; k < ISsize; k++) {
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
	int temp, size;
	temp=0;
	for (int i = 0; i < numOfColors; i++) {
		printf( "Colore %d: ", i );
		size=cumulSize[i+1]-cumulSize[i];
		for (int j = 0; j < size; j++){
			printf( "%d ", colClass[temp] );
			temp++;
		}
		printf( "\n" );
	}
#endif

	delete[] colClass;
	delete[] cumulSize;
}


// Rimuove dal vettore dei candidati i nodi già colorati
__global__ void ColoringLuby_k::prune_eligible( const int nnodes, const int * const coloring_d, bool *const cands_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nnodes)
		return;
	cands_d[idx] = (coloring_d[idx] == 0) ? 1 : 0;
}


// Scelta casuale (non controllata) tra i vari nodi candidati non ancora colorati
__global__ void ColoringLuby_k::set_initial_distr_k( int nnodes, curandState * states, const bool * const cands_d, bool * const i_i_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nnodes)
		return;

	/* Per ogni thread lancio una moneta
	Se esce 1 AND il nodo idx è un possibile candidato -> il nodo idx viene scelto
	Scelta casuale più efficiente di altri metodi (?)
	*/
	float randnum = curand_uniform( &states[idx] );
	i_i_d[idx] = (randnum < 0.5) ? (1 && cands_d[idx]) : 0;
}


// Controllo se esistono conflitti nella scelta dei candicati (presenza di nodi adiacenti) e li elimino
template<typename nodeW, typename edgeW>
__global__ void ColoringLuby_k::check_conflicts_k( int nnodes, const node_sz * const cumulDegs, const node * const neighs, bool * const i_i_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int deg_i, neigh_ijDeg;
	int offset, neigh_ij;

	if (idx >= nnodes)
		return;

	//se il nodo idx non è stato scelto da set_initial_distr_k, mi fermo
	if (i_i_d[idx] == 0)
		return;

	offset = cumulDegs[idx];
	deg_i = cumulDegs[idx+1] - cumulDegs[idx];

	for (int j = 0; j < deg_i; j++){
		neigh_ij = neighs[offset + j];
		neigh_ijDeg = cumulDegs[neigh_ij+1] - cumulDegs[neigh_ij];

		//se anche il nodo neigh_ij (vicino del nodo idx) è stato scelto, ne elimino uno da i_i_d
		//la discriminazione assicura che solo uno dei due venga rimosso
		if (i_i_d[neigh_ij] == 1) {
			if ((deg_i <= neigh_ijDeg) /*&& (neigh_ij >= idx)*/) {
				i_i_d[idx] = 0;
				//break;
			} else {
				i_i_d[neigh_ij] = 0;
			}
		}
	}
}


// Tutti i candidati sono stati controllati e sono stati eliminati i conflitti (presenza di nodi adiacenti)
// Rimuovo loro e tutti i loro vicini da cands_d
// Me li segno su is_d
template<typename nodeW, typename edgeW>
__global__ void ColoringLuby_k::update_eligible_k( int nnodes, const node_sz * const cumulDegs, const node * const neighs, const bool * const i_i_d, bool * const cands_d, bool * const is_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int deg_i, offset;

	if (idx >= nnodes)
		return;

	offset = cumulDegs[idx];
	deg_i = cumulDegs[idx+1] - cumulDegs[idx];

	is_d[idx] |= i_i_d[idx];

	//se il nodo idx non è stato scelto, mi fermo
	if (i_i_d[idx] == 0)
		return;

	//se il nodo idx è stato scelto e non è in conflitto
	//lo segno su is_d
	//is_d[idx] |= i_i_d[idx];

	//lo rimuovo dai candidati
	cands_d[idx] = 0;

	//e rimuovo dai candidati i suoi nodi adiacenti
	for (int j = 0; j < deg_i; j++)
		cands_d[ neighs[offset + j] ] = 0;
	return;
}


// Controllo se ci sono ancora dei candidati in cands_d
__global__ void ColoringLuby_k::check_finished_k( int nnodes, const bool * const cands_d, bool * const nodeLeft_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	if (cands_d[idx] == 1)
		*nodeLeft_d = true;
}


// Coloro i nodi candidati presenti in is_d e controllo se ci sono ancora nodi non colorati
__global__ void ColoringLuby_k::add_color_and_check_uncolored_k( int nnodes, int numOfColors, const bool * const is_d, bool * const uncoloredNodes_d, int * const coloring_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	if (is_d[idx] == 1) {
		coloring_d[idx] = numOfColors;
	}

	//ne basta uno per far ripartire il CICLO_1 in run()
	if (coloring_d[idx] == 0)
		*uncoloredNodes_d = true;
}


// Stampa grafo
template<typename nodeW, typename edgeW>
__global__ void ColoringLuby_k::print_graph_k( int nnodes, const node_sz * const cumulDegs, const node * const neighs ) {
	int deg_i, offset;

	printf( "numero nodi: %d", nnodes );
	printf( "numero nodi 2: %d", nnodes );

	for (int idx = 0; idx < nnodes; idx++) {
		offset = cumulDegs[idx];
		deg_i = cumulDegs[idx+1] - cumulDegs[idx];
		printf( "Nodo %d - Neigh: ", idx );
		for (int i = 0; i < deg_i; i++)
			printf( "%d ", neighs[offset + i] );
		printf( "\n" );
	}
	printf( "\n" );
}


/*
	Corpo della generazione colorazione Luby
	FLOW:
		cudaMemset inizializzo array interni
		CICLO_1(finchè tutti i nodi non sono colorati)
			rimuovo da array candidati cands_d i nodi già colorati
			azzero is_d
			CICLO_2(finchè ci sono ancora nodi in cands_d)
				scelta casuale (non controllata) tra i nodi non ancora colorati
				controllo i conflitti e li elimino (nodi adiacenti nella scelta)
				segno i candicati su is_d e rimuovo loro+vicini da cands_d
				controllo se cands_d è vuoto
			RIPETO CICLO_2
			creo un nuovo colore, coloro i candidati del CICLO2 e controllo se ci sono ancora nodi non colorati
		RIPETO CICLO_1

*/
template<typename nodeW, typename edgeW>
void ColoringLuby<nodeW,edgeW>::run() {

#ifdef PRINTCOLORINGTITLE
	std::cout << "\n\033[32;1m** Luby GPU colorer **\033[0m" << std::endl;
#endif

	// Inizializzo gli array per il primo ciclo
	cuSts = cudaMemset( is_d, 0, nnodes * sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( i_i_d, 0, nnodes * sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( cands_d, 1, nnodes * sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( coloring_d, 0, nnodes * sizeof( int ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( uncoloredNodes_d, 1, sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );

	std::unique_ptr<bool[]> tempPrint( new bool[nnodes] );

	uncoloredNodes_h = true;

#ifdef VERBOSECOLORING
	std::cout << "Colore: " << std::endl;
#endif

	while (uncoloredNodes_h) {
		// Rimuovo dal vettore dei candidati tutti i nodi gi� colorati
		ColoringLuby_k::prune_eligible <<< blocksPerGrid, threadsPerBlock >>> (nnodes, coloring_d, cands_d);
		cuSts = cudaMemset( is_d, 0, nnodes * sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
		cudaDeviceSynchronize();

#ifdef DEBUGPRINT_K
		cuSts = cudaMemcpy( tempPrint.get(), cands_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
		printf( "candidati:\n" );
		for (int i = 0; i < nnodes; i++) {
			if (tempPrint[i] == 1) printf( "%d ", i );
		}
		printf( "\n" );
#endif

		do {
			// Imposto vettore possibili candidati, prima scelta casuale
			ColoringLuby_k::set_initial_distr_k <<< blocksPerGrid, threadsPerBlock >>> (nnodes, randStates, cands_d, i_i_d);
			cudaDeviceSynchronize();

#ifdef DEBUGPRINT_K
			cuSts = cudaMemcpy( tempPrint.get(), i_i_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
			printf( "candidato ad IS:\n" );
			for (int i = 0; i < nnodes; i++) {
				if (tempPrint[i] == 1) printf( "%d ", i );
			}
			printf( "\n" );
#endif

			// Controllo sui conflitti (presenza di nodi adiacenti) e li elimino
			ColoringLuby_k::check_conflicts_k<nodeW,edgeW> <<< blocksPerGrid, threadsPerBlock >>> (nnodes, graphStruct_d->cumulDegs, graphStruct_d->neighs, i_i_d);
			cudaDeviceSynchronize();


#ifdef DEBUGPRINT_K
			cuSts = cudaMemcpy( tempPrint.get(), i_i_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
			printf( "candidato ad IS pulito:\n" );
			for (int i = 0; i < nnodes; i++) {
				if (tempPrint[i] == 1) printf( "%d ", i );
			}
			printf( "\n" );
#endif

			// Segno i candicati controllati su is_d e rimuovo loro+vicini da cands_d
			ColoringLuby_k::update_eligible_k<nodeW,edgeW> <<< blocksPerGrid, threadsPerBlock >>> (nnodes, graphStruct_d->cumulDegs, graphStruct_d->neighs, i_i_d, cands_d, is_d);
			cudaDeviceSynchronize();

#ifdef DEBUGPRINT_K
			cuSts = cudaMemcpy( tempPrint.get(), cands_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
			printf( "cands aggiornato:\n" );
			for (int i = 0; i < nnodes; i++) {
				if (tempPrint[i] == 1) printf( "%d ", i );
			}
			printf( "\n" );

			cuSts = cudaMemcpy( tempPrint.get(), is_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
			printf( "IS prima del controllo:\n" );
			for (int i = 0; i < nnodes; i++) {
				if (tempPrint[i] == 1) printf( "%d ", i );
			}
			printf( "\n" );
#endif
			// Controllo se ci sono ancora candidati per il nuovo colore

			cuSts = cudaMemset( nodeLeft_d, 0, sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
			ColoringLuby_k::check_finished_k <<< blocksPerGrid, threadsPerBlock >>> (nnodes, cands_d, nodeLeft_d);
			cudaDeviceSynchronize();
			cuSts = cudaMemcpy( &nodeLeft_h, nodeLeft_d, sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );

		} while (nodeLeft_h);

#ifdef DEBUGPRINT_K
		printf("OUT!\n");
		cuSts = cudaMemcpy( tempPrint.get(), is_d, nnodes * sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
		printf( "\n\ncolore: " );
		for (int i = 0; i < nnodes; i++) {
			if (tempPrint[i] == 1) printf( "%d ", i );
		}
		printf( "\n\n" );
#endif

		//add_color_and_check_uncolored();
		numOfColors++;
		cuSts = cudaMemset( uncoloredNodes_d, 0, sizeof( bool ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
		ColoringLuby_k::add_color_and_check_uncolored_k <<< blocksPerGrid, threadsPerBlock >>> (nnodes, numOfColors, is_d, uncoloredNodes_d, coloring_d);
		cudaDeviceSynchronize();
		cuSts = cudaMemcpy( &uncoloredNodes_h, uncoloredNodes_d, sizeof( bool ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );


#ifdef VERBOSECOLORING
		std::cout << numOfColors << " ";
#endif

		if (numOfColors > nnodes) {
			std::cout << "Qualcosa è andato storto..." << std::endl; abort();
		}

	}

#ifdef VERBOSECOLORING
	std::cout << std::endl << "Numero di colori trovati " << numOfColors << std::endl;
#endif

	cuSts = cudaMemcpy( coloring_h.get(), coloring_d, nnodes * sizeof( int ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );


#ifdef TESTCOLORINGCORRECTNESS
	// test, posso per verificare che tutti i nodi siano stati effettivamente colorati
	if (std::find( coloring_h.get(), coloring_h.get() + nnodes, 0 ) != coloring_h.get() + nnodes)
		std::cout << "Uh oh..." << std::endl;
#endif


	convert_to_standard_notation();

}




//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringLuby<col, col>;
template class ColoringLuby<float, float>;

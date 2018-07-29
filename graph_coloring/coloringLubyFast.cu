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
#include "graph/graph.h"

#include "GPUutils/GPUutils.h"
#include "GPUutils/GPUStream.h"

template<typename nodeW, typename edgeW>
void ColoringLuby<nodeW, edgeW>::run_fast() {

#ifdef PRINTCOLORINGTITLE
	std::cout << "\n\033[32;1m** Luby GPU fast colorer **\033[0m" << std::endl;
#endif

	//int numOfColors_d;
	//cuSts = cudaMalloc( (void**)&numOfColors_d, sizeof( int ) );

	// Preparo gli array per il primo giro
	cuSts = cudaMemset( is_d, 0, nnodes * sizeof( bool ) );  cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( i_i_d, 0, nnodes * sizeof( bool ) );  cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( cands_d, 1, nnodes * sizeof( bool ) );  cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( coloring_d, 0, nnodes * sizeof( int ) );  cudaCheck( cuSts, __FILE__, __LINE__ );

	ColoringLuby_k::fast_colorer_k << < 1, 1 >> > (nnodes, graphStruct_d->cumulDegs, graphStruct_d->neighs, randStates,
		uncoloredNodes_d, nodeLeft_d, i_i_d, is_d, cands_d, numOfColors_d, coloring_d);
	cudaDeviceSynchronize();
	cuSts = cudaGetLastError();
	if (cuSts != cudaSuccess) { std::cout << "CUDA ERROR: fast_colorer_k " << cudaGetErrorString( cuSts ) << std::endl; abort(); }			// DEBUG

	cuSts = cudaMemcpy( coloring_h.get(), coloring_d, nnodes * sizeof( int ), cudaMemcpyDeviceToHost );  cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( &numOfColors, numOfColors_d, sizeof( int ), cudaMemcpyDeviceToHost );  cudaCheck( cuSts, __FILE__, __LINE__ );

#ifdef VERBOSECOLORING
	std::cout << "Numero di colori trovati " << numOfColors << std::endl;
#endif

	convert_to_standard_notation();
	//cudaFree( numOfColors_d );
}

// Lanciare questo thread con configurazione <<< 1, 1 >>>
//template<typename nodeW, typename edgeW>
__global__ void ColoringLuby_k::fast_colorer_k(
	const int nnodes,
	const node_sz * const graphIdxDeg_d,
	const node * const graphNeigh_d,
	curandState * randStates,
	bool * const uncolored_d,
	bool * const nodeLeft_d,
	bool * const i_i_d,
	bool * const is_d,
	bool * const cands_d,
	int * const numOfColors_d,
	int * const coloring_d ) {

	dim3 threadsPerBlock( 128, 1, 1 );
	dim3 blocksPerGrid( (nnodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );

	*uncolored_d = true;
	*nodeLeft_d = false;
	*numOfColors_d = 0;
	//int loopCounter = 0;

	while (*uncolored_d) {
		// Rimuovo dal vettore dei candidati tutti i nodi gi� colorati
		ColoringLuby_k::prune_eligible_clear_is << < blocksPerGrid, threadsPerBlock >> > (nnodes, coloring_d, cands_d, is_d);
		cudaDeviceSynchronize();

		do {
			// imposto vettore possibili candidati
			//ColoringLuby_k::set_initial_distr_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, randStates, cands_d, graphIdxDeg_d, i_i_d);
			ColoringLuby_k::set_initial_distr_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, randStates, cands_d, i_i_d);
			cudaDeviceSynchronize();

			// controllo sui conflitti
			ColoringLuby_k::check_conflicts_fast_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, graphIdxDeg_d, graphNeigh_d, i_i_d);
			cudaDeviceSynchronize();

			// Rimuovo dalla lista dei candidati  tutti i is[i] = 1 e i loro vicini
			ColoringLuby_k::update_eligible_fast_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, graphIdxDeg_d, graphNeigh_d, i_i_d, cands_d, is_d);
			cudaDeviceSynchronize();

			// check_finished
			*nodeLeft_d = false;
			ColoringLuby_k::check_finished_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, cands_d, nodeLeft_d);
			cudaDeviceSynchronize();

			//loopCounter++;

		} while (*nodeLeft_d);

		(*numOfColors_d)++;
		(*uncolored_d) = false;
		ColoringLuby_k::add_color_and_check_uncolored_k << < blocksPerGrid, threadsPerBlock >> > (nnodes, *numOfColors_d, is_d, uncolored_d, coloring_d);
		cudaDeviceSynchronize();
	}
	//printf( "nnodes: %d - loop: %d \n", nnodes, loopCounter );

}


__global__ void ColoringLuby_k::prune_eligible_clear_is( const int nnodes, const int * const coloring_d, bool *const cands_d, bool * const is_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nnodes)
		return;
	cands_d[idx] = (coloring_d[idx] == 0) ? 1 : 0;
	is_d[idx] = 0;
}

__global__ void ColoringLuby_k::check_conflicts_fast_k( int nnodes, const node_sz * const cumulDegs, const node * const neighs, bool * const i_i_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int deg_i, neigh_ijDeg;
	int offset, neigh_ij;

	if (idx >= nnodes)
		return;

	if (i_i_d[idx] == 0)
		return;

	offset = cumulDegs[idx];
	deg_i = cumulDegs[idx + 1] - cumulDegs[idx];

	for (int j = 0; j < deg_i; j++){
		neigh_ij = neighs[offset + j];
		neigh_ijDeg = cumulDegs[neigh_ij + 1] - cumulDegs[neigh_ij];

		if (i_i_d[neigh_ij] == 1) {

			if ((deg_i <= neigh_ijDeg) /*&& (neigh_ij >= idx)*/) {
				i_i_d[idx] = 0;
			}
			else {
				i_i_d[neigh_ij] = 0;
			}
		}
	}
}

__global__ void ColoringLuby_k::update_eligible_fast_k( int nnodes, const node_sz * const cumulDegs, const node * const neighs, const bool * const i_i_d, bool * const cands_d, bool * const is_d ) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int deg_i, offset;

	if (idx >= nnodes)
		return;

	offset = cumulDegs[idx];
	deg_i = cumulDegs[idx + 1] - cumulDegs[idx];

	is_d[idx] |= i_i_d[idx];

	if (i_i_d[idx] == 0)
		return;

	cands_d[idx] = 0;

	for (int j = 0; j < deg_i; j++)
		cands_d[neighs[offset + j]] = 0;
	return;
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringLuby<col, col>;
template class ColoringLuby<float, float>;

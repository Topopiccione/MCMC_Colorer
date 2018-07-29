#pragma once
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "graph/graph.h"
#include "GPUutils/GPUStream.h"

//#define DEBUGPRINT_K
//#define TESTCOLORINGCORRECTNESS
//#define PRINTCOLORINGTITLE
//#define VERBOSECOLORING
//#define PRINT_COLORING


template<typename nodeW, typename edgeW>
class ColoringLuby : public Colorer<nodeW, edgeW> {
public:

	ColoringLuby( Graph<nodeW, edgeW> * inGraph_d, curandState * randStates );

	~ColoringLuby();

	void			run();
	void			run_fast();

	Coloring	*	getColoringGPU();

protected:
	int				nnodes;
	int				numOfColors;
	int				coloredNodesCount;

	std::unique_ptr<int[]> coloring_h;

	//dati del grafo
	const GraphStruct<nodeW, edgeW>	* const	graphStruct_d;
	std::unique_ptr<Coloring> outColoring_d;

	int				threadId;

	int			*	coloring_d;		// each element denotes a color
	bool		*	is_d;			// working array: IS candidate
	bool		*	cands_d;		// working array: list of available nodes to be picked as candidate
	bool		*	i_i_d;
	bool			nodeLeft_h;
	bool		*	nodeLeft_d;
	bool			uncoloredNodes_h;
	bool		*	uncoloredNodes_d;
	int			*	numOfColors_d;

	int			**	tempIS;			// Tiene traccia dei puntatori agli outColoring_d->IS[i]
									// Serve per la cudaFree!!!

	cudaError_t		cuSts;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
	curandState *	randStates;

	void			printgraph();

	void			convert_to_standard_notation();

};

// Global kernels
namespace ColoringLuby_k {

	__global__ void prune_eligible( const int nnodes, const int * const coloring_d, bool *const cands_d );
	__global__ void set_initial_distr_k( int nnodes, curandState * states, const bool * const cands_d, bool * const i_i_d );
	template<typename nodeW, typename edgeW>
	__global__ void check_conflicts_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, bool * const i_i_d );
	template<typename nodeW, typename edgeW>
	__global__ void update_eligible_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, const bool * const i_i_d, bool * const cands_d, bool * const is_d );
	__global__ void check_finished_k( int nnodes, const bool * const cands_d, bool * const nodeLeft_d );
	__global__ void add_color_and_check_uncolored_k( int nnodes, int numOfColors, const bool * const is_d, bool * const uncoloredNodes_d, int * const coloring_d );

	template<typename nodeW, typename edgeW>
	__global__ void print_graph_k( int nnodes, const node_sz * const cumulSize, const node * const neighs );

	__global__ void fast_colorer_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, curandState * randStates,
	bool * const uncolored_d, bool * const nodeLeft_d, bool * const i_i_d, bool * const is_d, bool * const cands_d,	int * const numOfColors_d,
	int * const coloring_d );

	__global__ void prune_eligible_clear_is( const int nnodes, const int * const coloring_d, bool *const cands_d, bool * const is_d );
	__global__ void check_conflicts_fast_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, bool * const i_i_d );
	__global__ void update_eligible_fast_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, const bool * const i_i_d, bool * const cands_d, bool * const is_d );

};

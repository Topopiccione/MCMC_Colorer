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


typedef uint32_t col;     // node color
typedef uint32_t col_sz;     // node color

#define FOR_ALL_NODE for (node i = 0; i < n; i++)   // loop on graph nodes
#define FOR_ALL_COL_0 for (col c = 0; c < nCol; c++)  // loop on coloring colors

// graph coloring
struct Coloring {
	uint32_t	 		nCol { 0 };				// num of color classes
	uint32_t		*	colClass { nullptr };	// list (array) of all node colors class by class
	uint32_t		*	cumulSize { nullptr };	// cumulative color class sizes

	/// return the size of class c
	uint32_t classSize(col c) {
		return cumulSize[c] - cumulSize[c-1];
	}
};

template<typename nodeW, typename edgeW> class Colorer {
public:
	Colorer(Graph<nodeW,edgeW>* g) : graph{g} {
		std::default_random_engine eng {};
		std::uniform_real_distribution<> randU(0.0, 1.0);
		seed = static_cast<float>( randU(eng) );
	}
	virtual ~Colorer() {};
	/*virtual*/ void run() /*= 0*/;
	float efficiencyNumProcessors(unsigned);
	Coloring* getColoring();
	bool checkColoring();
	col_sz getNumColor() const;
	void buildColoring(col*, node_sz);
	void print(bool);
	double getElapsedTime() {return elapsedTimeSec;};
	void setSeed(float s) {seed = s;};
	void verbose() {verb = 1;};

protected:
	std::string name;
	Graph<nodeW,edgeW>* graph;
	Coloring* coloring{};
	float* meanClassDeg{};	  // mean degs of class nodes
	float meanClassDegAll{0}; // mean deg over all class nodes
	float stdClassDegAll{0};  // std of deg of all IS nodes
	double elapsedTimeSec{0};
	float seed{0};            // for random generator
	bool verb{0};
};

/**
 *  Get a greedy colorings of a graph
 */
template<typename nodeW, typename edgeW>
class ColoringGreedyCPU: public Colorer<nodeW, edgeW> {
public:
	ColoringGreedyCPU( Graph<nodeW, edgeW>* g );
	void run() /*override*/;
	~ColoringGreedyCPU();
};




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

	//template<typename nodeW, typename edgeW>
	__global__ void fast_colorer_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, curandState * randStates,
	bool * const uncolored_d, bool * const nodeLeft_d, bool * const i_i_d, bool * const is_d, bool * const cands_d,	int * const numOfColors_d,
	int * const coloring_d );

	__global__ void prune_eligible_clear_is( const int nnodes, const int * const coloring_d, bool *const cands_d, bool * const is_d );
	__global__ void check_conflicts_fast_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, bool * const i_i_d );
	__global__ void update_eligible_fast_k( int nnodes, const node_sz * const cumulSize, const node * const neighs, const bool * const i_i_d, bool * const cands_d, bool * const is_d );

};

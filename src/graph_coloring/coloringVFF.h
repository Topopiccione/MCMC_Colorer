#pragma once
#include <graph/graph.h>
#include <memory>
#include <graph_coloring/coloringGreedyFF.h>

template<typename nodeW, typename edgeW>
class ColoringVFF : public Colorer<nodeW, edgeW>{
    public:

    //Constructor and destructor
    ColoringVFF(Graph<nodeW, edgeW>* graph_d);
    ~ColoringVFF();

    //Run function to start the Vertex-centric parallel scheme 
    //for running and balancing a Greedy First Fit coloring
    void run();

    Coloring* getColoring();
    void saveStats(size_t iteration, float duration, std::ofstream &file);
    void saveColor(std::ofstream &file);


    protected:
    uint32_t numNodes;                  //  number of nodes in the graph
    uint32_t numColors;                 //  number of colors used in the graph

    const GraphStruct<nodeW, edgeW>* const graphStruct_device;
    
    std::unique_ptr<uint32_t[]> coloring_host;      //  array of colors (unsigned integers) to be indexed with nodes
    uint32_t*                   coloring_device;    //  as coloring_host, but used by device

    cudaError_t cudaStatus;         //  used to check CUDA calls are ok and don't return errors
    dim3        threadsPerBlock;    //  number of threads in a block, as a 3D array
    dim3        blocksPerGrid;      //  number of blocks in the grid, as a 3D array
    
    void convert_to_standard_notation();

    private:
    void run_coloring();
    void run_balancing();
};

namespace BalancingVFF_k{

    __global__ void detect_unbalanced_nodes(const uint32_t numNodes, const uint32_t* coloring_device, const uint32_t* cumulBinSizes, const uint32_t gamma, bool* unbalanced_nodes);
    
    __global__ void is_unbalanced(const uint32_t numNodes, const bool* unbalanced_nodes, bool* result);

    template<typename nodeW, typename edgeW>
    __global__ void tentative_rebalancing(const uint32_t numNodes, const uint32_t numColors, const uint32_t* input_coloring, const uint32_t* cumulBinSizes, const node* const neighs, const node_sz* const cumulDegs, const uint32_t gamma, uint32_t* output_coloring, bool* unbalanced_nodes, uint32_t* forbidden_colors);
    
    __global__ void update_bins(const uint32_t numNodes, const uint32_t numColors, const uint32_t* coloring, uint32_t* binSizes);
    
    template<typename nodeW, typename edgeW>
    __global__ void solve_conflicts(const uint32_t numNodes, const uint32_t* coloring, const node* neighs, const node_sz* cumulDegs, bool* unbalanced_nodes);

    //  NOTE 1: This function is currently unused  
    //  __global__ void cumulate_bins(const uint32_t numColors, uint32_t* binCumulSizes);                                       //NOTE 2: to be called with one thread

    __global__ void ensure_not_looping(const uint32_t numNodes, bool* unbalanced_nodes, uint32_t store_dim, bool* output);      //NOTE: to be called with one thread
};

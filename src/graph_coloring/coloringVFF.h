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
    
    protected:
    uint32_t numNodes;
    uint32_t numColors;

    const GraphStruct<nodeW, edgeW>* const graphStruct_device;
    
    std::unique_ptr<uint32_t[]> coloring_host;
    uint32_t*                   coloring_device;

    cudaError_t cudaStatus;
    dim3        threadsPerBlock;
    dim3        blocksPerGrid;
    
    void convert_to_standard_notation();

    private:
    void run_coloring();
    void run_balancing();
};

namespace BalancingVFF_k{

    __global__ void detect_unbalanced_nodes(const uint32_t numNodes, const uint32_t* coloring_device, const uint32_t* cumulBinSizes, const uint32_t gamma, bool* unbalanced_nodes);
    
    template<typename nodeW, typename edgeW>
    __global__ void tentative_rebalancing(const uint32_t numNodes, const uint32_t numColors, const uint32_t* input_coloring, const uint32_t* cumulBinSizes, const node* const neighs, const node_sz* const cumulDegs, const uint32_t gamma, uint32_t* output_coloring, bool* unbalanced_nodes, uint32_t* forbidden_colors);
    
    __global__ void update_bins(const uint32_t numNodes, const uint32_t numColors, const uint32_t* coloring, uint32_t* binSizes);
    
    template<typename nodeW, typename edgeW>
    __global__ void solve_conflicts(const uint32_t numNodes, const uint32_t* coloring, const node* neighs, const node_sz* cumulDegs, bool* unbalanced_nodes);
};

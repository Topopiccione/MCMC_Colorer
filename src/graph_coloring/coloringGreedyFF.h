#pragma once
#include <graph/graph.h>
#include <memory>

template<typename nodeW, typename edgeW>
class ColoringGreedyFF : public Colorer<nodeW, edgeW>{
    public:
    
    //  Constructor and destructor
    ColoringGreedyFF(Graph<nodeW, edgeW>* graph_d);
    ~ColoringGreedyFF();

    //  Run function to start the Greedy First Fit coloring
    void run();
    
    Coloring* getColoring();
    void saveStats(size_t iteration, float duration, std::ofstream &file);
    void saveColor(std::ofstream &file);
    
    protected:
    uint32_t numNodes;                                              // number of nodes in the graph
    uint32_t numColors;                                             // number of colors used in the graph

    const GraphStruct<nodeW, edgeW> * const graphStruct_device;

    std::unique_ptr<uint32_t[]> coloring_host;                      // array of colors (unsigned integers) to be indexed with nodes
    uint32_t*                   coloring_device;                    // as coloring_host, but used by device
    
    cudaError_t cudaStatus;                                         // used to check CUDA calls are ok and don't return errors
    dim3        threadsPerBlock;                                    // number of threads in a block, as a 3D array
    dim3        blocksPerGrid;                                      // number of blocks in the grid, as a 3D array

    void convert_to_standard_notation();

    private:                        
    //  Data structures used by run() for coloring
    uint32_t    maxColors;
    uint32_t*   temp_coloring;
    uint32_t*   forbiddenColors;
    bool*       uncolored_nodes_device;
};

namespace ColoringGreedyFF_k{

    template<typename nodeW, typename edgeW>
    __global__ void tentative_coloring(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs, uint32_t* forbidden_colors, const uint32_t maxColors);

    template<typename nodeW, typename edgeW>
    __global__ void conflict_detection(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs); 

    //  Note that input_coloring and output_coloring must be pointers to GPU memory
    __global__ void update_coloring_GPU(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring);

    __global__ void check_uncolored_nodes(const uint32_t numNodes, const uint32_t* coloring, bool* uncolored_nodes);
};
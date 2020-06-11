#pragma once
#include <graph/graph.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//other includes 

template<typename nodeW, typename edgeW>
class ColoringGreedyFF : public Colorer<nodeW, edgeW>{
    public:
    
    //Constructor and destructor
    ColoringGreedyFF(Graph<nodeW, edgeW>* graph_d);
    ~ColoringGreedyFF();

    //Run function to start the Greedy First Fit coloring
    void run();

    protected:
    uint32_t numNodes;              //number of nodes in the graph
    uint32_t numColors;             //number of colors used in the graph
//  uint32_t numColors_device;

    const GraphStruct<nodeW, edgeW> * const graphStruct_device;

    thrust::host_vector<uint32_t>   coloring_host;        //array of colors (unsigned integers) to be indexed with nodes
    thrust::device_vector<uint32_t> coloring_device;      //as coloring_host, but used by device
    
    cudaError_t cudaStatus;         //used to check CUDA calls are ok and don't return errors
    dim3        threadsPerBlock;    //number of threads in a block, as a 3D array
    dim3        blocksPerGrid;      //number of blocks in the grid, as a 3D array

    template<typename nodeW, typename edgeW>
    __global__ void tentative_coloring(const uint32_t numNodes, thrust::device_vector<uint32_t> input_coloring, thrust::device_vector<uint32_t> output_coloring, const node_sz * const cumulDegs, const node * const neighs, const uint32_t maxColors);
    template<typename nodeW, typename edgeW>
    __global__ void conflict_detection(const uint32_t numNodes, thrust::device_vector<uint32_t> input_coloring, thrust::device_vector<uint32_t> output_coloring, const node_sz * const cumulDegs, const node * const neighs, const uint32_t maxColors);
    template<typename nodeW, typename edgeW>
    void convert_to_standard_notation();
}
#include <coloring.h>
#include <coloringVFF.h>
#include <graph/graph.h>

#include "GPUutils/GPUutils.h"          //is required in order to use cudaCheck
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define BIN_SIZE(cumulBinSize, binIndex) (cumulBinSize[binIndex] - cumulBinSize[binIndex])

//Constructor - initialization as in ColoringGreedyFF
template<typename nodeW, typename edgeW>
ColoringVFF<nodeW, edgeW>::ColoringVFF(Graph<nodeW, edgeW>* graph_d) : 
    Colorer<nodeW, edgeW>(graph_d), graphStruct_device(graph_d->getStruct()),
    numNodes(graph_d->getStruct()->nNodes), numColors(0) 
{

    coloring_host = std::unique_ptr<uint32_t[]>(new uint32_t[numNodes]);

    cudaStatus = cudaMalloc((void**)&coloring_device, numNodes * sizeof(uint32_t));     cudaCheck(cudaStatus, __FILE__, __LINE__);

    threadsPerBlock = dim3(128, 1, 1);
    blocksPerGrid = dim3((numNodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
}

template<typename nodeW, typename edgeW>
ColoringVFF<nodeW, edgeW>::~ColoringVFF(){
    cudaStatus = cudaFree(coloring_device);     cudaCheck(cudaStatus, __FILE__, __LINE__);

    if(this->coloring != nullptr)
        free(this->coloring);
}

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run(){
    run_coloring();

    convert_to_standard_notation();

    run_balancing();
} 

template <typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run_coloring(){
    //We are assuming getMaxNodeDeg() returns a valid result here
    uint32_t maxColors = this->graph->getMaxNodeDeg() + 1;   

    cudaStatus = cudaMemset(coloring_device, 0, numNodes * sizeof(uint32_t));           
    cudaCheck(cudaStatus, __FILE__, __LINE__); 

    uint32_t* temp_coloring;
    cudaStatus = cudaMalloc((void**)&temp_coloring, numNodes * sizeof(uint32_t));       
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    uint32_t* forbiddenColors;
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * maxColors * sizeof(uint32_t));
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    bool* uncolored_nodes_device;
    cudaStatus = cudaMalloc((void**)&uncolored_nodes_device, sizeof(bool));
    cudaCheck(cudaStatus, __FILE__, __LINE__)

    bool uncolored_nodes = true;
    while(uncolored_nodes){
        //Tentative coloring on the whole graph, in parallel
        ColoringGreedyFF_k::tentative_coloring<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs, forbiddenColors, maxColors);
        cudaDeviceSynchronize();

        //Update the coloring now that we are sure there is no conflict
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();

        //Checking for conflicts and letting lower nodes win over the others, in parallel
        ColoringGreedyFF_k::conflict_detection<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs);
        cudaDeviceSynchronize();

        //Update the coloring before next loop
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();
        
        cudaStatus = cudaMemset(uncolored_nodes_device, 0, sizeof(bool));
        cudaCheck(cudaStatus, __FILE__, __LINE__);
        ColoringGreedyFF_k::check_uncolored_nodes<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, uncolored_nodes_device);
        cudaDeviceSynchronize();
        cudaStatus = cudaMemcpy(&uncolored_nodes, uncolored_nodes_device, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheck(cudaStatus, __FILE__, __LINE__);
    }

    cudaStatus = cudaFree(temp_coloring);                                               cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(forbiddenColors);                                             cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(uncolored_nodes_device);                                      cudaCheck(cudaStatus, __FILE__, __LINE__);

    cudaStatus = cudaMemcpy(coloring_host.get(), coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    std::set<uint32_t> color_set(coloring_host.get(), coloring_host.get() + numNodes);
    numColors = color_set.size();
}

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run_balancing(){
    const uint32_t gamma_threshold = numNodes / numColors;

    bool* unbalanced_nodes;
    cudaStatus = cudaMalloc((void**)&unbalanced_nodes, sizeof(bool) * numNodes);
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    uint32_t* temp_coloring;
    cudaStatus = cudaMalloc((void**)&temp_coloring, numNodes * sizeof(uint32_t));       
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    uint32_t* forbiddenColors;
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * numColors * sizeof(uint32_t));
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    uint32_t* binCumulSizes_device;
    cudaStatus = cudaMalloc((void**)&binCumulSizes_device, sizeof(uint32_t) * (numColors + 1));
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMemcpy(this->coloring->cumulSize, binCumulSizes_device, sizeof(uint32_t) * (numColors + 1), cudaMemcpyHostToDevice);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    //We use a non-blocking CUDA Stream in order to make bin updating  
    //and conflict checking parallel
    cudaStream_t non_default_stream;
    cudaStreamCreateWithFlags(&non_default_stream, cudaStreamNonBlocking);
    BalancingVFF_k::detect_unbalanced_nodes<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, binCumulSizes_device, gamma_threshold, unbalanced_nodes);
    
    //until there is even one of the nodes flagged as unbalanced
    while(thrust::any_of(thrust::device, unbalanced_nodes, unbalanced_nodes + numNodes, thrust::identity<bool>())){
        BalancingVFF_k::tentative_rebalancing<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, numColors, coloring_device, binCumulSizes_device, graphStruct_device->neighs, graphStruct_device->cumulDegs, gamma_threshold, temp_coloring, unbalanced_nodes, forbiddenColors);
        cudaDeviceSynchronize();
        cudaStatus = cudaMemset(binCumulSizes_device, 0, sizeof(uint32_t) * (numColors + 1));
        cudaCheck(cudaStatus, __FILE__, __LINE__);
        cudaDeviceSynchronize();
        BalancingVFF_k::update_bins<<<blocksPerGrid, threadsPerBlock, 0, non_default_stream>>>(numNodes, numColors, temp_coloring, binCumulSizes_device);
        BalancingVFF_k::solve_conflicts<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, graphStruct_device->neighs, graphStruct_device->cumulDegs, unbalanced_nodes);
        cudaStreamSynchronize(non_default_stream);
        thrust::inclusive_scan(thrust::device, binCumulSizes_device, binCumulSizes_device+(numColors+1), binCumulSizes_device);
        cudaDeviceSynchronize();
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();
    }

    cudaStatus = cudaFree(unbalanced_nodes);                            cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(temp_coloring);                               cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(forbiddenColors);                            cudaCheck(cudaStatus, __FILE__, __LINE__);

    cudaStatus = cudaMemcpy(coloring_host.get(), coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    //Update coloring data
    cudaStatus = cudaMemcpy(this->coloring->cumulSize, binCumulSizes_device, sizeof(uint32_t) * (numColors + 1), cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(binCumulSizes_device);                        cudaCheck(cudaStatus, __FILE__, __LINE__);
    this->coloring->colClass = coloring_host.get();
    //Note that this->coloring->nCol shouldn't be updated
}

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::convert_to_standard_notation(){
    uint32_t *cumulColorClassesSize = new uint32_t[numColors + 1];
    memset(cumulColorClassesSize, 0, (numColors+1)*sizeof(uint32_t));

    //Count how many nodes are colored with <col> and store them in the array, before...
    for(uint32_t col = 1; col < numColors + 1; ++col){
        cumulColorClassesSize[col] = std::count(coloring_host.get(), coloring_host.get() + numNodes, col);
    }
    
    //... you accumulate them in place
    // NOTE: index 0 is skipped, and we can start from 2 because cumulColorClassesSize[0] = 0
    for(uint32_t col = 2; col < numColors + 1; ++col)
    {
        cumulColorClassesSize[col] = cumulColorClassesSize[col] + cumulColorClassesSize[col-1];
    }

    this->coloring = new Coloring();
    this->coloring->nCol = numColors;
    this->coloring->colClass = coloring_host.get();
    this->coloring->cumulSize = cumulColorClassesSize;
}

template<typename nodeW, typename edgeW>
Coloring* ColoringVFF<nodeW, edgeW>::getColoring(){
    return this->coloring;
}

__global__ void BalancingVFF_k::detect_unbalanced_nodes(const uint32_t numNodes, const uint32_t* coloring_device, const uint32_t* cumulBinSizes, const uint32_t gamma, bool* nodes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }

    uint32_t idx_color = coloring_device[idx];
    // gamma is less than the number of nodes in idx bin/color class
    if(gamma < BIN_SIZE(cumulBinSizes, idx_color)){
        nodes[idx] = true;  //idx gets flagged as a node belonging to an unbalanced bin/color class
    }
}

template<typename nodeW, typename edgeW>
__global__ void BalancingVFF_k::tentative_rebalancing(const uint32_t numNodes, const uint32_t numColors, const uint32_t* input_coloring, const uint32_t* cumulBinSizes, const node* const neighs, const node_sz* const cumulDegs, const uint32_t gamma, uint32_t* output_coloring, bool* unbalanced_nodes, uint32_t* forbidden_colors){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }
    if(!unbalanced_nodes[idx]){
        return;
    }
    uint32_t* idx_forbidden_colors = forbidden_colors + idx * numColors;
    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    for(uint32_t j = 0; j < numNeighs; ++j){
        neighbor = neighs[neighsOffset + j];
        idx_forbidden_colors[input_coloring[neighbor]] = idx;
    }

    for(uint32_t i = 1; i <= numColors; ++i){
        if(idx_forbidden_colors[i] != idx && gamma < BIN_SIZE(cumulBinSizes, idx)){
            output_coloring[idx] = i;
            return;            
        }
    }
}

__global__ void BalancingVFF_k::update_bins(const uint32_t numNodes, const uint32_t numColors, const uint32_t* coloring, uint32_t* binSizes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    //Note that "color 0" is invalid and colors are numColors (counting from 1)
    if(idx > numColors || idx == 0){        
        return;
    }

    for(uint32_t i = 0; i < numNodes; ++i){
        if(coloring[i] == idx){
            binSizes[idx] = binSizes[idx] + 1;
        }
    }
}

template<typename nodeW, typename edgeW>
__global__ void BalancingVFF_k::solve_conflicts(const uint32_t numNodes, const uint32_t* coloring, const node* neighs, const node_sz* cumulDegs, bool* unbalanced_nodes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }
    if(unbalanced_nodes[idx] == false){
        return;
    }

    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    for(uint32_t i = 0; i < numNeighs; ++i){
        neighbor = neighs[neighsOffset + i];
        if(coloring[neighbor] == coloring[idx] && idx > neighbor){
            return;
        }
    }

    //there are no conflict on node idx and it can be considered balanced
    unbalanced_nodes[idx] = false;
}

template class ColoringVFF<float, float>;
#include <coloring.h>
#include <coloringGreedyFF.h>
#include <graph/graph.h>

#include "GPUutils/GPUutils.h"          //is required in order to use cudaCheck


#include <set>                          //is used for counting the number of colors
#include <algorithm>                    //count is used for conversion to standard notation

#include <thrust/execution_policy.h>   
#include <thrust/device_ptr.h> 
#include <thrust/find.h>                //is used for finding the first occurence
#include <thrust/scan.h>


template<typename nodeW, typename edgeW>
ColoringGreedyFF<nodeW, edgeW>::ColoringGreedyFF(Graph<nodeW, edgeW>* graph_d) :
    Colorer<nodeW, edgeW>(graph_d), graphStruct_device(graph_d->getStruct()),
    numNodes(graph_d->getStruct()->nNodes), numColors(0) {
    
    // We need to have an array representing the colors of each node
    //both on host...
    coloring_host = std::unique_ptr<int[]>(new int[numNodes]);
    
    //...and on device
    cudaStatus = cudaMalloc((void**)&coloring_device, numNodes * sizeof(uint32_t));     cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    //We setup our grid to be divided in blocks of 128 threads 
    threadsPerBlock = dim3(128, 1, 1);
    blocksPerGrid = dim3((numNodes + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);
}

template<typename nodeW, typename edgeW>
ColoringGreedyFF<nodeW, edgeW>::~ColoringGreedyFF(){
    //We only need to deallocate what we allocated in the constructor
    cudaStatus = cudaFree(coloring_device);     cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    if(this->coloring != nullptr)                 //NOTE: this may be unnecessary
        free(this->coloring);
}

template<typename nodeW, typename edgeW>
void ColoringGreedyFF<nodeW, edgeW>::run(){
    //We are assuming getMaxNodeDeg() returns a valid result here
    uint32_t maxColors = graphStruct_device->getMaxNodeDeg() + 1;   

    cudaStatus = cudaMemset(coloring_device, 0, numNodes * sizeof(uint32_t));           
    cudaCheck(cudaStatus, __FILE__, __LINE__); 

    uint32_t* temp_coloring;
    cudaStatus = cudaMalloc((void**)&temp_coloring, numNodes * sizeof(uint32_t));       
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    uint32_t* forbiddenColors;
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * maxColors * sizeof(uint32_t));
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    thrust::device_ptr<uint32_t> firstUncolored         = thrust::device_ptr<uint32_t>(coloring_device);
    thrust::device_ptr<uint32_t> coloring_device_begin  = thrust::device_ptr<uint32_t>(coloring_device);
    while(firstUncolored != (coloring_device_begin + numNodes)){
        
        //Tentative coloring on the whole graph, in parallel
        ColoringGreedyFF_k::tentative_coloring<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs, forbiddenColors, maxColors);
        cudaDeviceSynchronize();

        //Update the coloring now that we are sure there is no conflict
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();

        //Checking for conflicts and letting lower nodes win over the others, in parallel
        ColoringGreedyFF_k::conflict_detection<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs);
        cudaDeviceSynchronize();

        //Update the coloring before next loop
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();


        firstUncolored = thrust::find(thrust::device, firstUncolored, coloring_device_begin + numNodes, 0);
    }

    cudaStatus = cudaFree(temp_coloring);                                               cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(forbiddenColors);                                             cudaCheck(cudaStatus, __FILE__, __LINE__);

    cudaStatus = cudaMemcpy(coloring_host.get(), coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    std::set<uint32_t> color_set(coloring_host.get(), coloring_host.get() + numNodes);
    numColors = color_set.size();
    convert_to_standard_notation();
}

template<typename nodeW, typename edgeW>
__global__ void ColoringGreedyFF_k::tentative_coloring(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs, uint32_t* forbidden_colors, const uint32_t maxColors){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    //If idx doesn't correspond to a node, it is excluded from this computation
    if(idx >= numNodes){     
        return;
    }
    //If the idx-th node has already been colored, it is excluded from this computation
    if(input_coloring[idx] != 0){
        return;
    }
    
    uint32_t* idx_forbidden_colors = forbidden_colors + idx * maxColors; 
    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    for(uint32_t j = 0; j < numNeighs; ++j){
        neighbor = neighs[j];
        idx_forbidden_colors[input_coloring[neighbor]] = idx;
    }
    
    for(uint32_t i = 1; i < maxColors; ++i){
        if(idx_forbidden_colors[i] != idx){
            output_coloring[idx] = i;
            break;
        }
    }
}

template<typename nodeW, typename edgeW>
__global__ void ColoringGreedyFF_k::conflict_detection(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    //If idx doesn't correspond to a node, it is excluded from this computation
    if(idx >= numNodes){
        return;
    }

    //If the idx-th node has no color, it is excluded from this computation
    //  NOTE: this should not happen
    if(input_coloring[idx] == 0){
        return;
    }

    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    //   If the idx-th node has the same color of a neighbor 
    //  and its id is greater than the one of its neighbor
    //  we make it uncolored
    for(uint32_t j = 0; j < numNeighs; ++j){
        neighbor = neighs[neighsOffset + j];
        if(input_coloring[idx] == input_coloring[neighbor] && idx > neighbor){
            output_coloring[idx] = 0;
            break;
        }
    }
}

//  Note that input_coloring and output_coloring must be pointers to GPU memory
__global__ void ColoringGreedyFF_k::update_coloring_GPU(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx >= numNodes){
        return;
    }

    output_coloring[idx] = input_coloring[idx];
}

//   This method is implemented without testing coloring correctness
//  by translating what was already done in coloringLuby.cu and without
//  filling a Coloring on device
template<typename nodeW, typename edgeW>
void ColoringGreedyFF<nodeW, edgeW>::convert_to_standard_notation(){

    //   Since we already use 0 as an "uncolored" identifier, there's no need to do
    //  what coloringLuby.cu does on lines colClass (lines 93-to-103, see NB on line 95) 
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
    this->coloring->colClass = coloring_host;
    this->coloring->cumulSize = cumulColorClassesSize;
}
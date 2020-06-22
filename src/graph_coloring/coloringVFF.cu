#include <coloring.h>
#include <coloringVFF.h>
#include <graph/graph.h>

#include "GPUutils/GPUutils.h"          //is required in order to use cudaCheck          
#include <thrust/count.h>               //is used to count colors in the coloring
#include <thrust/scan.h>                //is used for inclusive scan during the balancing loop
#include <thrust/copy.h>                //is used for updating the array of unbalanced nodes
#include <thrust/execution_policy.h>    //specifies where the data are and which policy some thrust functions should use 

//  These may be used for thrust::any_of, in case you'd want to switch 
// to using more thrust functions in the balancing loop:
//      #include <thrust/logical.h>
//      #include <thrust/functional.h>

#define BIN_SIZE(cumulBinSize, binIndex) (cumulBinSize[binIndex] - cumulBinSize[binIndex-1])
#define UNBALANCED_HISTORY 4
#define DEBUGGING

//  Constructor - initialization as in ColoringGreedyFF
template<typename nodeW, typename edgeW>
ColoringVFF<nodeW, edgeW>::ColoringVFF(Graph<nodeW, edgeW>* graph_d) : 
    Colorer<nodeW, edgeW>(graph_d), graphStruct_device(graph_d->getStruct()),
    numNodes(graph_d->getStruct()->nNodes), numColors(0) {

    //  We need to have an array representing the colors of each node
    // both on host...
    coloring_host = std::unique_ptr<uint32_t[]>(new uint32_t[numNodes]);
    
    // ...and on device
    cudaStatus = cudaMalloc((void**)&coloring_device, numNodes * sizeof(uint32_t));                     cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  We are assuming getMaxNodeDeg() returns a valid result here
    maxColors = this->graph->getMaxNodeDeg() + 1;   

    //  Data structures initialization
    cudaStatus = cudaMemset(coloring_device, 0, numNodes * sizeof(uint32_t));                           cudaCheck(cudaStatus, __FILE__, __LINE__); 
    cudaStatus = cudaMalloc((void**)&temp_coloring, numNodes * sizeof(uint32_t));                       cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * maxColors * sizeof(uint32_t));         cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMalloc((void**)&uncolored_nodes_device, sizeof(bool));                             cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  Array of bools corresponding to each node; used in balancing
    // If a node is flagged, it is in the set of nodes of the unbalanced bins
    cudaStatus = cudaMalloc((void**)&unbalanced_nodes, sizeof(bool) * numNodes * UNBALANCED_HISTORY);   cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMalloc((void**)&unbalanced_d, sizeof(bool));                                       cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  Note that binCumulSizes_device is initialized later since it needs 
    // results from coloring to be allocated; it gets freed in the destructor.

    threadsPerBlock = dim3(128, 1, 1);
    blocksPerGrid = dim3((numNodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);
}

//  Destructor
template<typename nodeW, typename edgeW>
ColoringVFF<nodeW, edgeW>::~ColoringVFF(){
    cudaStatus = cudaFree(coloring_device);     cudaCheck(cudaStatus, __FILE__, __LINE__);

    cudaStatus = cudaFree(temp_coloring);                                                               cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(forbiddenColors);                                                             cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(uncolored_nodes_device);                                                      cudaCheck(cudaStatus, __FILE__, __LINE__);

    cudaStatus = cudaFree(unbalanced_nodes);                                                            cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(binCumulSizes_device);                                                        cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(unbalanced_d);                                                                cudaCheck(cudaStatus, __FILE__, __LINE__);

    if(this->coloring != nullptr)
        free(this->coloring);
}

//  Entry point of the coloring + balancing; main should call this
template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run(){
    run_coloring();

    convert_to_standard_notation();

    run_balancing();
} 

//  Using run() implementation from ColoringGreedyFF for coloring
template <typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run_coloring(){
    bool uncolored_nodes = true;
    while(uncolored_nodes){
        //  Tentative coloring on the whole graph, in parallel
        ColoringGreedyFF_k::tentative_coloring<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs, forbiddenColors, maxColors);
        cudaDeviceSynchronize();

        //  Update the coloring now that we are sure there is no conflict
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();

        //  Checking for conflicts in colors and letting lower-id nodes win over the others, in parallel
        ColoringGreedyFF_k::conflict_detection<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs);
        cudaDeviceSynchronize();

        //  Update the coloring before next loop
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();
        
        //  Set <uncolored_nodes_device> to false and update it with <check_uncolored_nodes>
        cudaStatus = cudaMemset(uncolored_nodes_device, 0, sizeof(bool));                                           cudaCheck(cudaStatus, __FILE__, __LINE__);
        ColoringGreedyFF_k::check_uncolored_nodes<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, uncolored_nodes_device);
        cudaDeviceSynchronize();

        //  Note: we need to bring the value on host memory for the while loop
        cudaStatus = cudaMemcpy(&uncolored_nodes, uncolored_nodes_device, sizeof(bool), cudaMemcpyDeviceToHost);    cudaCheck(cudaStatus, __FILE__, __LINE__);
    }

    cudaStatus = cudaMemcpy(coloring_host.get(), coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    std::set<uint32_t> color_set(coloring_host.get(), coloring_host.get() + numNodes);
    numColors = color_set.size();

    //  Note we don't call convert_to_standard_notation() here, so that there's a separation of concerns
}

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::run_balancing(){

    //  Calculate gamma
    const uint32_t gamma_threshold = numNodes / numColors;

    cudaStatus = cudaMemset(unbalanced_nodes, 0, sizeof(bool) * numNodes * UNBALANCED_HISTORY);          // Note: it is initialized to false
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    //  Coloring array, temporary; used for parallelization and avoiding data races
    // Note memory was allocated in constructor, so it has to be reset now
    cudaStatus = cudaMemcpy(temp_coloring, coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToDevice);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

     
    //  Matrix of colors per array (from 0 to numColors); a pointer to the first element                //  Personal note on performance: the paper that inspired this uses an array that is
    // is passed to <tentative_rebalancing> kernel, and each thread will dereference only               // private to each node, and it considers it as an unsigned int array; I implemented it
    // a part of it (numColors + 1 elements per node).                                                  // as the paper did, but better performances and a lower memory impact could be achieved
                                                                                                        // with a matrix of boolean values.
    
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * (numColors+1) * sizeof(uint32_t));     // numColors+1 since we need to consider the 0 
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  Array of cumulative sizes of the color classes/bins
    // Note that it is initialized to the value obtained from the coloring in the previous step
    cudaStatus = cudaMalloc((void**)&binCumulSizes_device, sizeof(uint32_t) * (numColors + 1));
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMemcpy(binCumulSizes_device, this->coloring->cumulSize, sizeof(uint32_t) * (numColors + 1), cudaMemcpyHostToDevice);
    cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  We use a non-blocking CUDA Stream in order to make bin updating  
    // and conflict checking parallel
    cudaStream_t non_default_stream;
    cudaStreamCreateWithFlags(&non_default_stream, cudaStreamNonBlocking);

    //  Detection of the set of unbalanced nodes, by checking which bin sizes are greater than gamma
    BalancingVFF_k::detect_unbalanced_nodes<<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, binCumulSizes_device, gamma_threshold, unbalanced_nodes);
    
    //  Define and initialize a bool that will allow to loop until rebalancing is completed
    bool unbalanced;
    cudaStatus = cudaMemset(unbalanced_d, 0, sizeof(bool));                                     cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  Rebalancing starts here, properly.
    //  Computing <unbalanced_nodes> in order to check if there is even one node that is still considered unbalanced
    BalancingVFF_k::is_unbalanced<<<blocksPerGrid, threadsPerBlock>>>(numNodes, unbalanced_nodes, unbalanced_d);
    cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(&unbalanced, unbalanced_d, sizeof(bool), cudaMemcpyDeviceToHost);   cudaCheck(cudaStatus, __FILE__, __LINE__);

    #ifdef DEBUGGING
    std::cout << "\n---  DEBUGGING  ---";
    std::cout << "\nThe graph is: " << (unbalanced ? "unbalanced" : "balanced");
    std::cout << "\nNodes that are unbalanced: " << thrust::count_if(thrust::device, unbalanced_nodes, unbalanced_nodes + numNodes, thrust::identity<bool>()) << "\n";
    std::cout << "\nSizes of the bins are:\n";
    for(int i = 1; i <= numColors; ++i){
        std::cout << *(this->coloring->cumulSize + i) - *(this->coloring->cumulSize + i - 1) << "\t";
    }
    std::cout << "\nGamma is " << gamma_threshold << "\n";
    #endif
    
    //  Until there is even one of the nodes flagged as unbalanced...
    while(unbalanced){
        cudaStatus = cudaMemset(forbiddenColors, 0, numNodes * (numColors + 1) * sizeof(uint32_t));     cudaCheck(cudaStatus, __FILE__, __LINE__);
        cudaDeviceSynchronize();

        //  Tentative rebalancing on the <unbalanced_nodes> subset of nodes, in parallel
        BalancingVFF_k::tentative_rebalancing<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, numColors, coloring_device, binCumulSizes_device, graphStruct_device->neighs, graphStruct_device->cumulDegs, gamma_threshold, temp_coloring, unbalanced_nodes, forbiddenColors);
        cudaDeviceSynchronize();

        //  Bin update on a non-default, non-blocking CUDA stream
        BalancingVFF_k::update_bins<<<blocksPerGrid, threadsPerBlock, 0, non_default_stream>>>(numNodes, numColors, temp_coloring, binCumulSizes_device);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);
        //  ...while we solve conflicts on the default stream
        BalancingVFF_k::solve_conflicts<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, graphStruct_device->neighs, graphStruct_device->cumulDegs, unbalanced_nodes);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);
        cudaStreamSynchronize(non_default_stream);

        //  When the bin update is done, <binCumulSizes_device> contains the size of the i-th  
        // color class at index i; we apply an inclusive scan to get a cumulative array of sizes
        thrust::inclusive_scan(thrust::device, binCumulSizes_device, binCumulSizes_device + (numColors + 1), binCumulSizes_device);
        cudaDeviceSynchronize();

        //  Copying <temp_coloring> in <coloring_device>
        ColoringGreedyFF_k::update_coloring_GPU<<<blocksPerGrid, threadsPerBlock>>>(numNodes, temp_coloring, coloring_device);
        cudaDeviceSynchronize();

        //  Checking if there are more nodes flagged as unbalanced
        cudaStatus = cudaMemset(unbalanced_d, 0, sizeof(bool));                                         cudaCheck(cudaStatus, __FILE__, __LINE__);
        cudaDeviceSynchronize();
        BalancingVFF_k::is_unbalanced<<<blocksPerGrid, threadsPerBlock>>>(numNodes, unbalanced_nodes, unbalanced_d);
        cudaDeviceSynchronize();

        for(uint32_t i = (UNBALANCED_HISTORY - 1); i > 0; --i){
            thrust::copy_n(thrust::device, unbalanced_nodes + (i-1) * numNodes, numNodes, unbalanced_nodes + i * numNodes);
            cudaDeviceSynchronize();
        }

        BalancingVFF_k::ensure_not_looping<<<1, 1>>>(numNodes, unbalanced_nodes, UNBALANCED_HISTORY, unbalanced_d);
        cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(&unbalanced, unbalanced_d, sizeof(bool), cudaMemcpyDeviceToHost);       cudaCheck(cudaStatus, __FILE__, __LINE__);
    }

    cudaStatus = cudaMemcpy(coloring_host.get(), coloring_device, sizeof(uint32_t) * numNodes, cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    //  Update coloring data
    cudaStatus = cudaMemcpy(this->coloring->cumulSize, binCumulSizes_device, sizeof(uint32_t) * (numColors + 1), cudaMemcpyDeviceToHost);
    cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    this->coloring->colClass = coloring_host.get();         
    //  Note that this->coloring->nCol shouldn't be updated 
    // since the number of used colors should stay the same

    #ifdef DEBUGGING
    std::cout << "\nNew sizes of the bins are:\n";
    for(int i = 1; i <= numColors; ++i){
        std::cout << *(this->coloring->cumulSize + i) - *(this->coloring->cumulSize + i - 1) << "\t";
    }
    std::cout << "\n---END DEBUGGING---\n";
    #endif  
}

////////////////////////////// BASE CLASSES FUNCTIONS //////////////////////////////

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::convert_to_standard_notation(){
    uint32_t *cumulColorClassesSize = new uint32_t[numColors + 1];
    memset(cumulColorClassesSize, 0, (numColors+1)*sizeof(uint32_t));

    //  Count how many nodes are colored with <col> and store them in the array, before...
    for(uint32_t col = 1; col < numColors + 1; ++col){
        cumulColorClassesSize[col] = std::count(coloring_host.get(), coloring_host.get() + numNodes, col);
    }
    
    // ... you accumulate them in place
    // NOTE: index 0 is skipped, and we can start from 2 because cumulColorClassesSize[0] = 0
    // and cumulColorClassesSize[1] = cumulColorClassesSize[1] - cumulColorClassesSize[0] 
    for(uint32_t col = 2; col < numColors + 1; ++col){
        cumulColorClassesSize[col] = cumulColorClassesSize[col] + cumulColorClassesSize[col-1];
    }

    this->coloring = new Coloring();
    this->coloring->nCol = numColors;
    this->coloring->colClass = coloring_host.get();
    this->coloring->cumulSize = cumulColorClassesSize;
}

//// Simple getter
template<typename nodeW, typename edgeW>
Coloring* ColoringVFF<nodeW, edgeW>::getColoring(){
    return this->coloring;
}

//  Kernel that finds a subset of V of G(V, E) that is unbalanced
__global__ void BalancingVFF_k::detect_unbalanced_nodes(const uint32_t numNodes, const uint32_t* coloring_device, const uint32_t* cumulBinSizes, const uint32_t gamma, bool* nodes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }

    uint32_t idx_color = coloring_device[idx];
    //  <gamma> is lower than the number of nodes in the color class of idx_color
    if(gamma < BIN_SIZE(cumulBinSizes, idx_color)){
        nodes[idx] = true;  //node idx gets flagged as unbalanced
    }
}

//  Kernel searching for unbalanced nodes in a bool array; returns
// the result of the searching in the out-variable <result> 
__global__ void BalancingVFF_k::is_unbalanced(const uint32_t numNodes, const bool* unbalanced_nodes, bool* result){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }

    if(unbalanced_nodes[idx]){
        *result = true;
    }
}

//////////////////////////////   REBALANCING   //////////////////////////////
//  Kernel that tries to rebalance the coloring passed as input
template<typename nodeW, typename edgeW>
__global__ void BalancingVFF_k::tentative_rebalancing(const uint32_t numNodes, const uint32_t numColors, const uint32_t* input_coloring, const uint32_t* cumulBinSizes, const node* const neighs, const node_sz* const cumulDegs, const uint32_t gamma, uint32_t* output_coloring, bool* unbalanced_nodes, uint32_t* forbidden_colors){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }

    //  Nodes that are not flagged as unbalanced can be excluded from the computation
    if(unbalanced_nodes[idx] == false){
        return;
    }

    //  Calculate offsets for indexing <neighs> and <forbidden_colors>
    uint32_t* idx_forbidden_colors = forbidden_colors + idx * (numColors+1);
    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    //  Note: this was not [explicitly] included in the algorithm, but
    // it is needed; otherwise, the coloring will stay the same
    idx_forbidden_colors[input_coloring[idx]] = idx;

    //  Each color used by the neighbors is flagged as forbidden
    for(uint32_t j = 0; j < numNeighs; ++j){
        neighbor = neighs[neighsOffset + j];
        idx_forbidden_colors[input_coloring[neighbor]] = idx;
    }
    
    //  Pick the lowest (FF) color that is: 
    // - permissible [read: not forbidden] 
    // - belongs to an undersized bin
    for(uint32_t i = 1; i <= numColors; ++i){
        if(idx_forbidden_colors[i] != idx && gamma < BIN_SIZE(cumulBinSizes, i)){
            output_coloring[idx] = i;
            return;            
        }
    }
}

//  Kernel for updating bins
//  Note that this kernel works per-color; idx stands for the corresponding color, 
// and goes from 1 to numColors, both included
__global__ void BalancingVFF_k::update_bins(const uint32_t numNodes, const uint32_t numColors, const uint32_t* coloring, uint32_t* binSizes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    //  Also note that "color 0" is "invalid" (counting from 1)
    if(idx > numColors || idx == 0){        
        return;
    }

    binSizes[idx] = 0;
    for(uint32_t i = 0; i < numNodes; ++i){
        if(coloring[i] == idx){
            binSizes[idx] = binSizes[idx] + 1;
        }
    }
}

//  Kernel for solving conflicts created by <tentative_rebalancing>
template<typename nodeW, typename edgeW>
__global__ void BalancingVFF_k::solve_conflicts(const uint32_t numNodes, const uint32_t* coloring, const node* neighs, const node_sz* cumulDegs, bool* unbalanced_nodes){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= numNodes){
        return;
    }
    //  Nodes that are not flagged as unbalanced can be excluded from the computation
    if(unbalanced_nodes[idx] == false){
        return;
    }

    //  Calculate offsets for indexing <neighs> and <forbidden_colors>
    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    //  If there is a conflict with a neighbor and idx is greater than
    // the neighbor index, node idx stays in the unbalanced set... 
    for(uint32_t i = 0; i < numNeighs; ++i){
        neighbor = neighs[neighsOffset + i];
        if(coloring[neighbor] == coloring[idx] && idx > neighbor){
            return;
        }
    }

    // ... otherwise no conflicts are there and it can be removed from the set
    unbalanced_nodes[idx] = false;
}

//  Note: parallel reduction may be implemented here - but it could be overkill,
// since numColors is usually a "small number".
// __global__ void BalancingVFF_k::cumulate_bins(const uint32_t numColors, uint32_t* binCumulSizes){
//     for(uint32_t i = 1; i <= numColors; ++i){
//         binCumulSizes[i] = binCumulSizes[i] + binCumulSizes[i-1];
//     }
// }

__global__ void BalancingVFF_k::ensure_not_looping(const uint32_t numNodes, bool* unbalanced_nodes, uint32_t store_dim, bool* output){
    if(*output == false){
        return;
    }
    
    if(threadIdx.x > 0 || blockIdx.x > 0){
        return;
    }

    for(uint32_t node = 0; node < numNodes; ++node){
        for(uint32_t offset = 1; offset < store_dim; ++offset){
            if(unbalanced_nodes[node] != unbalanced_nodes[node + numNodes * offset]){
                return;
            }
        }
    }
    
    #ifdef DEBUGGING
    printf("WARNING >>> It was looping");
    #endif
    *output = false;
}


//////////////////////////////     LOGGING     //////////////////////////////
template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::saveStats(size_t iteration, float duration, std::ofstream &file){
    file << "Greedy FF Colorer followed by Vertex First Fit Rebalancing - GPU implementation - Report\n";
    file << "-------------------------------------------\n";
    file << "GRAPH INFO\n";
    file << "Nodes: " << numNodes << " - Edges: " << graphStruct_device->nEdges << "\n";
    file << "Max deg: " << this->graph->getMaxNodeDeg() << " - Min deg: " << this->graph->getMinNodeDeg() 
         << " - Avg deg: " << this->graph->getMeanNodeDeg() << "\n";
    file << "Edge Probability (for randomly generated graphs): " << this->graph->prob << "\n";
    file << "-------------------------------------------\n";
    file << "EXECUTION INFO\n";
    file << "Repetition: " << iteration << "\n";
    file << "Execution time: " << duration << "\n";
    file << "-------------------------------------------\n";
    file << "Number of colors: " << numColors << "\n";
    file << "Color histogram: \n";

    std::vector<uint32_t> histogram(numColors, 0);
    uint32_t* cumul = this->coloring->cumulSize;                //  We use the structure that is available at this point of the execution
    for(uint32_t i = 1; i < numColors + 1; ++i){                //  std::adjacent_difference but with printing
        histogram[i-1] = cumul[i] - cumul[i-1];
        file << i << "\t: " << histogram[i-1] << "\n";
    }
    float mean = std::accumulate(std::begin(histogram), std::end(histogram), 0) / static_cast<float>(numColors);
    float variance = 0;
    auto add_quadratic_distance_to_mean = [&](uint32_t value){
        variance += (value - mean) * (value - mean);
    };
    std::for_each(std::begin(histogram), std::end(histogram), add_quadratic_distance_to_mean);
    variance /= static_cast<float>(numColors);
    float std = sqrtf(variance);

    file << "Average number of nodes for each color: " << mean << "\n";
    file << "Variance: " << variance << "\n";
    file << "StD: " << std << "\n";
}

template<typename nodeW, typename edgeW>
void ColoringVFF<nodeW, edgeW>::saveColor(std::ofstream &file){
    uint32_t* coloring = this->coloring->colClass;
    for(uint32_t i = 0; i < numNodes; ++i){
        file << i << " " << coloring[i] << "\n";
    }
}

template class ColoringVFF<float, float>;
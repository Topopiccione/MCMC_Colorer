#include <coloring.h>
#include <coloringGreedyFF.h>
#include <graph/graph.h>

#include "GPUutils/GPUutils.h"          //is required in order to use cudaCheck

#include <set>                          //is used for counting the number of colors
#include <algorithm>                    //count is used for conversion to standard notation
#include <numeric>                      //transform_reduce is used for stats

//#define TESTCOLORINGCORRECTNESS

//  Constructor implementation
template<typename nodeW, typename edgeW>
ColoringGreedyFF<nodeW, edgeW>::ColoringGreedyFF(Graph<nodeW, edgeW>* graph_d) :
    Colorer<nodeW, edgeW>(graph_d), graphStruct_device(graph_d->getStruct()),
    numNodes(graph_d->getStruct()->nNodes), numColors(0) {
    
    //  We need to have an array representing the colors of each node
    // both on host...
    coloring_host = std::unique_ptr<uint32_t[]>(new uint32_t[numNodes]);
    
    // ...and on device
    cudaStatus = cudaMalloc((void**)&coloring_device, numNodes * sizeof(uint32_t));     cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    //  We are assuming getMaxNodeDeg() returns a valid result here
    maxColors = this->graph->getMaxNodeDeg() + 1;   

    //  Data structures initialization
    cudaStatus = cudaMemset(coloring_device, 0, numNodes * sizeof(uint32_t));                       cudaCheck(cudaStatus, __FILE__, __LINE__); 
    cudaStatus = cudaMalloc((void**)&temp_coloring, numNodes * sizeof(uint32_t));                   cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMalloc((void**)&forbiddenColors, numNodes * maxColors * sizeof(uint32_t));     cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaMalloc((void**)&uncolored_nodes_device, sizeof(bool));                         cudaCheck(cudaStatus, __FILE__, __LINE__);

    //  We setup our grid to be divided in blocks of 128 threads 
    threadsPerBlock = dim3(128, 1, 1);
    blocksPerGrid = dim3((numNodes + threadsPerBlock.x - 1)/threadsPerBlock.x, 1, 1);
}

//  Destructor implementation
template<typename nodeW, typename edgeW>
ColoringGreedyFF<nodeW, edgeW>::~ColoringGreedyFF(){
    //  We only need to deallocate what we allocated in the constructor
    cudaStatus = cudaFree(coloring_device);                                             cudaCheck(cudaStatus, __FILE__, __LINE__);
    
    cudaStatus = cudaFree(temp_coloring);                                               cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(forbiddenColors);                                             cudaCheck(cudaStatus, __FILE__, __LINE__);
    cudaStatus = cudaFree(uncolored_nodes_device);                                      cudaCheck(cudaStatus, __FILE__, __LINE__);

    if(this->coloring != nullptr)                 //Note: this may be unnecessary
        free(this->coloring);
}

template<typename nodeW, typename edgeW>
void ColoringGreedyFF<nodeW, edgeW>::run(){
    bool uncolored_nodes = true;
    while(uncolored_nodes){
        //  Tentative coloring on the whole graph, in parallel
        ColoringGreedyFF_k::tentative_coloring<nodeW, edgeW><<<blocksPerGrid, threadsPerBlock>>>(numNodes, coloring_device, temp_coloring, graphStruct_device->cumulDegs, graphStruct_device->neighs, forbiddenColors, maxColors);
        cudaDeviceSynchronize();

        //  Update the coloring now that we are sure there is no writing conflict
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
    convert_to_standard_notation();
}


//  Kernel that tries to find a color for each node; conflicts are tolerated in this stage.
template<typename nodeW, typename edgeW>
__global__ void ColoringGreedyFF_k::tentative_coloring(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs, uint32_t* forbidden_colors, const uint32_t maxColors){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    //  If idx doesn't correspond to a node, it is excluded from this computation
    if(idx >= numNodes){     
        return;
    }

    //  Line "idx_forbidden_colors[i] != idx", later in the code, didn't allow idx = 0 
    // to get a color different from 0; this resulted in endless looping as a bug.
    // Since this algorithm is a First Fit, node 0 can be put to 1 asap since it would win anyway.
    if(idx == 0){                   
        output_coloring[idx] = 1;
    }

    //  If the idx-th node has already been colored, it is excluded from this computation
    if(input_coloring[idx] != 0){
        return;
    }
    
    uint32_t* idx_forbidden_colors = forbidden_colors + idx * maxColors; 
    uint32_t numNeighs = cumulDegs[idx+1] - cumulDegs[idx];
    uint32_t neighsOffset = cumulDegs[idx];
    uint32_t neighbor;

    //  Note that <idx_forbidden_colors> is an array with size equal to <maxColors>.
    // Colors index this array; we flag it with idx if a color is forbidden to idx.
    //  Personal note: a first analysis makes me think that this array may be a bool array,
    // so that some memory could be saved.
    for(uint32_t j = 0; j < numNeighs; ++j){
        neighbor = neighs[neighsOffset + j];
        idx_forbidden_colors[input_coloring[neighbor]] = idx;
    }

    for(uint32_t i = 1; i < maxColors; ++i){
        if(idx_forbidden_colors[i] != idx){
            output_coloring[idx] = i;
            break;
        }
    }
}

//  Kernel that checks if <tentative_coloring> produced conflicts;
// if so, it is invalidated by putting its color to 0 in the <output_coloring>;
// otherwise, <output_coloring> stays as it was, updated to <input_coloring>
template<typename nodeW, typename edgeW>
__global__ void ColoringGreedyFF_k::conflict_detection(const uint32_t numNodes, const uint32_t* input_coloring, uint32_t* output_coloring, const node_sz * const cumulDegs, const node * const neighs){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    //  If idx doesn't correspond to a node, it is excluded from this computation
    if(idx >= numNodes){
        return;
    }

    //  If the idx-th node has no color, it is excluded from this computation
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

//  Kernel that searches nodes flagge as uncolored; if even one of them is there,
// the bool pointed by <uncolored_nodes> is set to true.
// Used as a way to update the state of the loop of the algorithm.
__global__ void ColoringGreedyFF_k::check_uncolored_nodes(const uint32_t numNodes, const uint32_t* coloring, bool* uncolored_nodes){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= numNodes){
        return;
    }

    if(coloring[idx] == 0){
        *uncolored_nodes = true;
    }
}

//  This method is implemented by translating what was already done in coloringLuby.cu
// and without filling a Coloring on device
template<typename nodeW, typename edgeW>
void ColoringGreedyFF<nodeW, edgeW>::convert_to_standard_notation(){

    //  Since we already use 0 as an "uncolored" identifier, there's no need to do
    // what coloringLuby.cu does on lines colClass (lines 93-to-103, see NB on line 95) 
    uint32_t *cumulColorClassesSize = new uint32_t[numColors + 1];
    memset(cumulColorClassesSize, 0, (numColors+1)*sizeof(uint32_t));

    //  Count how many nodes are colored with <col> and store them in the array, before...
    for(uint32_t col = 1; col < numColors + 1; ++col){
        cumulColorClassesSize[col] = std::count(coloring_host.get(), coloring_host.get() + numNodes, col);
    }

    #ifdef TESTCOLORINGCORRECTNESS
    std::cout << "Color Classes:\n";
    for(uint32_t col = 0; col < numColors + 1; ++col){
        std::cout << cumulColorClassesSize[col] << " ";
    }
    std::cout << "\n";
    #endif
    
    // ... you accumulate them in place
    // NOTE: index 0 is skipped, and we can start from 2 because cumulColorClassesSize[0] = 0
    for(uint32_t col = 2; col < numColors + 1; ++col){
        cumulColorClassesSize[col] = cumulColorClassesSize[col] + cumulColorClassesSize[col-1];
    }
    
    //DEBUG CORRECTNESS, BRUTE FORCE
    #ifdef TESTCOLORINGCORRECTNESS
    std::cout << "Cumulative Sizes of Color Classes:\n";
    for(uint32_t col = 0; col < numColors + 1; ++col){
        std::cout << cumulColorClassesSize[col] << " ";
    }
    std::cout << "\n";

	std::cout << "Test colorazione attivato!\n";
    
    uint32_t* test_coloring = coloring_host.get();
    std::unique_ptr<node_sz[]> cumulDegs( new node_sz[graphStruct_device->nNodes + 1]);
	std::unique_ptr<node[]>  neighs( new node[graphStruct_device->nEdges] );
	cudaStatus = cudaMemcpy( cumulDegs.get(), graphStruct_device->cumulDegs, (graphStruct_device->nNodes + 1) * sizeof(node_sz),    cudaMemcpyDeviceToHost );   cudaCheck( cudaStatus, __FILE__, __LINE__ );
	cudaStatus = cudaMemcpy( neighs.get(),    graphStruct_device->neighs,    graphStruct_device->nEdges       * sizeof(node_sz),    cudaMemcpyDeviceToHost );   cudaCheck( cudaStatus, __FILE__, __LINE__ );
    
    uint32_t offset;
    uint32_t size;
    uint32_t neighbor;
    for(uint32_t i = 0; i < numNodes; ++i){
        size    = cumulDegs[i+1] - cumulDegs[i];
        offset  = cumulDegs[i];
        
        for(uint32_t j = 0; j < size; ++j){
            neighbor = neighs[offset + j];
            if(test_coloring[i] == test_coloring[neighbor]){
                std::cout << "NO! Il nodo " << i << " e il nodo " << neighbor << " sono vicini e colorati entrambi come " << test_coloring[i] << "\n";
                abort();
            }
        }
    }
    #endif
    //END DEBUG CORRECTNESS

    //  Set all the variables of the new Coloring
    this->coloring = new Coloring();
    this->coloring->nCol = numColors;
    this->coloring->colClass = coloring_host.get();
    this->coloring->cumulSize = cumulColorClassesSize;
}


//// Simple getter
template<typename nodeW, typename edgeW>
Coloring* ColoringGreedyFF<nodeW, edgeW>::getColoring(){
    return this->coloring;
}

template<typename nodeW, typename edgeW>
void ColoringGreedyFF<nodeW, edgeW>::saveStats(size_t iteration, float duration, std::ofstream &file){
    file << "Greedy First Fit Colorer - GPU implementation - Report\n";
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

    //  Personal note: C++14 doesn't have transform_reduce, I fear,
    // so the following cannot be compiled:
    // 
    // auto distance_to_mean = [&](uint32_t value){
    //     return (value - mean) * (value - mean);
    // };
    //  float variance = std::transform_reduce(std::begin(histogram), std::end(histogram), 0, std::plus<>, distance_to_mean); 

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
void ColoringGreedyFF<nodeW, edgeW>::saveColor(std::ofstream &file){
    uint32_t* coloring = this->coloring->colClass;
    for(uint32_t i = 0; i < numNodes; ++i){
        file << i << " " << coloring[i] << "\n";
    }
}

//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class ColoringGreedyFF<float, float>;
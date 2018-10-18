#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cinttypes>

#define CUDACHECK(call) \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
	    {                                                                      \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
	    }                                                                      \
}

// A quanto pare, non e' possibile passare le direttive del preprocessore __FILE__ e __LINE__ come
// argomento di una funzione; la soluzione e' di usare delle macro
#define cudaCheck(cuSts,file,line) {\
	if (cuSts != cudaSuccess) {\
		std::cout << "Cuda error in file " << file << " at line " << line << std::endl;\
		std::cout << "CUDA Report: " << cudaGetErrorString( cuSts ) << std::endl;\
		abort();\
	}\
}

#define curandCheck(cuSts,file,line) {\
	if (cuSts != CURAND_STATUS_SUCCESS) {\
		std::cout << "Cuda error in file " << file << " at line " << line << std::endl;\
		std::cout << "CuRand error code: " << cuSts << std::endl;\
		abort();\
	}\
}

struct GPUMemTracker {
public:
	static uint64_t				graphStructSize;
	static uint64_t				graphDegsSize;
	static uint64_t				graphNeighsSize;
	static uint64_t				graphNodeWSize;
	static uint64_t				graphEdgeWSize;
	static uint64_t				graphNodeTSize;

	static uint64_t				coloringColorsSize;
	static uint64_t				coloringColorBinsSize;
	static uint64_t				coloringColorHistSize;
	static uint64_t				coloringNconflictsSize;
	static uint64_t				coloringQjSize;

    static void printGraphReport() {
        std::cout << "GPU mem allocation report (graph)" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Struct: " << graphStructSize << std::endl;
        std::cout << "Degs:   " << graphDegsSize << std::endl;
        std::cout << "Neighs: " << graphNeighsSize << std::endl;
        std::cout << "Node W: " << graphNodeWSize << std::endl;
        std::cout << "Edge W: " << graphEdgeWSize << std::endl << std::endl;
		std::cout << "Node T: " << graphNodeTSize << std::endl << std::endl;

    }
    static void printColoringReport() {
        std::cout << "GPU mem allocation report (coloring)" << std::endl;
        std::cout << "------------------------------------" << std::endl;
        std::cout << "Colors array:      " << coloringColorsSize << std::endl;
        std::cout << "Color bins:        " << coloringColorBinsSize << std::endl;
        std::cout << "Color histogram:   " << coloringColorHistSize << std::endl;
        std::cout << "Numb of conflicts: " << coloringNconflictsSize << std::endl;
        std::cout << "Qj array:          " << coloringQjSize << std::endl << std::endl;
    }
};

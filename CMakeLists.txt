CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(MCMC_Colorer)

### Googletest
# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(test "Build all tests." OFF)

include_directories(.)
include_directories(GPUutils)
include_directories(graph)
include_directories(graph_coloring)
include_directories(utils)
include_directories(easylogging)

INCLUDE(FindCUDA)

FIND_PACKAGE(OpenMP REQUIRED) 		#Required?
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_52,code=sm_52;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_35,code=sm_35;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -default-stream per-thread;)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O3; -Xcompiler -fPIC;)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
## Debug! (potremmo mettere una variabile...)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O0; -G; -Xcompiler -rdynamic; -keep; -Xcompiler -fPIC; -Xcompiler -fopenmp;)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")

SET(CUDA_SEPARABLE_COMPILATION TRUE) # -rdc true
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NON_PROPAGATED_HOST_FLAGS ON)

FIND_PACKAGE(CUDA REQUIRED QUIET)

add_subdirectory(easyloggingpp)

SET(SOURCESCPP      "graph/graphCPU.cpp"
					"graph_coloring/colorer.cpp"
					"utils/ArgHandle.cpp"
					"utils/fileImporter.cpp"
#					"utils/fileExporter.cpp"
					"utils/timer.cpp")
SET(SOURCESCUDA     "GPUutils/GPUCudaCleaner.cu"
                	"GPUutils/GPUStream.cu"
                	"GPUutils/GPURandomizer.cu"
					"graph/graphGPU.cu"
                	"graph_coloring/coloringLuby.cu"
					"graph_coloring/coloringLubyFast.cu"
					"testColoring_Benchmark.cu")
SET(SOURCEGEN		"datasetGenerator.cpp")
FILE(GLOB LOGGINGPP "easyloggingpp/easylogging++.cc")

CUDA_ADD_EXECUTABLE(MCMC_Colorer ${SOURCESCUDA} ${SOURCESCPP} ${LOGGINGPP})
add_executable(datasetGen ${SOURCEGEN})
add_dependencies(MCMC_Colorer easyloggingpp)

target_link_libraries(MCMC_Colorer    ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudadevrt.a" ${CUDA_curand_LIBRARY})

### Googletest
if (test)
	add_subdirectory(test)
	enable_testing()

	include_directories(${GTEST_INCLUDE_DIRS})
	link_directories(${GTEST_LIBS_DIR})

	add_executable(runUnitTests graph/graphCPU.cpp graph/graphGPU.cu test/fillTest.cpp)
	add_dependencies(runUnitTests googletest)
	target_link_libraries(runUnitTests gtest gtest_main)

	add_test(test1 runUnitTests)
endif()

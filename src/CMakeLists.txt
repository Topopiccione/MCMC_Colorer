CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(MCMC_Colorer)

# Options. Turn on with 'cmake -Dmyvarname=ON'.
option(test "Build all tests." OFF)								# Attiva GoogleTest
option(sm61 "Activate nvcc compiling for 6.1 capability" ON)	# dalab
option(sm60 "Activate nvcc compiling for 6.0 capability" OFF)
option(sm52 "Activate nvcc compiling for 5.2 capability" OFF)
option(sm50 "Activate nvcc compiling for 5.0 capability" OFF)	# Default sul mio pc
option(sm35 "Activate nvcc compiling for 3.5 capability" OFF)

add_subdirectory(easyloggingpp)

include_directories(.)
include_directories(GPUutils)
include_directories(graph)
include_directories(graph_coloring)
include_directories(utils)
include_directories(easyloggingpp)

include(FindCUDA)

find_package(OpenMP REQUIRED) 		#Required?
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
if (sm61)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61;)
endif()
if (sm60)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60;)
endif()
if (sm52)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_52,code=sm_52;)
endif()
if (sm50)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_50,code=sm_50;)
endif()
if (sm35)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_35,code=sm_35;)
endif()
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -default-stream per-thread;)

### Release
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O3; -Xcompiler -fPIC;)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
### Debug! (potremmo mettere una variabile...)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O0; -G; -Xcompiler -rdynamic; -keep; -Xcompiler -fPIC; -Xcompiler -fopenmp;)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")

SET(CUDA_SEPARABLE_COMPILATION TRUE) # -rdc true
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NON_PROPAGATED_HOST_FLAGS ON)

FIND_PACKAGE(CUDA REQUIRED QUIET)

SET(SOURCESCPP      "graph/graphCPU.cpp"
					"graph_coloring/colorer.cpp"
					"graph_coloring/coloringMCMC_CPU.cpp"
					"graph_coloring/coloringMCMC_CPUutils.cpp"
					"utils/ArgHandle.cpp"
					"utils/dbg.cpp"
					"utils/fileImporter.cpp"
					"utils/timer.cpp")
SET(SOURCESCUDA     "GPUutils/GPUCudaCleaner.cu"
                	"GPUutils/GPUStream.cu"
                	"GPUutils/GPURandomizer.cu"
					"graph/graphGPU.cu"
                	"graph_coloring/coloringLuby.cu"
					"graph_coloring/coloringLubyFast.cu"
					"graph_coloring/coloringMCMC_main.cu"
					"graph_coloring/coloringMCMC_standard.cu"
					"graph_coloring/coloringMCMC_decrease.cu"
					"graph_coloring/coloringMCMC_balance.cu"
					"graph_coloring/coloringMCMC_utils.cu"
					"testColoring_Benchmark.cu")
SET(SOURCEGEN		"datasetGenerator.cpp")
SET(LOGGINGPP		"easyloggingpp/easyloggingpp/easylogging++.cc")

cuda_add_executable(MCMC_Colorer ${SOURCESCUDA} ${SOURCESCPP} ${LOGGINGPP})
add_executable(datasetGen ${SOURCEGEN})
add_dependencies(MCMC_Colorer easyloggingpp)

target_link_libraries(MCMC_Colorer ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudadevrt.a" ${CUDA_curand_LIBRARY})

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

// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include "GPUCudaCleaner.h"

CudaCleaner::CudaCleaner() {}

CudaCleaner::~CudaCleaner() {
	std::cout << "calling cudaDeviceReset()..." << std::endl;
	cudaDeviceReset();
	std::cout << "Done." << std::endl;
}

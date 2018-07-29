// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <memory>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include "coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"

using namespace std;

namespace ColoringMCMC_k {
__global__ void initColoring(curandState*, expCumulatedDiscreteDistribution_t, col*, col_sz, node_sz);
__global__ void drawNewColoring(unsigned int*, curandState*, GraphStruct<col, col>*, col*, col*, col*);
__inline__ __host__ __device__ node_sz checkConflicts(node, node_sz, node*, col*);
__inline__  __device__ col newFreeColor(curandState*, node, node_sz, node*, col*, col*);
__inline__  __device__ col fixColor(curandState*, node*, col);
__device__ int discreteSampling(curandState *, cumulatedDiscreteDistribution_t);
}

__constant__ float LAMBDA;
__constant__ col_sz NCOLS;
__constant__ float EPSILON;
__constant__ float CUMSUMDIST;
__constant__ float RATIOFREEZED;

/**
 * Samples a discrete distribution
 * @param states curand state
 * @param x the element sampled in the range [0,n-1]
 * @param dist the probability mass function (not normalized)
 * @param n the distribution size
 */
__device__ int ColoringMCMC_k::discreteSampling(curandState* states,
		cumulatedDiscreteDistribution_t dist) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int n = dist->length;
	printf("n = %d\n", n);
	unsigned int l = 0, r = n - 1;
	float u = curand_uniform(&states[tid]);
	float bin = dist->prob[n - 1] * u;
	printf("bin = %f\n", bin);
	while (l < r) {
		unsigned int m = floorf((l + r) / 2);
		if (bin < dist->prob[m])  // RIFARE
			r = m;
		else
			l = m + 1;
	}
	return r;
}

__global__ void ColoringMCMC_k::initColoring(curandState* state,
		expCumulatedDiscreteDistribution_t dist, col* C, col_sz nCol,
		node_sz n) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) {
		cumulatedDiscreteDistribution_st d = dist->CDF;
		printf("length = %d\n", d.length);
//		C[i] = ColoringMCMC_k::discreteSampling(state, d);
		printf("init_col[%d] = %d\n", i, C[i]);
	}
}

/**
 * Counts the number of conflicts for a node with respect a coloring
 * @param nodeID the node
 * @param deg the node degree
 * @param neighs neighborhood
 * @param C the current coloring
 * @return the number of conflicts
 */
__inline__ __host__ __device__ node_sz ColoringMCMC_k::checkConflicts(
		node nodeID,
		node_sz deg,
		node* neighs,
		col* C) {

	node_sz nConf = 0;
	col nodeCol = C[nodeID];
	for (node_sz i = 0; i < deg; i++)
		if (C[neighs[i]] == nodeCol)
			nConf++;
	return nConf;
}

/**
 * Assigns a new color to a node drawn among the free colors
 * @param state curandState
 * @param deg
 * @param neigh
 * @param C1
 * @param param
 * @return
 */
__inline__  __device__ col ColoringMCMC_k::newFreeColor(
		curandState *state,
		node nodeID,
		node_sz deg,
		node* neighs,
		col* C,
		col* colNeighs) {   // colors of neighborhood

	// conflict analysis
	col_sz k = NCOLS;
	float lambda = LAMBDA;
	float epsilon = EPSILON;

	// find num unique neigh colors
//	unsigned l = 1;
	colNeighs[0] = C[neighs[0]];
	col_sz nColNotFree = 1;
	for (node j = 1; j < deg; j++) {
		col cc = C[neighs[j]];
		bool FOUND = 0;
		for (col i = 0; i < nColNotFree; i++)
			if (colNeighs[i] == cc) {
				FOUND = 1;
				break;
			}
		if (!FOUND) {
			colNeighs[nColNotFree] = cc;
			nColNotFree++;
		}
	}

//	printf("nodeID[%d] nColNotFree = %d\n",nodeID,nColNotFree);


	// bubble sort neigh nodes
	for (col c = 0; c < nColNotFree - 1; c++)
		for (col g = c + 1; g < nColNotFree; g++)
			if (colNeighs[c] > colNeighs[g]) {
				col tmp = colNeighs[g];
				colNeighs[g] = colNeighs[c];
				colNeighs[c] = tmp;
			}


//	if (nodeID == 9)
//	for (col c = 0; c < nColNotFree; c++)
//		printf("colNeighs[%d] = %d\n", c,colNeighs[c]);

//		printf("epsilon = %f\n", epsilon);

	// compute cumSum probability
	float cumProb = 0;
	col j = 0;
	for (col c = 1; c <= k; c++) {
		if ((j < nColNotFree) && (c == colNeighs[j])) {
			j++;
		}
		else {
			cumProb += expf(-lambda * (float)c);
		}
	}
	col_sz nColFree = k - nColNotFree;
	float a = (1.0f - epsilon)/nColFree;
	cumProb = cumProb*a + epsilon;

//	if (nodeID == 9)
//		printf("cumProb[%d] TOTALE= %f\n", nodeID, cumProb);
//
	// sample the discrete distribution
	float u = cumProb * curand_uniform(state);
	cumProb = 0;
	col newCol;
	j = 0;

	bool vicinato;
	for (col c = 1; c <= k; c++) {
		if ((j < nColNotFree) && (c == colNeighs[j])) {
			cumProb += epsilon/(float)nColNotFree;
			j++;
			vicinato = 1;
//			if (nodeID == 5)
//				printf("VICINO: Prob[%d][%d] = %f\n", nodeID,c, epsilon/(float)nColNotFree);
		} else {
			cumProb += a * __expf(-lambda * (float)c);
			vicinato = 0;
//				printf("NON VICINO: Prob[%d][%d] = %f\n", nodeID,c, a * __expf(-lambda * (float)c));
		}
//		printf("cumProb = %f -- u = %f -- vicinato = %d \n", cumProb,u,vicinato);

		// choose bin
		if (cumProb >= u) {
			printf("vicinato = %d\n", vicinato);
			newCol = c;
			break;
		}
	}
	return newCol;
}

/**
 * Assigns a new color to a node drawn among the free colors
 * @param state curandState
 * @param deg
 * @param neigh
 * @param C1
 * @param param
 * @return
 */
__inline__  __device__ col ColoringMCMC_k::fixColor(
		curandState *state,
		node* nodeNeighs,
		col nodeCol) {
	float lambda = LAMBDA;
	float epsilon = EPSILON;
	unsigned k = NCOLS;

	// sampling the discrete distribution
	float u = curand_uniform(state);
	col newCol;
	float cumProb = 1 - epsilon;
	if (u < cumProb)
		newCol = nodeCol;
	else {
		float P = __expf(-lambda)*(1.0-__expf(-lambda * (float)k)) / ((1.0-__expf(-lambda))) - __expf(-lambda * (float)nodeCol);
		for (col c = 1; c <= k; c++) {
			if (c != nodeCol)
				cumProb += epsilon * __expf(-lambda * (float)c) / P;
			if (u < cumProb) {
				newCol = c;
				break;
			}
		}
	}
	return newCol;
}

/**
 * Draws a new coloring based on the current coloring
 * @param states curand state
 * @param str graph structure
 * @param C1 current coloring
 * @param C2 new coloring
 * @param colSupport support for random generation
 * @param numConflicts overall conflicts
 * @param PHI support distribution
 */
__global__ void ColoringMCMC_k::drawNewColoring(
		unsigned int* numConflicts_d,
		curandState* states,
		GraphStruct<col, col>* str,
		col* C1_d,
		col* C2_d,
		col* colSupport) {

	node nodeID = threadIdx.x + blockDim.x * blockIdx.x;
	node_sz n = str->nNodes;
	unsigned int nConf;

	// even & odd
	bool EVEN_ODD = 0;
	if (*numConflicts_d) {
		EVEN_ODD = 1;
		*numConflicts_d = 0;
	}

	if (nodeID < n) {
		node_sz cumDeg = str->cumDegs[nodeID];
		node_sz deg = str->cumDegs[nodeID + 1] - cumDeg;
		//		printf("deg [%d] = %d\n",nodeID,deg);
		if (EVEN_ODD)
			nConf = ColoringMCMC_k::checkConflicts(nodeID, deg,	str->neighs + cumDeg, C1_d);
		else
			nConf = ColoringMCMC_k::checkConflicts(nodeID, deg,	str->neighs + cumDeg, C2_d);
		atomicAdd(numConflicts_d, nConf);
		printf("num conflicts[%d] = %d\n",nodeID,nConf);

		// CASE: conflicts
		if (nConf) {
			if (RATIOFREEZED > curand_uniform(&states[nodeID])) {
				if (EVEN_ODD)
					C2_d[nodeID] = newFreeColor(states + nodeID, nodeID, deg,	str->neighs + cumDeg, C1_d, colSupport + cumDeg);
				else
					C1_d[nodeID] = newFreeColor(states + nodeID, nodeID, deg,	str->neighs + cumDeg, C2_d, colSupport + cumDeg);
//								printf("new color C2[%d] = %d   -- deg = %d  -- cumDeg = %d\n",nodeID,C2[nodeID],deg,cumDeg);
			} else {
				if (EVEN_ODD)
					C2_d[nodeID] = C1_d[nodeID];
				else
					C1_d[nodeID] = C2_d[nodeID];
				//				printf("new color (NO conf) C2[%d] = %d\n",nodeID,C2[nodeID]);
			}

		// CASE: NO conflicts
		} else {
			if (EVEN_ODD)
				C2_d[nodeID] = fixColor(states + nodeID, str->neighs + cumDeg, C1_d[nodeID]);
			else
				C1_d[nodeID] = fixColor(states + nodeID, str->neighs + cumDeg, C2_d[nodeID]);
			//			printf("fix C2[%d] = %d\n",nodeID,C2[nodeID]);
		}
	}
}

void print_C(col* C, unsigned n) {

	for (int i= 0; i < n; i++) {
		cout << "C[" << i <<  "] = " << C[i] << endl;
	}
}

/**
 * Start the coloring on the graph
 */
void ColoringMCMCGPU::run() {
	GraphStruct<col, col>* str = graph->getStruct();
	node_sz n = str->nNodes;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError cuSts;

	int numThreads = 32;
	dim3 block(numThreads);
	dim3 grid((n + numThreads - 1) / numThreads);

	// init curand
	curandState* states;
	cuSts = cudaMalloc((void **) &states, n * sizeof(curandState));
	cudaCheck(cuSts, __FILE__, __LINE__);
	long seed = 0;
	GPURand_k::initCurand<<<grid, block>>>(states, seed, n);
	cudaDeviceSynchronize();

	// coloring
	col_sz nCol = graph->getMaxNodeDeg() + 1;
	col *C, *C1_d, *C2_d;
	C = (col*)malloc(n * sizeof(col));
	cuSts = cudaMalloc(&C1_d, n * sizeof(col)); cudaCheck(cuSts, __FILE__, __LINE__);
	cuSts = cudaMalloc(&C2_d, n * sizeof(col)); cudaCheck(cuSts, __FILE__, __LINE__);
//	cuSts = cudaMallocManaged(&C1, n * sizeof(col));
//	cudaCheck(cuSts, __FILE__, __LINE__);
//	cuSts = cudaMallocManaged(&C2, n * sizeof(col));
//	cudaCheck(cuSts, __FILE__, __LINE__);

	// neighborhood coloring
	col* colNeigh;
	cuSts = cudaMalloc(&colNeigh, str->cumDegs[n]*sizeof(col));
	cudaCheck(cuSts, __FILE__, __LINE__);

	// MCMC priori distribution parameters
	param.nCol = nCol;
	param.lambda = param.lambda/(float)nCol;
	expCumulatedDiscreteDistribution dist;
	dist.lambda = param.lambda;
	CPURand::createExpDistribution(&dist, dist.lambda, nCol); // form: exp(-lambda*c/nCol), c = 1,2...


	//	float* PHI;
//	cuSts = cudaMallocManaged(&PHI, nCol * sizeof(float)); cudaCheck(cuSts, __FILE__, __LINE__);

	// normalization factors used to sample a distribution on GPU
//	float a = (1 - param.epsilon * (nCol - 1));
//	for (unsigned i = 0; i < nCol; i++) {
//		PHI[i] = a * dist.CDF.prob[i];
//		for (unsigned k = 0; k < nCol; k++) {
//			if (k != i)
//				PHI[i] += param.epsilon * dist.CDF.prob[k];
//		}
//	}

	cout << "lambda = " << param.lambda << endl;
	// fill constant memory
	cudaMemcpyToSymbol(LAMBDA, &(param.lambda), sizeof(float));
	cudaMemcpyToSymbol(NCOLS, &(param.nCol), sizeof(col_sz));
	cudaMemcpyToSymbol(EPSILON, &(param.epsilon), sizeof(float));
	cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
	cudaMemcpyToSymbol(CUMSUMDIST, &(dist.CDF.prob[dist.CDF.length - 1]),
			sizeof(float));

	// init coloring CPU
	CPURand::discreteSampling(&(dist.CDF), C, n);
	print_C(C,n);

//	std::cout << "OKKIO "<< nCol <<  std::endl;
//	for (int i= 0; i < n; i++)
//		cout << "C[" << i <<  "] = " << C1[i] << endl;


	// manage conflicts
	unsigned int numConflictsOLD = 0;
	for (int i = 0; i < n; i++) {
		node_sz cumDeg = str->cumDegs[i];
		node_sz deg = str->cumDegs[i + 1] - cumDeg;
		unsigned int nc = ColoringMCMC_k::checkConflicts(i, deg, str->neighs + cumDeg, C);
		numConflictsOLD += nc;
	}
	cout << "numConflicts init = " << numConflictsOLD << endl;
	cuSts = cudaMemcpy(C1_d, C, n*sizeof(col), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);
	unsigned int* numConflicts_d;
	cuSts = cudaMalloc((void**)&numConflicts_d, sizeof(unsigned int)); cudaCheck(cuSts, __FILE__, __LINE__);
	unsigned int numConflicts = 1;

	// start coloring
	unsigned nRound = 0;
	bool STOP = 1;
	cudaEventRecord(start);
	while (STOP) {
		nRound++;
		cout << "ROUND # " << nRound << endl;
		cuSts = cudaMemcpy(numConflicts_d, &numConflicts, sizeof(unsigned int), cudaMemcpyHostToDevice); cudaCheck(cuSts, __FILE__, __LINE__);
		ColoringMCMC_k::drawNewColoring<<<grid, block>>>(numConflicts_d, states, str, C1_d, C2_d, colNeigh);
		cuSts = cudaMemcpy(&numConflicts, numConflicts_d, sizeof(unsigned int), cudaMemcpyDeviceToHost); cudaCheck(cuSts, __FILE__, __LINE__);
		cudaDeviceSynchronize();
		cout << "GPU num GLOBAL conflicts = " << numConflicts << endl;

		if (!numConflicts)
			STOP = 0;
		else {
			if (numConflicts > numConflictsOLD) {
				if (param.ratioFreezed > 0.45) {
					param.ratioFreezed = param.ratioFreezed-0.5;
					cout << "numConflicts = " << numConflicts << "    numConflictsOLD = " << numConflictsOLD << endl;
					cout << "freeze = " << param.ratioFreezed << endl;
					cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
				}
				else {
					param.lambda = param.lambda*0.9;
					param.ratioFreezed = 1;
					CPURand::createExpDistribution(&dist, dist.lambda, nCol); // form: exp(-lambda*c/nCol), c = 1,2...
					cudaMemcpyToSymbol(LAMBDA, &(param.lambda), sizeof(float));
					cudaMemcpyToSymbol(RATIOFREEZED, &(param.ratioFreezed), sizeof(float));
				}
				numConflictsOLD = numConflicts;
				if (nRound % 2)
					numConflicts = 0;
				else
					numConflicts = 1;
			}
		}
//		cin.ignore();
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	elapsedTimeSec = milliseconds/1000.0f;


	// build coloring
	buildColoring(C, n);
//		print_C(C1,n);


	cudaFree(states);
	cudaFree(C1_d);
	cudaFree(C2_d);
	cudaFree(colNeigh);
//	cudaFree(numConflicts);
//	cudaDeviceReset();

}

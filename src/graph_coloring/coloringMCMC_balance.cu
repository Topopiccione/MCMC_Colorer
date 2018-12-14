// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

#ifdef COLOR_BALANCE_ON_NODE_CUMULATIVE
__global__ void ColoringMCMC_k::selectStarColoringBalanceOnNode_cumulative(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, curandState * states, float partition, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to count how many times a color is used from the neighbors
	for (int i = 0; i < nneighs; i++) {
		colorsChecker[coloring_d[neighs[index + i]]]++;
	}

	if (colorsChecker[nodeCol] > 0) {
		float randnum = curand_uniform(&states[idx]);				//random number

		uint32_t Zp = 0;											//number of free colors (p) and occupied colors (n)
		for (int i = 0; i < nCol; i++)
		{
			//Zn += colorsChecker[i] != 0;
			if (colorsChecker[i] == 0)
				Zp++;
		}

		float threshold = 0;
		float q;
		int i = 0;
		do {
			q = (1 - ((float)colorsChecker[i] / (float)nneighs)) / ((float)nCol - 1);
			q /= partition;
			if (colorsChecker[i] == 0)
				q += ((partition - 1) / partition) / Zp;

			threshold += q;
			i++;
		} while (threshold < randnum);

		qStar_d[idx] = q;											//save the probability of the color chosen
		starColoring_d[idx] = i - 1;
	}
	else {
		qStar_d[idx] = (1 - ((float)colorsChecker[nodeCol] / (float)nneighs)) / ((float)nCol - 1);
		starColoring_d[idx] = nodeCol;
	}
}
#endif // COLOR_BALANCE_ON_NODE_CUMULATIVE
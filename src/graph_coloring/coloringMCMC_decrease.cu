// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

#if defined(COLOR_DECREASE_LINE_CUMULATIVE) || defined(COLOR_DECREASE_EXP_CUMULATIVE)
__global__ void ColoringMCMC_k::selectStarColoringDecrease_cumulative(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float * probDistribution_d, curandState * states, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set if a color is used from the neighbors
	for (int i = 0; i < nneighs; i++) {
		colorsChecker[coloring_d[neighs[index + i]]] = 1;
	}

	float reminder = 0;
	uint32_t Zn = 0, Zp = nCol;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
	{
		Zn += colorsChecker[i] != 0;
		reminder += (colorsChecker[i] != 0) * (probDistribution_d[i] - epsilon);
	}
	Zp -= Zn;
	reminder /= Zp;

	if (!Zp)													//manage exception of no free colors
	{
#ifdef FIXED_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // FIXED_N_COLORS
#ifdef DYNAMIC_N_COLORS
		starColoring_d[idx] = nodeCol;
		qStar_d[idx] = 1;
#endif // DYNAMIC_N_COLORS
		return;
	}

	int i = 0;
	float q;
	float threshold = 0;
	float randnum = curand_uniform(&states[idx]);				//random number
	if (colorsChecker[nodeCol])									//if node color is used by neighbors
	{
		do {
			q = (probDistribution_d[i] + reminder) * (!colorsChecker[i]) + (epsilon) * (colorsChecker[i]);
			threshold += q;
			i++;
		} while (threshold < randnum);
	}
	else
	{
		do {
			q = (1.0f - (nCol - 1) * epsilon) * (nodeCol == i) + (epsilon) * (nodeCol != i);
			threshold += q;
			i++;
		} while (threshold < randnum);
	}
	qStar_d[idx] = q;											//save the probability of the color chosen
	if ((i - 1) >= nCol)										//TEMP
		i = nCol;
	starColoring_d[idx] = i - 1;
}
#endif // COLOR_DECREASE_LINE_CUMULATIVE || COLOR_DECREASE_EXP_CUMULATIVE


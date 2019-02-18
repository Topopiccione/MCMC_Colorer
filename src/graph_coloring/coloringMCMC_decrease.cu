// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

#if defined(COLOR_DECREASE_LINE) || defined(COLOR_DECREASE_EXP)
__global__ void ColoringMCMC_k::selectStarColoringDecrease(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, float * probDistribution_d, curandState * states, float lambda, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

#ifdef TABOO
	if (taboo_d[idx] > 0) {
		taboo_d[idx]--;
		qStar_d[idx] = (1.0f - (nCol - 1) * epsilon);			//save the probability of the color chosen
		return;
	}
#endif // TABOO

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

	float denomReminder = 0;
	for (int i = 0; i < Zp; i++)
	{
		denomReminder += exp(-lambda * i);
	}
	/*if (idx == 0) {
		printf("---------------------------------------\n");
		printf("reminder: %f\n", reminder);
		printf("denom reminder: %f\n", denomReminder);
		float tot = 0, tot2 = 0;
		float ex = reminder / Zp;
		for (int i = 0; i < Zp; i++)
		{
			float r = reminder * (exp(-lambda * i) / denomReminder);
			printf("r: %f\n", r);
			printf("ex: %f\n", ex);
			tot += r;
			tot2 += ex;
		}
		printf("TOT: %f\n", tot);
		printf("TOT ex: %f\n", tot2);
		printf("---------------------------------------\n");
	}*/


	int i = 0, j = 0;
	float q;
	float threshold = 0;
	float randnum = curand_uniform(&states[idx]);				//random number
	if (colorsChecker[nodeCol])									//if node color is used by neighbors
	{
		do {
			float r = reminder * (exp(-lambda * j) / denomReminder);
			q = (probDistribution_d[i] + r) * (!colorsChecker[i]) + (epsilon) * (colorsChecker[i]);
			threshold += q;
			/*if (idx == 0) {
				printf("i = %d\n", i);
				printf("probDistribution_d[i] = %f\n", probDistribution_d[i]);
				printf("r = %f\n", r);
				printf("q = %f\n", q);
				printf("threshold = %f\n", threshold);
				printf("randnum = %f\n", randnum);
			}*/
			j += !colorsChecker[i];
			i++;
		} while (threshold < randnum && i < nCol);
	}
	else
	{
		do {
			q = (1.0f - (nCol - 1) * epsilon) * (nodeCol == i) + (epsilon) * (nodeCol != i);
			threshold += q;
			i++;
		} while (threshold < randnum && i < nCol);
	}
	qStar_d[idx] = q;											//save the probability of the color chosen
	starColoring_d[idx] = i - 1;

#ifdef TABOO
	taboo_d[idx] = (starColoring_d[idx] == nodeCol) * tabooIteration;
#endif // TABOO
}
#endif // COLOR_DECREASE_LINE || COLOR_DECREASE_EXP


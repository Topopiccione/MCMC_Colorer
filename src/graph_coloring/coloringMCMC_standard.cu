// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "coloringMCMC.h"

/**
* For every node, look at neighbors and select a new color.
* This will be write in starColoring_d and the probability of the chosen color will be write in qStar_d
*/
#ifdef STANDARD
__global__ void ColoringMCMC_k::selectStarColoring(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * taboo_d, uint32_t tabooIteration, curandState * states, float epsilon, uint32_t * statsFreeColors_d) {

	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

#ifdef TABOO
	if (taboo_d[idx] > 0) {
		taboo_d[idx]--;
		return;
	}
#endif // TABOO

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set to 1 or 0 the colors occupied from the neighbors
	for (int i = 0; i < nneighs; i++)
		colorsChecker[coloring_d[neighs[index + i]]] = 1;

	uint32_t Zp = nCol, Zn = 0;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
	{
		Zn += colorsChecker[i];
	}
	Zp -= Zn;

#ifdef STATS
	statsFreeColors_d[idx] = Zp;
#endif // STATS

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

	float randnum = curand_uniform(&states[idx]);				//random number

	float threshold = 0;
	float q;
	uint32_t selectedIndex = 0;									//selected index for orderedColors to select the new color
	do {
		q =
			(((1 - epsilon * Zn) / Zp) * (!colorsChecker[selectedIndex]) + (epsilon) * (colorsChecker[selectedIndex]))
			* colorsChecker[nodeCol]
			+ ((1.0f - (nCol - 1) * epsilon) * (nodeCol == selectedIndex) + (epsilon) * (nodeCol != selectedIndex))
			* !colorsChecker[nodeCol];
		threshold += q;
		selectedIndex++;
	} while (threshold < randnum && selectedIndex < nCol);

	//if (colorsChecker[nodeCol])									//if node color is used by neighbors == 1
	//{
	//	do {
	//		q = ((1 - epsilon * Zn) / Zp) * (!colorsChecker[selectedIndex]) + (epsilon) * (colorsChecker[selectedIndex]);
	//		threshold += q;
	//		selectedIndex++;
	//	} while (threshold < randnum && selectedIndex < nCol);
	//}
	//else
	//{
	//	do {
	//		q = (1.0f - (nCol - 1) * epsilon) * (nodeCol == selectedIndex) + (epsilon) * (nodeCol != selectedIndex);
	//		threshold += q;
	//		selectedIndex++;
	//	} while (threshold < randnum && selectedIndex < nCol);
	//}
	qStar_d[idx] = q;											//save the probability of the color chosen
	starColoring_d[idx] = selectedIndex - 1;

#ifdef TABOO
	taboo_d[idx] = (starColoring_d[idx] == nodeCol) * tabooIteration;
#endif // TABOO
}
#endif // STANDARD

/**
* For every node, look at neighbors.
* The probability of the old color will be write in probColoring_d
*/
__global__ void ColoringMCMC_k::lookOldColoring(uint32_t nnodes, uint32_t * starColoring_d, float * q_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, float epsilon) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= nnodes)
		return;

	uint32_t index = cumulDegs[idx];							//index of the node in neighs
	uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

	uint32_t nodeCol = coloring_d[idx];							//node color
	uint32_t nodeStarCol = starColoring_d[idx];					//node new color

	bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set to 1 or 0 the colors occupied from the neighbors
	for (int i = 0; i < nneighs; i++)
		colorsChecker[starColoring_d[neighs[index + i]]] = 1;

	uint32_t Zp = nCol, Zn = 0;									//number of free colors (p) and occupied colors (n)
	for (int i = 0; i < nCol; i++)
		Zn += colorsChecker[i];
	Zp = nCol - Zn;

	if (!Zp)													//manage exception of no free colors
	{
		q_d[idx] = 1;
		return;
	}

	//save the probability of the old color
	q_d[idx] =
		((colorsChecker[nodeStarCol]) *
		((!colorsChecker[nodeCol]) * ((1 - epsilon * Zn) / Zp) + ((colorsChecker[nodeCol])) * epsilon))
		+ ((!colorsChecker[nodeStarCol]) *
		((nodeStarCol == nodeCol) * (1 - ((nCol - 1) * epsilon)) + (nodeStarCol != nodeCol) * epsilon));

	// ORIGINAL
	//if (colorsChecker[nodeStarCol])								//if node color is used by neighbors
	//{
	//	if (!colorsChecker[nodeCol])
	//		q_d[idx] = (1 - epsilon * Zn) / Zp;					//save the probability of the old color
	//	else
	//		q_d[idx] = epsilon;									//save the probability of the old color
	//}
	//else
	//{
	//	if (nodeStarCol == nodeCol)
	//		q_d[idx] = 1 - ((nCol - 1) * epsilon);				//save the probability of the old color
	//	else
	//		q_d[idx] = epsilon;									//save the probability of the old color
	//}
}


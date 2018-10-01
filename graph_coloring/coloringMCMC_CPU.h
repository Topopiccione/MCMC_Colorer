#pragma once
#include <vector>
#include <iostream>
#include <random>

#include "graph.h"
#include "coloring.h"
#include "easyloggingpp/easylogging++.h"

template<typename nodeW, typename edgeW>
class ColoringMCMC_CPU : public Colorer<nodeW, edgeW> {
public:
	ColoringMCMC_CPU( Graph<nodeW, edgeW>* g, ColoringMCMCParams params, uint32_t seed );
	~ColoringMCMC_CPU();

	void run();

protected:
	// Main data containers
	std::vector<uint32_t>				C;
	std::vector<uint32_t>				Cstar;

	std::vector<float>					p;
	std::vector<float>					pstar;
	std::vector<float>					q;
	std::vector<float>					qstar;

	std::vector<bool>					freezed;
	std::vector<bool>					freeColors;

	std::vector<float>					nodeProbab;

	// RNG
	std::default_random_engine					gen;
	//std::normal_distribution<> 			disxNeg;	// Don't need that
	std::uniform_int_distribution<uint32_t>		unifInitColors;	// For initial coloring extraction
	std::uniform_real_distribution<float>		unifDistr;		// Generic [0,1] extractor
	std::bernoulli_distribution					bernieFreeze;	// For freezing nodes

	// Parameters
	const size_t						nNodes;
	const uint32_t						seed;
	uint32_t							nCol;
	float								lambda;
	float								epsilon;
	float								ratioFreezed;

	// Vars
	size_t								Cviol;		// # of violation in C
	size_t								Cstarviol;	// # of violation in Cstar
	float								alpha;		// prob of rejecting the new coloration


};

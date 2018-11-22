#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <numeric>

#include "graph/graph.h"
#include "coloring.h"
#include "easyloggingpp/easyloggingpp/easylogging++.h"

template<typename nodeW, typename edgeW>
class ColoringMCMC_CPU : public Colorer<nodeW, edgeW> {
public:
	ColoringMCMC_CPU(Graph<nodeW, edgeW>* g, ColoringMCMCParams params, uint32_t seed);
	~ColoringMCMC_CPU();

	void			run();
	void			show_histogram();
	// Sort of protected methods gone public for integration with GoogleTest
	size_t			violation_count(const std::vector<uint32_t> & currentColoring, std::vector<bool> & violations);
	size_t			count_free_colors(const size_t node, const std::vector<uint32_t> & currentColoring, std::vector<bool> & freeColors);
	void			fill_p(const size_t currentNode, const size_t Zv);
	void			extract_new_color(const size_t currentNode, const std::vector<float> & pVect,
		const std::vector<float> & experimentVect, std::vector<float> & qVect, std::vector<uint32_t> & newColoring);
	void 			fill_qstar(const size_t currentNode, const size_t Zv, const std::vector<uint32_t> & newColoring,
		const std::vector<uint32_t> & oldColoring, const std::vector<bool> & freeCols, const std::vector<bool> & newColoringViols,
		std::vector<float> & qVect);

	// Just for class dbg...
	std::vector<uint32_t>	*	getC() { return &C; }
	std::vector<uint32_t> 	* 	getCstar() { return &Cstar; }
	std::vector<float>		*	getp() { return &p; }
	std::vector<float>		*	getpstar() { return &pstar; }
	std::vector<float>		*	getq() { return &q; }
	std::vector<float>		*	getqstar() { return &qstar; }
	std::vector<float>		*	getnodeProbab() { return &nodeProbab; }
	std::vector<bool>		*	getfreezed() { return &freezed; }
	std::vector<bool>		*	getfreeColors() { return &freeColors; }
	std::vector<bool>		*	getCviols() { return &Cviols; }
	std::vector<bool>		*	getCstarviols() { return &Cstarviols; }
	uint32_t				*	getnCol() { return &nCol; }
	float					*	getlambda() { return &lambda; }
	float					*	getepsilon() { return &epsilon; }
	size_t					*	getCviol() { return &Cviol; }
	size_t					*	getCstarviol() { return &Cstarviol; }
	float					*	getalpha() { return &alpha; }

protected:
	// Main data containers
	std::vector<uint32_t>						C;
	std::vector<uint32_t>						Cstar;

	std::vector<float>							p;
	std::vector<float>							pstar;
	std::vector<float>							q;
	std::vector<float>							qstar;

	std::vector<float>							nodeProbab;

	std::vector<bool>							freezed;
	std::vector<bool>							freeColors;

	std::vector<bool>							Cviols;
	std::vector<bool>							Cstarviols;

	std::vector<size_t>							colorIdx;

	// RNG
	std::default_random_engine					gen;
	std::uniform_int_distribution<uint32_t>		unifInitColors;	// For initial coloring extraction
	std::uniform_real_distribution<float>		unifDistr;		// Generic [0,1] extractor
	//std::bernoulli_distribution					bernieFreeze;	// For freezing nodes

	// Parameters
	const GraphStruct<nodeW, edgeW> *	const	str;
	const size_t								nNodes;
	const uint32_t								seed;
	uint32_t									nCol;
	float										lambda;
	float										epsilon;
	float										ratioFreezed;

	float										expLambda;
	float										divider;

	// Vars
	size_t										Cviol;		// # of violation in C
	size_t										Cstarviol;	// # of violation in Cstar
	float										alpha;		// prob of rejecting the new coloration

	// For stats...
	size_t										Zvcomp_max;
	size_t										Zvcomp_min;
	size_t										Zvcomp_avg;


};

#pragma once

#include <memory>
#include "graph/graph.h"
#include "utils/miscUtils.h"

typedef uint32_t col;     // node color
typedef uint32_t col_sz;     // node color

// graph coloring
struct Coloring {
	uint32_t	 		nCol{ 0 };				// num of color classes
	uint32_t		*	colClass{ nullptr };	// list (array) of all node colors class by class
	uint32_t		*	cumulSize{ nullptr };	// cumulative color class sizes

	/// return the size of class c
	uint32_t classSize(col c) {
		return cumulSize[c] - cumulSize[c - 1];
	}
};

template<typename nodeW, typename edgeW> class Colorer {
public:
	Colorer(Graph<nodeW, edgeW>* g) : graph{ g } {
		//std::default_random_engine eng{};
		//std::uniform_real_distribution<> randU(0.0, 1.0);
		//seed = static_cast<float>(randU(eng));
	}
	virtual ~Colorer() {};

	void 					run();
	float					efficiencyNumProcessors(uint32_t);
	Coloring			*	getColoring();
	bool					checkColoring();
	col_sz 					getNumColor() const;
	void 					buildColoring(col*, node_sz);
	void 					print(bool);
	double 					getElapsedTime() { return elapsedTimeSec; };
	void 					setSeed(float s) { seed = s; };
	void 					verbose() { verb = 1; };

protected:
	std::string 			name;
	Graph<nodeW, edgeW>	*	graph;
	Coloring			* 	coloring;
	float				*	meanClassDeg;	  		// mean degs of class nodes
	float					meanClassDegAll{ 0 };	// mean deg over all class nodes
	float 					stdClassDegAll{ 0 };	// std of deg of all IS nodes
	double 					elapsedTimeSec{ 0 };
	float 					seed{ 0 };            	// for random generator
	bool 					verb{ 0 };
};

////////////////////////////////////////////////

template<typename nodeW, typename edgeW>
class ColoringGreedyCPU : public Colorer<nodeW, edgeW> {
public:
	ColoringGreedyCPU(Graph<nodeW, edgeW>* g);
	void run();
	~ColoringGreedyCPU();
};

////////////////////////////////////////////////
struct ColoringMCMCParams {
	uint32_t		maxRip;
	col_sz			nCol;
	float			numColorRatio;
	float			lambda;
	float			epsilon;
	float			ratioFreezed;
	uint32_t		tabooIteration;
	bool			tailcut;
};

#pragma once
#include <iostream>
#include <vector>
#include <string>
#ifdef __unix
#include <sys/ioctl.h>
#endif
#ifdef WIN32
#include <conio.h>
#endif

template <typename nodeW, typename edgeW>
class ColoringMCMC_CPU;

#include "graph/graph.h"
#include "graph_coloring/coloring.h"
#include "graph_coloring/coloringMCMC_CPU.h"
#include "graph_coloring/coloringLuby.h"
//#include "graph_coloring/coloringMCMC.h"

class dbg {
public:
	dbg();
	dbg( Graph<float, float> * gg, ColoringMCMC_CPU<float, float> * colMCMC );
	~dbg();

	bool 						check_F12keypress();
	void 						stop_and_debug();
private:
	int 						kbhit();
	std::vector<std::string>	split_str( std::string s );
	uint32_t					parse_and_exec( std::vector<std::string> ss );
	void 						print_var( std::vector<std::string> ss );
	void 						edit_var( std::vector<std::string> ss );

	Graph<float, float>		*	gr;
	ColoringMCMC_CPU<float, float> * col;

	// Get all the relevant vars and containers from graph and colorer
	// graphStruct:
	node						nNodes;			//
	node_sz						nEdges;			//
	node_sz					*	cumulDegs;		//
	node					*	neighs;			//
	// colorer:
	std::vector<uint32_t>	*	C;				//
	std::vector<uint32_t>	*	Cstar;			//
	std::vector<float>		*	p;				//
	std::vector<float>		*	pstar;
	std::vector<float>		*	q;
	std::vector<float>		*	qstar;
	std::vector<float>		*	nodeProbab;		//
	std::vector<bool>		*	freezed;		//
	std::vector<bool>		*	freeColors;		//
	std::vector<bool>		*	Cviols;			//
	std::vector<bool>		*	Cstarviols;		//
	uint32_t				*	nCol;			//
	float					*	lambda;			//
	float					*	epsilon;		//
	size_t					*	Cviol;			//
	size_t					*	Cstarviol;		//
	float					*	alpha;
};

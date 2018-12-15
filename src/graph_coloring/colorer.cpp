// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <map>
#include "coloring.h"
#include "graph/graph.h"
#include "utils/timer.h"

using namespace std;

/**
 * Build a Coloring for a graph
 * @param C array of node colors
 * @param n num of node
 */
template<typename nodeW, typename edgeW>
void Colorer<nodeW, edgeW>::buildColoring( col* C, node_sz n ) {
	coloring = new Coloring {};
	coloring->colClass = new col[n]{};

	set<col> color_set;
	multimap<col, node> count_map;
	for (unsigned i = 0; i < n; i++) {
		count_map.insert(pair<col, node>(C[i],i));
		color_set.insert(C[i]);
	}

	unsigned nCol = (uint32_t) color_set.size();
	coloring->nCol = nCol;
	coloring->cumulSize = new col_sz[nCol+1]{};
	GraphStruct<nodeW, edgeW>* str = graph->getStruct();
	meanClassDeg = new float[nCol]{};
	coloring->cumulSize[0] = 0;
	col j = 1;
	node v = 0;
	for (set<col>::iterator it = color_set.begin(); it != color_set.end(); it++) {
		col color_K = *it;
		std::pair <std::multimap<unsigned,unsigned>::iterator, std::multimap<unsigned,unsigned>::iterator> ret;
		ret = count_map.equal_range(color_K);
		coloring->cumulSize[j] = coloring->cumulSize[j-1];
		for (std::multimap<unsigned,unsigned>::iterator ii = ret.first; ii != ret.second; ++ii) {
			coloring->colClass[v++] = ii->second;
			coloring->cumulSize[j]++;
			meanClassDeg[j-1] += str->deg(ii->second);
		}
		meanClassDeg[j-1] /= (float)(coloring->cumulSize[j]-coloring->cumulSize[j-1]);
		meanClassDegAll += meanClassDeg[j-1];
		j++;
	}
	meanClassDegAll /= (float)nCol;
	stdClassDegAll = 0;
	for (unsigned c = 0; c < nCol; c++)
		stdClassDegAll += (meanClassDeg[c]-meanClassDegAll)*(meanClassDeg[c]-meanClassDegAll);
	stdClassDegAll /= (float)nCol;
}

/**
 * Return the graph coloring
 * @return the coloring
 */
template<typename nodeW, typename edgeW>
Coloring* Colorer<nodeW, edgeW>::getColoring() {
	return coloring;
}

/**
 * Return the coloring number of colors
 */
template<typename nodeW, typename edgeW>
unsigned Colorer<nodeW, edgeW>::getNumColor() const {
	return coloring->nCol;
}

/**
 * Print a coloring
 */
template<typename nodeW, typename edgeW>
void Colorer<nodeW, edgeW>::print( bool verbose ) {
	col_sz nCol = coloring->nCol;
	cout << "\n** Coloring method:" << name.c_str() << endl;
	cout << "   - num colors: " << nCol << endl;
	cout << "   - mean color class deg: " << meanClassDegAll << endl;
	cout << "   - std color class deg: " << stdClassDegAll << endl;

	if (verbose) {
		for (col c = 1; c <= nCol; c++) {
			cout << "   col(" << c << ")" << "[" << coloring->classSize(c) << "]-> ";
			for (unsigned int j = 0; j < coloring->classSize(c); j++) {
				cout << coloring->colClass[coloring->cumulSize[c-1] + j] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}


/**
 * Efficiency measure for a given coloring based on number of processors
 */
template<typename nodeW, typename edgeW>
float Colorer<nodeW, edgeW>::efficiencyNumProcessors( unsigned nProc ) {

	// compute efficiency
	float E = 0;
	for (col c = 1; c <= coloring->nCol; c++) {
		col_sz cs = coloring->classSize(c);
			E += cs/(float)nProc / (float)ceil(cs/(float)nProc);
	}
	return E/(float)coloring->nCol;
}

template<typename nodeW, typename edgeW>
bool Colorer<nodeW,edgeW>::checkColoring() {
	col_sz nCol = coloring->nCol;
	GraphStruct<nodeW, edgeW>* str = graph->getStruct();
	for (col c = 1; c <= nCol; c++) {
		unsigned start = coloring->cumulSize[c-1];
		unsigned c_size = coloring->classSize(c);
		for (unsigned i = 0; i < c_size-1; i++)
			for (unsigned j = i+1; j < c_size; j++) {
				node u = coloring->colClass[start+j];
				node v = coloring->colClass[start+i];
				if (str->areNeighbor(v,u))
					cout << "COLORING ERROR: nodes " << u << " and " << v << "have the same color " << c << endl;
			}
	}
	return true;
}

template<typename nodeW, typename edgeW>
ColoringGreedyCPU<nodeW,edgeW>::ColoringGreedyCPU( Graph<nodeW, edgeW>* g ) : Colorer<nodeW, edgeW>( g ) {
	this->coloring = new Coloring();
	this->name = "Greedy-CPU";
}

template<typename nodeW, typename edgeW>
ColoringGreedyCPU<nodeW,edgeW>::~ColoringGreedyCPU() {
	delete this->coloring;
	delete[] this->meanClassDeg;
}

/**
 * Greedy First Fit algorithm for coloring building
 */
template<typename nodeW, typename edgeW>
void ColoringGreedyCPU<nodeW,edgeW>::run() {
	Timer time;   // timer objects
	time.startTime();
	GraphStruct<nodeW, edgeW>* str = this->graph->getStruct();
	node_sz n = str->nNodes;

	// sort node idx based on node degrees
	node_sz *buffDeg = new node_sz[n];
	node *nodePerm = new node[n];
	for (node i = 0; i < n; i++) {
		buffDeg[i] = str->deg(i);
		nodePerm[i] = i;
	}
	sort(nodePerm, nodePerm + n, [buffDeg](size_t k, size_t j) {return buffDeg[k] < buffDeg[j];});

	// vector of vectors, i.e. color classes
	vector< vector<col> > col_class {vector<col> {nodePerm[0]}};
	int nCol = 1;
	for (unsigned i = 1; i < n; i++) {
		bool DONE_OUTER = false;
		for (auto& c : col_class) {
			bool DONE_INNER = true;
			for (unsigned j = 0; j < c.size(); j++) {
				if (str->areNeighbor(c.at(j), nodePerm[i])) {
					DONE_INNER = false;
					break;
				}
			}
			if (DONE_INNER) {
				DONE_OUTER = true;
				c.push_back(nodePerm[i]);
				break;
			}
		}
		if (!DONE_OUTER) {
			vector<col> c {nodePerm[i]};
			col_class.push_back(c);
			nCol++;
		}
	}

	// fill the Coloring data structure
	col* C = new col[n];
	col color = 1;
	for (vector< vector<col> >::iterator i = col_class.begin(); i != col_class.end(); ++i) {
		for (vector<col>::iterator j = i->begin(); j != i->end(); ++j) {
			C[*j] = color;
		}
		color++;
	}
	time.endTime();
	this->elapsedTimeSec = time.duration();
	this->buildColoring(C, n);

	// clean
	delete[] nodePerm;
	delete[] buffDeg;
	delete[] C;
}



//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class Colorer<col,col>;
template class Colorer<float,float>;
template class ColoringGreedyCPU<col, col>;
template class ColoringGreedyCPU<float, float>;

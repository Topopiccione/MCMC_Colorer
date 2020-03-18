#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cinttypes>
#include "utils/miscUtils.h"

class fileImporter {
public:
	fileImporter( std::string graphFileName, std::string labelFileName );
	~fileImporter();

	void 		fRewind();
	bool		getNextEdge();
	void		getNextLabelling( std::string & currentClassName );
	uint32_t	getNumberOfClasses();

	std::ifstream graphFile;
	std::ifstream labelFile;

	std::map<std::string, int> geneMap;
	std::map<int, std::string> inverseGeneMap;

	uint32_t nNodes;
	uint32_t nEdges;
	uint32_t nOfClasses;

	// questi vengono passati alla funzione di costruzione del grafo
	bool					edgeIsValid;
	uint32_t				srcIdx, dstIdx;
	double					edgeWgh;
	std::vector<int32_t>	labelsFromFile;
	uint32_t				nLabelsFromFile;


private:
	// Variabili utilizzate nella lettura dei file
	std::stringstream 		ss;
	std::string				inStr;
	std::string				lab1, src, dst;
	float ww;
	double ww_d;

	std::string				currentClass, geneIn, classIn;
	bool 					firstRound;

};

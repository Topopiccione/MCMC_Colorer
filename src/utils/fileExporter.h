#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cinttypes>

#include "COSNet/COSNet.h"

class fileExporter {
public:
	fileExporter( std::string outFileName, std::string geneFileName, std::string foldsFilename, std::string statesFilename,
		std::map<int, std::string> * inverseGeneMap, uint32_t nNodes );
	~fileExporter();

	void saveGeneNames();
	template <typename nodeW, typename edgeW>
	void saveClass( const std::string & currentClassName, COSNet<nodeW, edgeW> * CN );

	std::ofstream outFile;
	std::ofstream geneFile;
	std::ofstream foldsFile;
	std::ofstream statesFile;

	std::string outFileName;
	std::string geneFileName;
	std::string foldsFilename;
	std::string statesFilename;

	std::map<int, std::string> * inverseGeneMap;

	uint32_t nNodes;


private:
	// Variabili utilizzate nella lettura dei file
	std::stringstream 		ss;
	uint32_t			*	tempFolds;

};

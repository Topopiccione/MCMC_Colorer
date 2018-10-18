// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "utils/fileExporter.h"

fileExporter::fileExporter( std::string outFileName, std::string geneFileName, std::string foldsFilename, std::string statesFilename,
	std::map<int, std::string> * inverseGeneMap, uint32_t nNodes ) :
		geneFileName( geneFileName ), outFileName( outFileName ),
		foldsFilename( foldsFilename ), statesFilename( statesFilename ),
		outFile{ std::ofstream( outFileName.c_str(), std::ios::out ) },
		inverseGeneMap { inverseGeneMap }, nNodes{ nNodes } {

	if (foldsFilename != "")
		foldsFile = std::ofstream( foldsFilename.c_str(), std::ios::out );
	if (statesFilename != "")
		statesFile = std::ofstream( statesFilename.c_str(), std::ios::out );

	tempFolds = new uint32_t[nNodes];
}

fileExporter::~fileExporter() {
	delete[] tempFolds;
	outFile.close();
	if (geneFile.is_open())
		geneFile.close();
	if (foldsFile.is_open())
		foldsFile.close();
	if (statesFile.is_open())
		statesFile.close();
}

void fileExporter::saveGeneNames() {
	geneFile = std::ofstream( geneFileName.c_str()/*, std::ios::out*/ );

	if (geneFile.is_open()) {
		//std::cout << "Salvo i nomi su un file" << std::endl;

		geneFile << nNodes << std::endl;

		for (uint32_t i = 0; i < nNodes; i++) {
			geneFile << inverseGeneMap->at(i) << std::endl;
		}

		geneFile.close();
	}
}

template <typename nodeW, typename edgeW>
void fileExporter::saveClass( const std::string & currentClassName, COSNet<nodeW, edgeW> * CN ) {
	if (outFile.is_open()) {
		outFile << currentClassName << "\t";

		uint32_t N = CN->nNodes;
		for (uint32_t i = 0; i < N; i++) {
			outFile << CN->scores[i] << "\t";
		}

		outFile << std::endl;
	}

	if (statesFile.is_open()) {
		statesFile << currentClassName << "\t";

		uint32_t N = CN->nNodes;
		for (uint32_t i = 0; i < N; i++) {
			statesFile << CN->states[i] << "\t";
		}

		statesFile << std::endl;
	}

	if (foldsFile.is_open()) {
		std::fill( tempFolds, tempFolds + nNodes, 0 );
		// Conversione da folds/foldIdx verso formato esplicito nodo -> fold di appartenenza
		for (uint32_t i = 0; i < CN->numberOfFolds; i++) {
			for (uint32_t j = CN->foldIndex[i]; j < CN->foldIndex[i + 1]; j++)
				tempFolds[CN->folds[j]] = i;
		}

		foldsFile << currentClassName << "\t";

		uint32_t N = CN->nNodes;
		for (uint32_t i = 0; i < N; i++) {
			foldsFile << tempFolds[i] << "\t";
		}

		foldsFile << std::endl;
	}

}

template void fileExporter::saveClass<float,float>( const std::string & currentClassName, COSNet<float, float> * CN );

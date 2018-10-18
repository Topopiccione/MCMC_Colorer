// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "utils/fileImporter.h"

fileImporter::fileImporter( std::string graphFileName, std::string labelFileName ) :
		graphFile{ std::ifstream( graphFileName.c_str(), std::ios::in ) },
		labelFile{ std::ifstream( labelFileName.c_str(), std::ios::in ) },
		currentClass( "" ), geneIn( "" ), classIn( "" ), firstRound{ true }  {
	// leggo dal file tutti i nomi dei geni e costruisco il set temporaneo
	// intanto raccatto qualche info prliminare sul grafo...
	//std::string inStr;
	//std::string lab1, src, dst;
	int i = 0;
	int edges = 0;
	//float ww;
	//std::stringstream ss;
	std::set<std::string> tempGeneNamesSet;

	// Skip prima linea di header
	// (e intanto controllo che i file siano stati aperti)
	if (graphFile)
		std::getline( graphFile, inStr );
	else
		std::cout << "errore apertura file net" << std::endl;
	if (labelFile)
		std::getline( labelFile, inStr );
	else
		std::cout << "errore apertura file etichette" << std::endl;

	while (graphFile) {
		std::getline( graphFile, inStr );
		if (inStr == "")
			continue;
		ss << inStr;

		ss >> src;
		ss >> dst;
		ss >> ww;
		tempGeneNamesSet.insert( src );
		tempGeneNamesSet.insert( dst );

		ss.str( "" );
		ss.clear();
	}

	nNodes = (uint32_t) tempGeneNamesSet.size();

	// Ora costruisco le mappe dirette e inverse
	for (auto it = tempGeneNamesSet.begin(); it != tempGeneNamesSet.end(); ++it) {
		geneMap.insert( std::pair<std::string, int>( *it, i ) );
		inverseGeneMap.insert( std::pair<int, std::string>( i, *it ) );
		i++;
	}

	// Ora che conosco da quanti elementi e' formata la rete, posso pre-allocare
	// il vettore delle etichette
	labelsFromFile = std::vector<int32_t>(nNodes, 0);

	// Al termine del costruttore, avrò a disposizione le due mappe, il numero
	// di nodi e il vettore dell'importazione delle etichette
}

fileImporter::~fileImporter() {
	graphFile.close();
	labelFile.close();
}

void fileImporter::fRewind() {
	// "riavvolge" il file e salta l'header
	std::string inStr;

	graphFile.clear();
	graphFile.seekg( 0 );
	std::getline( graphFile, inStr );

	// Reset dello stringstream
	ss.str( "" );
	ss.clear();
}

uint32_t fileImporter::getNumberOfClasses() {
	std::string inStr;
	std::set<std::string> tempLabelNamesSet;

	labelFile.clear();
	labelFile.seekg( 0 );
	std::getline( labelFile, inStr );

	while (labelFile) {
		std::getline( labelFile, inStr );
		if (inStr == "")
			continue;
		ss << inStr;

		ss >> src;
		ss >> dst;
		tempLabelNamesSet.insert( dst );

		ss.str( "" );
		ss.clear();
	}

	nOfClasses =(uint32_t) tempLabelNamesSet.size();

	// Riavvolge il file e salta linea di header
	labelFile.clear();
	labelFile.seekg( 0 );
	std::getline( labelFile, inStr );

	return nOfClasses;
}

bool fileImporter::getNextEdge() {
	edgeIsValid = false;
	if (!graphFile)
		return false;

	do {
		std::getline( graphFile, inStr );
	} while ((inStr == "") && (graphFile));

	if (!graphFile)
		return false;

	ss << inStr;
	ss >> src;
	ss >> dst;
	ss >> ww_d;

	srcIdx = geneMap.at( src );
	dstIdx = geneMap.at( dst );
	edgeWgh = ww_d;
	edgeIsValid = true;

	ss.str( "" );
	ss.clear();
	return true;
}

void fileImporter::getNextLabelling( std::string & currentClassName ) {
	nLabelsFromFile = 0;
	// etichette tutte negative di default, modifico solo le positive
	std::fill( labelsFromFile.begin(), labelsFromFile.end(), -1 );

	bool classNameNotSet = true;

	do {
		if (inStr != "") {
			// questa versione aggiunge le etichette positive al vettore labelsFromFile
			//labelsFromFile[nLabelsFromFile++] = geneMap[geneIn];
			// questa, invece, modifica in "+1" le posizioni relative alle etichette positive
			labelsFromFile[geneMap[geneIn]] = 1;
			nLabelsFromFile++;
			if (classNameNotSet) {
				currentClassName = classIn;
				classNameNotSet = false;
			}
		}
		std::getline( labelFile, inStr );
		if (!labelFile)
			break;
		if (inStr == "")
			continue;

		ss.str( "" );
		ss.clear();
		ss << inStr;
		ss >> geneIn;
		ss >> classIn;
		if (classIn != currentClass) {
			currentClass = classIn;
			if (!firstRound) {
				break;
			}
			firstRound = false;
		}
	} while (true);
}

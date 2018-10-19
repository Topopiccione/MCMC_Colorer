// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <algorithm>
#include <random>
#include <string>
#include <cstring>
#include <vector>
#include <set>
//#include <ctime>
//#include "graph/graph.h"	// in realta' non serve!

#define DUPLICATE_CHECK
#define BIDIR_CHECK

#ifdef WIN32
#define INLINE_ARGS
#endif

std::vector<std::string> generateRandomName(const int n);

int main(int argc, const char ** argv) {

#ifdef INLINE_ARGS
	argv[1] = "25000";
	argv[2] = "lab.txt";
	argv[3] = "net.txt";
	argv[4] = "0.01";
#else
	//printf(argc + "");
	//return;
	if (argc < 5) {
		std::cout << "usage: ./datasetGenerator numberOfNodes nameOfPositiveLabelFile nameOfNetFile edgeProbability" << std::endl;
		return -1;
	}
#endif // INLINE_ARGS

	// argc
	// 1: nNodes
	// 2: nome label
	// 3: nome rete
	// 4: densita'
	uint32_t		nNodes = atoi(argv[1]);
	uint32_t		nClasses = 5;
	std::string		labelFileName(argv[2]);
	std::string		netFileName(argv[3]);
	float			probLabels = 0.01f;
	float			probDensity = atof(argv[4]);

	std::cout << "nNodes: " << nNodes << " - probDensity: " << probDensity << " - label: "
		<< labelFileName << " - net: " << netFileName << std::endl;

	auto seed = 10000;

	//std::default_random_engine eng( time( NULL ) );
	std::default_random_engine eng(seed);
	std::uniform_real_distribution<> randR(0.0, 1.0);
	std::normal_distribution<> randNorm(0, 0.1);

	std::ofstream labelFile(labelFileName.c_str(), std::ios::out);
	std::ofstream netFile(netFileName.c_str(), std::ios::out);

	if (!labelFile.is_open()) {
		std::cerr << "errore aperture file etichette" << std::endl;
		exit(-1);
	}

	if (!netFile.is_open()) {
		std::cerr << "errore aperture file rete" << std::endl;
		exit(-1);
	}

	// Richiama generateRandomNames per generare il vettore dei nomi dei nodi
	std::vector<std::string> nodeNames = generateRandomName(nNodes);

	std::string classBaseName("GO::00");
	// Ciclo for da 0 a nClasses per generare le etichettature
	// Ogni iterazione genera una nuova string formata da classBaseName + numIterazione
	// esempio: "GO::001", poi "GO::002", ecc...
	// Nel file devono essere salvati solo i nomi dei nodi positivi
	for (uint32_t i = 0; i < nClasses; i++) {
		std::string currentClassName(classBaseName + std::to_string(i));
		for (uint32_t j = 0; j < nNodes; j++) {
			// estrazione di un numero random
			// se estrazione ha successo
			if (randR(eng) < probLabels) {
				labelFile << nodeNames[j] << "\t" << currentClassName << std::endl;
			}
		}
	}

	labelFile.close();

	// Ciclo per generazione della rete
	// Usare stessa strategia di generazione del generatore di grafi di Erdos
	// gia' implementata.
	// Per la scrittura dell'arco generato, usare l'istruzione:
	// netfile << nodeNames[i] << " " << nodeNames[j] << " " << randomWeight << std::endl;
	uint64_t 	*	cumulDegs = new uint64_t[nNodes + 1];
	uint64_t	*	neighs;
	float		*	weights;
	uint64_t		nEdges = 0;
	// uint64_t		nodiIsolatiCorretti = 0;

	std::fill(cumulDegs, cumulDegs + nNodes + 1, 0);
	std::cout << "|--------------------|" << std::endl << "\033[F|";

	std::vector<uint64_t>	* edges = new std::vector<uint64_t>[nNodes];
	for (uint32_t i = 0; i < nNodes - 1; i++) {
		if (i % (nNodes / 20) == 0)
			std::cout << "#" << std::flush;
		bool haiAlmenoUnArco = false;
		for (uint32_t j = i + 1; j < nNodes; j++)
			if (randR(eng) < probDensity) {
				edges[i].push_back(j);
				//edges[j].push_back(i);
				cumulDegs[i + 1]++;
				//cumulDegs[j + 1]++;
				nEdges += 1;// 2;
				haiAlmenoUnArco = true;
			}
		// if (!haiAlmenoUnArco) {
		// 	//std::cout << "Nodo isolato: " << i << std::endl;
		// 	uint32_t aa = (rand() % (nNodes - i)) + 1;
		// 	//std::cout << "Creato edge con: " << aa << std::endl;
		// 	edges[i].push_back(aa);
		// 	//edges[aa].push_back(i);
		// 	cumulDegs[i + 1]++;
		// 	//cumulDegs[aa + 1]++;
		// 	nEdges += 1;// 2;
		// 	nodiIsolatiCorretti++;
		// }
	}
	cumulDegs[0] = 0;
	for (uint32_t i = 0; i < nNodes; i++)
		cumulDegs[i + 1] += cumulDegs[i];

	std::cout << std::endl << "nEdges: " << nEdges << std::endl;

	neighs = new uint64_t[nEdges];
	for (uint32_t i = 0; i < nNodes; i++)
		memcpy((neighs + cumulDegs[i]), edges[i].data(), sizeof(uint64_t) * edges[i].size());


	// Controlli:
	// Arco non deve apparire due volte
#ifdef DUPLICATE_CHECK
	std::cout << "Checking for duplicate edges in neighbor lists..." << std::endl;
	for (size_t i = 0; i < nNodes; i++) {
		size_t nodeDeg = cumulDegs[i + 1] - cumulDegs[i];
		auto neighIdx = neighs + cumulDegs[i];
		auto aa = std::unique(neighIdx, neighIdx + nodeDeg);
		if (aa != (neighs + cumulDegs[i + 1])) {
			std::cout << "Aborting: duplicate in edge list of node " << i << std::endl;
			exit(-1);
		}
	}
#endif

	// Grafo NON deve essere bidirezionale, perche' l'arco di ritorno viene aggiunto dal costruttore di Graph
#ifdef BIDIR_CHECK
	for (size_t i = 0; i < nNodes; i++) {
		size_t nodeDeg = cumulDegs[i + 1] - cumulDegs[i];
		auto neighIdx = neighs + cumulDegs[i];
		for (size_t j = 0; j < nodeDeg; j++) {
			auto neighNode = neighIdx[j];
			auto neighNodeDeg = cumulDegs[neighNode + 1] - cumulDegs[neighNode];
			auto neighNodeIdx = neighs + cumulDegs[neighNode];
			// look for i in the neigh list of neighNode. There should be none;
			auto aa = std::find(neighNodeIdx, neighNodeIdx + neighNodeDeg, i);
			if (aa != neighNodeIdx + neighNodeDeg) {
				std::cout << "Aborting: found bidirectional edge between nodes " << i << " and " << neighNode << std::endl;
				std::cout << i << " neighs list: ";
				std::for_each(neighIdx, neighIdx + nodeDeg, [&](uint32_t val) {std::cout << val << " "; });
				std::cout << std::endl << neighNode << " neighs list: ";
				std::for_each(neighNodeIdx, neighNodeIdx + neighNodeDeg, [&](uint32_t val) {std::cout << val << " "; });
				exit(-1);
			}
		}
	}
#endif

	std::cout << "Saving..." << std::endl;
	std::cout << "|--------------------|" << std::endl << "\033[F|";
	netFile << nNodes << "\t" << nEdges << std::endl;

	for (uint32_t i = 0; i < nNodes; i++) {
		if (i % (nNodes / 20) == 0)
			std::cout << "#" << std::flush;
		for (uint64_t j = cumulDegs[i]; j < cumulDegs[i + 1]; j++) {
			netFile << nodeNames[i] << "\t" << nodeNames[neighs[j]] << "\t" << randR(eng) << std::endl;
			//netFile << nodeNames[i] << "\t" << nodeNames[neighs[j]] << "\t" << fabs( randNorm( eng ) ) << std::endl;
		//netFile << i << "\t" << neighs[j] << "\t" <<  fabs( randNorm( eng ) ) << std::endl;
		}
	}

	std::cout << std::endl;
	// std::cout << "Nodi Isolati: " << nodiIsolatiCorretti << std::endl;

	netFile.close();
	return 0;

}


std::vector<std::string> generateRandomName(const int n) {
	const char alphanum[] =
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";
	std::vector<std::string> out;
	std::set<std::string> tempSet;
	const int slen = 12;	// <- dovrebbe bastare
	char stringa[slen + 1];
	stringa[slen] = 0;

	while (tempSet.size() < n) {
		std::for_each(stringa, stringa + slen, [alphanum](char &c) {c = alphanum[rand() % (sizeof(alphanum) - 1)]; });
		tempSet.emplace(stringa);
	}

	for (auto it = tempSet.begin(); it != tempSet.end(); it++) {
		out.push_back(std::string(*it));
	}

	/*for (int i = 0; i < n; i++) {
		std::for_each( stringa, stringa + slen, [alphanum](char &c){c = alphanum[rand() % (sizeof(alphanum) - 1)];} );
		out.push_back( std::string( stringa ) );
	}*/

	return out;
}

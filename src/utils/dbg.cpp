#include "dbg.h"


dbg::dbg( Graph<float, float> * g, ColoringMCMC_CPU<float, float> * colMCMC ) :
	gr( g ), col( colMCMC ) {

	// graphstruct
	nNodes			= g->getStruct()->nNodes;
	nEdges			= g->getStruct()->nEdges;
	cumulDegs		= g->getStruct()->cumulDegs;
	neighs			= g->getStruct()->neighs;

	// colorer
	C				= col->getC();
	Cstar			= col->getCstar();
	p				= col->getp();
	pstar			= col->getpstar();
	q				= col->getq();
	qstar			= col->getqstar();
	nodeProbab		= col->getnodeProbab();
	freezed			= col->getfreezed();
	freeColors		= col->getfreeColors();
	Cviols			= col->getCviols();
	Cstarviols		= col->getCstarviols();
	nCol			= col->getnCol();
	lambda			= col->getlambda();
	epsilon			= col->getepsilon();
	Cviol			= col->getCviol();
	Cstarviol		= col->getCstarviol();
	alpha			= col->getalpha();
};

dbg::~dbg() {
#ifdef __unix
	system("stty cooked echo");
#endif
}

// Due to non-portability of console-related commands, this function has to be implemented
// in different ways depending on the OS
bool dbg::check_F12keypress() {
// on linux, use syscalls
// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
#ifdef __unix
	system("stty raw -echo");
	int i = 0;
	int ii = 0;
	int c;
	if ( ii = kbhit() ) {
		c = getchar();
		system("stty cooked echo");
		if (c == 27)	// Escape
			return true;
		else
			return false;
	}
	system("stty cooked echo");
	// Note that it swithces the console back and forth between different modes and this may affect performances
#endif
// on win / dos, uses conio.h
#ifdef WIN32
	if (GetKeyState(VK_ESCAPE) & 0x8000)
		return true;
	else
		return false;
#endif
}



void dbg::stop_and_debug() {
	std::cout << "Esc key pressed. Start debugging... (type h for help or q for continue with the execution)" << std::endl;
	std::string inputStr;
	std::vector<std::string> splittedStr;
	do {
		std::cout << ">>> " <<std::flush;
		getline (std::cin, inputStr);
		if (inputStr == "")
			continue;
		splittedStr = split_str( inputStr );
		//std::cout << "command size: " << splittedStr.size() << std::endl;
		//std::for_each( std::begin(splittedStr), std::end(splittedStr), [&](std::string val) {std::cout << val << std::endl;} );
		parse_and_exec( splittedStr );
		inputStr.clear();
	} while (splittedStr[0] != "q");
}

int dbg::kbhit() {
#ifdef __unix
	int i;
 	ioctl( 0, FIONREAD, &i );
	return i; /* return a count of chars available to read */
#endif
#ifdef WIN32
	return 0;
#endif
}

std::vector<std::string> dbg::split_str( std::string s ) {
	std::vector<std::string> toBeRet;
	std::string delimiters = " ";
	size_t current;
	size_t next = -1;
	do {
		current = next + 1;
		next = s.find_first_of( delimiters, current );
		if (s.substr( current, next - current ) != "")
 			toBeRet.push_back( s.substr( current, next - current ) );
	} while (next != std::string::npos);
	return toBeRet;
}

uint32_t dbg::parse_and_exec( std::vector<std::string> ss ) {
	std::string comm = ss[0];
	if ((comm == "p") | (comm == "print")) {
		print_var( ss );
	} else if ((comm == "e") | (comm == "edit")) {
		edit_var( ss );
	} else if ((comm == "q") | (comm == "quit")) {
		std::cout << "continuing execution..." << std::endl;
	} else if ((comm == "h") | (comm == "help")) {
		std::cout << "  -- Available commands: " << std::endl;
		std::cout << "  ----- p: print a variable " << std::endl;
		std::cout << "  ----- e: edit a variable " << std::endl;
		std::cout << "  ----- h: show this help " << std::endl;
		std::cout << "  ----- q: quit debugger " << std::endl;
		std::cout << "  -- Available variable for printing: " << std::endl;
		std::cout << "  ----- p nNodes:            prints the number of nodes in the graph " << std::endl;
		std::cout << "  ----- p nEdges:            prints the number of edges in the graph " << std::endl;
		std::cout << "  ----- p deg <node>:        prints the degree of node <node> " << std::endl;
		std::cout << "  ----- p neighs <node>:     prints the list of neighbors of node <node> " << std::endl;
		std::cout << "  ----- p color <node>:      prints the current color of node <node> " << std::endl;
		std::cout << "  ----- p nextcolor <node>:  prints the next (extracted) color of node <node> " << std::endl;
		std::cout << "  ----- p p <node>:          prints the vector of current probability p for node <node> " << std::endl;
		std::cout << "  ----- p nodeProb <node>:   prints the extracted probability for the cdf experiment for node <node> " << std::endl;
		std::cout << "  ----- p freezed:           prints the list of freezed nodes" << std::endl;
		std::cout << "  ----- p freezed <node>:    prints if node <node> is freezed or not" << std::endl;
		std::cout << "  ----- p freeColors <node>: prints the list of free colors for node <node>" << std::endl;
		std::cout << "  ----- p Cviols:            prints the list of conflicting nodes in the current coloration" << std::endl;
		std::cout << "  ----- p Cviols <node>:     prints if node <node> is conflicting or not in the current coloration" << std::endl;
		std::cout << "  ----- p Cstarviols:        prints the list of conflicting nodes in the next (extracted) coloration" << std::endl;
		std::cout << "  ----- p Cstarviols <node>: prints if node <node> is conflicting or not in the next (extracted) coloration" << std::endl;
		std::cout << "  ----- p Cviol:             prints the number of conflicts in the current coloration" << std::endl;
		std::cout << "  ----- p Cstarviol:         prints the number of conflicts in the next (extracted) coloration" << std::endl;
		std::cout << "  ----- p nCol:              prints the number of colors" << std::endl;
		std::cout << "  ----- p lambda:            prints the lambda parameter" << std::endl;
		std::cout << "  ----- p epsilon:           prints the epsilon parameter" << std::endl;
		std::cout << "  -- Available variable for editing: " << std::endl;
		std::cout << "  ----- e epsilon <val>:     change epsilon value; range: 0 <= val < 1" << std::endl;
	} else {
		std::cout << "Invalid command (type 'h' or 'help' for a quick guide)" << std::endl;
	}
	return 0;
}

void dbg::print_var( std::vector<std::string> ss ) {
	std::string varName = ss[1];
	std::string param;
	int paramInt;
	if (ss.size() == 3)
		param = ss[2];
	// Mege if-else...
	if (varName == "nNodes")
		std::cout << "nNodes: " << nNodes << std::endl;
	if (varName == "nEdges")
		std::cout << "nEdges: " << nEdges << std::endl;
	if (varName == "deg") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p deg 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else
				std::cout << "Degree of node " << varName << ": " << cumulDegs[paramInt + 1] - cumulDegs[paramInt] << std::endl;
		}
	}
	if (varName == "neighs") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p neighs 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				std::cout << "Neighbors of node " << paramInt << ": ";
				size_t deg = cumulDegs[paramInt + 1] - cumulDegs[paramInt];
				auto neighStart = neighs + cumulDegs[paramInt];
				std::for_each( neighStart, neighStart + deg, [&](node val) {std::cout << val << " ";} );
				std::cout << std::endl;
			}
		}
	}
	if (varName == "color") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p color 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else
				std::cout << "Color of node " << paramInt << ": " << C->at(paramInt) << std::endl;
		}
	}
	if (varName == "nextcolor") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p nextcolor 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else
				std::cout << "Color of node " << paramInt << ": " << Cstar->at(paramInt) << std::endl;
		}
	}
	if (varName == "p") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p p 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				if (freezed->at(paramInt))
					std::cout << "Not available: node freezed..." << std::endl;
				else {
					if (Cviols->at(paramInt))
						std::cout << "Node is conflicting" << std::endl;
					else
						std::cout << "Node is NOT conflicting" << std::endl;
					size_t Zvcomp = col->count_free_colors( paramInt, *C, *freeColors );
					size_t Zv = *nCol - Zvcomp;
					// Fill vect p.
					col->fill_p( paramInt, Zv );
					size_t ccc = Cstar->at(paramInt);
					//std::for_each( std::begin(*p), std::end(*p), [&](float val) {std::cout << val << " ";} );
					for (size_t i = 0; i < p->size(); i++) {
						if (i != ccc)
							std::cout << p->at(i) << " ";
						else
							std::cout << TXT_BIBLU << p->at(i) << TXT_NORML <<" ";
					}
					std::cout << std::endl;
				}
			}
		}
	}
	if (varName == "nodeProb") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p nodeProb 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else
				std::cout << "Extacted probabilit for node " << paramInt << ": " << nodeProbab->at(paramInt) << std::endl;
		}
	}
	if (varName == "freezed") {
		if (ss.size() < 3) {
			std::cout << "List of freezed nodes: ";
			for (size_t i = 0; i < freezed->size(); i++) {
				if (freezed->at(i))
					std::cout << i << " ";
			}
			std::cout << std::endl;
		} else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				if (freezed->at(paramInt))
					std::cout << "Node freezed" << std::endl;
				else
					std::cout << "Node NOT freezed" << std::endl;
			}
		}
	}
	if (varName == "freeColors") {
		if (ss.size() < 3)
			std::cout << "Specify the node (es. p freeColors 12)" << std::endl;
		else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				col->count_free_colors( paramInt, *C, *freeColors );
				std::cout << "List of available colors for node " << paramInt << ": ";
				for (size_t i = 0; i < freeColors->size(); i++) {
					if (freeColors->at(i))
						std::cout << i << " ";
				}
				std::cout << std::endl;
			}
		}
	}
	if (varName == "Cviols") {
		if (ss.size() < 3) {
			std::cout << "List of conflicting nodes: ";
			for (size_t i = 0; i < Cviols->size(); i++) {
				if (Cviols->at(i))
					std::cout << i << " ";
			}
			std::cout << std::endl;
		} else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				if (Cviols->at(paramInt))
					std::cout << "Node conflicting" << std::endl;
				else
					std::cout << "Node NOT conflicting" << std::endl;
			}
		}
	}
	if (varName == "Cstarviols") {
		if (ss.size() < 3) {
			std::cout << "List of conflicting nodes: ";
			for (size_t i = 0; i < Cstarviols->size(); i++) {
				if (Cstarviols->at(i))
					std::cout << i << " ";
			}
			std::cout << std::endl;
		} else {
			paramInt = atoi( param.c_str() );
			if (paramInt >= nNodes)
				std::cout << "Node out of range" << std::endl;
			else {
				if (Cstarviols->at(paramInt))
					std::cout << "Node conflicting" << std::endl;
				else
					std::cout << "Node NOT conflicting" << std::endl;
			}
		}
	}
	if (varName == "nCol")
		std::cout << "number of colors: " << *nCol << std::endl;
	if (varName == "lambda")
		std::cout << "lambda: " << *lambda << std::endl;
	if (varName == "epsilon")
		std::cout << "epsilon: " << *epsilon << std::endl;
	if (varName == "Cviol")
		std::cout << "number of violations current coloring: " << *Cviol << std::endl;
	if (varName == "Cstarviol")
		std::cout << "number of violations next coloring: " << *Cstarviol << std::endl;
}

void dbg::edit_var( std::vector<std::string> ss ) {
	std::string varName = ss[1];
	std::string param1;
	std::string param2;
	int param1Int, param2Int;
	float param1float, param2float;
	if ((ss.size() == 3) | (ss.size() == 4))
		param1 = ss[2];
	if (ss.size() == 4)
		param2 = ss[3];
	if (varName == "epsilon") {
		if (ss.size() < 3)
			std::cout << "Specify the new value for epsilon (es. e epsilon 1e-4)" << std::endl;
		else {
			param1float = atof( param1.c_str() );
			if ((param1float < 0) | (param1float > 1)) {
				std::cout << "New epsilon value out of range (0 < epsilon < 1)" << std::endl;
			} else {
				*epsilon = param1float;
			}
		}
	}
}

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
	// TODO: WIN32 version
	if (GetKeyState(VK_ESCAPE) & 0x8000)
		return true;
	else
		return false;
#endif
}



void dbg::stop_and_debug() {
	std::cout << "Esc key pressed. Start debugging..." << std::endl;
	std::string inputStr;
	std::vector<std::string> splittedStr;
	do {
		getline (std::cin, inputStr);
		splittedStr = split_str( inputStr );
		//std::cout << "command size: " << splittedStr.size() << std::endl;
		//std::for_each( std::begin(splittedStr), std::end(splittedStr), [&](std::string val) {std::cout << val << std::endl;} );
		parse_and_exec( splittedStr );
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
		// Do nothing... at least now
	} else if ((comm == "q") | (comm == "quit")) {
		std::cout << "continuing execution..." << std::endl;
	} else if ((comm == "h") | (comm == "help")) {
		std::cout << "  -- Available commands: " << std::endl;
		// TODO: complete
	} else {
		std::cout << "Invalid command (type 'h' or 'help' for a quick guide)" << std::endl;
	}
	return 0;
}

void dbg::print_var( std::vector<std::string> ss ) {
	std::string varName = ss[1];
	std::string param;
	if (ss.size() == 3)
		param = ss[2];
	// Mege if-else...
	if (varName == "nNodes")
		std::cout << "nNodes: " << nNodes << std::endl;
}

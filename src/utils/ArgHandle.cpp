// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
// COSnet - Commandline Argument Handler
// Alessandro Petrini, 2017
#include "ArgHandle.h"
#ifdef __unix
	#include <getopt.h>
#endif
#ifdef WIN32
	#include "getopt/getopt.h"
#endif

ArgHandle::ArgHandle( int argc, char **argv ) :
		dataFilename( "" ), foldFilename( "" ), labelFilename( "" ), outFilename( "" ), geneOutFilename( "" ),
		statesOutFilename( "" ), foldsOutFilename( "" ), timeOutFilename( "" ),
		m( 0 ), n( 0 ), prob( 0.0 ), numColRatio( 1.0 ), nCol(0),
		nFolds( 0 ), seed( 0 ), verboseLevel(0),
		nThreads( 0 ), repetitions( 1 ), tabooIteration(2),
		generateRandomFold( false ), simulate( false ), argc( argc ), argv( argv ) {
}

ArgHandle::~ArgHandle() {}

void ArgHandle::processCommandLine() {

	char const *short_options = "d:l:f:m:n:s:N:o:C:b:g:u:j:S:q:r:t:v:h:T";
	const struct option long_options[] = {

		{ "data",			required_argument, 0, 'd' },
		{ "labels",			required_argument, 0, 'l' },
		{ "folds",			required_argument, 0, 'f' },
		{ "features",		required_argument, 0, 'm' },
		{ "variables",		required_argument, 0, 'n' },
		{ "simulate",		required_argument, 0, 's' },
		{ "nFolds",			required_argument, 0, 'N' },
		{ "out",			required_argument, 0, 'o' },
		{ "nCol",			required_argument, 0, 'C' },
		{ "numColRatio",	required_argument, 0, 'b' },
		{ "geneOut",		required_argument, 0, 'g' },
		{ "foldsOut",		required_argument, 0, 'u' },
		{ "statesOut",		required_argument, 0, 'j' },
		{ "seed",			required_argument, 0, 'S' },
		{ "nThrd",			required_argument, 0, 'q' },
		{ "repet",			required_argument, 0, 'r' },
		{ "tttt",			required_argument, 0, 't' },
		{ "verbose-level",	required_argument, 0, 'v' },
		{ "help",			no_argument,	   0, 'h' },
		{ "tabooIteration", required_argument, 0, 'T' },
		{ 0, 0, 0, 0 }
	};

	while (1) {
		int option_index = 0;
		int c = getopt_long( argc, argv, short_options, long_options, &option_index );

		if (c == -1) {
			break;
		}

		switch (c) {
		case 's':
			simulate = true;
			try {
				double temp = std::stod( optarg );
				if ((temp < 0) | (temp > 1)) {
					std::cout << "\033[31;1mSimulation: probabilty of positive class must be 0 < prob < 1.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					prob = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mArgument missing: specify the probabilty for positive class.\033[0m" << std::endl;
				exit( -1 );
			}
			break;
		case 'd':
			dataFilename = std::string( optarg );
			break;

		case 'l':
			labelFilename = std::string( optarg );
			break;

		case 'f':
			foldFilename = std::string( optarg );
			break;

		case 'o':
			outFilename = std::string( optarg );
			break;

		case 'g':
			geneOutFilename = std::string( optarg );
			break;

		case 'u':
			foldsOutFilename = std::string( optarg );
			break;

		case 'j':
			statesOutFilename = std::string( optarg );
			break;

		case 't':
			timeOutFilename = std::string( optarg );
			break;

		case 'C':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mnCol must be a positive integer.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					nCol = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mnCol must be a positive integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'b':
			try {
				double temp = std::stod( optarg );
				if ((temp < 1.0f) | (temp > 16.0f)) {
					std::cout << "\033[31;1mColor ratio must be 1.0 < numColRatio < 16.0.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					numColRatio = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mArgument missing: specify color ratio 1 <= numColRatio <= 16.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'n':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mn must be a positive integer.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					n = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mn must be a positive integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'm':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mm must be a positive integer.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					m = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mm must be a positive integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;
		case 'N':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mnFold argument must be a positive integer.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					nFolds = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mnFold argument must be a positive integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'S':
			try {
				int temp = std::stoi( optarg );
				seed = temp;
			}
			catch (...) {
				std::cout << "\033[31;1mseed argument must be integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'q':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					temp = 0;
				}
				else {
					nThreads = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mensThrd argument must be integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'r':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mrepetitions must be a positive integer.\033[0m" << std::endl;
					exit( -1 );
				}
				else {
					repetitions = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mrepetitions must be a positive integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'v':
			try {
				int temp = std::stoi( optarg );
				verboseLevel = temp;
			}
			catch (...) {
				std::cout << "\033[31;1mverbose-level argument must be integer.\033[0m" << std::endl;
				exit( -1 );
			}
			break;

		case 'h':
			displayHelp();
			exit( 0 );
			break;

		case 'T':
			try {
				int temp = std::stoi(optarg);
				if (temp < 1) {
					std::cout << "\033[31;1mtabooIteration must be a positive integer.\033[0m" << std::endl;
					exit(-1);
				}
				else {
					tabooIteration = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mtabooIteration must be a positive integer.\033[0m" << std::endl;
				exit(-1);
			}
			break;

		default:
			break;
		}
	}

	if (outFilename.empty()) {
		std::cout << "\033[33;1mNo output file name defined. Default used (--out).\033[0m" << std::endl;
		outFilename = std::string( "output.txt" );
	}

	if (simulate && (n == 0)) {
		std::cout << "\033[31;1mSimualtion enabled: specify n (-n).\033[0m" << std::endl;
		exit( -1 );
	}

	if (simulate && ((prob < 0) | (prob > 1))) {
		std::cout << "\033[31;1mSimulation: probabilty of positive class must be 0 < prob < 1.\033[0m" << std::endl;
		exit( -1 );
	}

	if (!simulate) {
		if (dataFilename.empty()) {
			std::cout << "\033[31;1mMatrix file undefined (--data).\033[0m" << std::endl;
			exit( -1 );
		}

		if (labelFilename.empty()) {
			std::cout << "\033[31;1mLabel file undefined (--label).\033[0m" << std::endl;
			exit( -1 );
		}

		if (foldFilename.empty()) {
			std::cout << "\033[33;1mNo fold file name defined. Random generation of folds enabled (--fold).\033[0m";
			generateRandomFold = true;
			if (nFolds == 0) {
				std::cout << "\033[33;1m [nFold = 3 as default (--nFolds)]\033[0m";
				nFolds = 3;
			}
			std::cout << std::endl;
		}

		if (!foldFilename.empty() && (nFolds != 0)) {
			std::cout << "\033[33;1mnFolds option ignored (mumble, mumble...).\033[0m" << std::endl;
		}

		if (geneOutFilename.empty()) {
			std::cout << "\033[33;1mNo output gene names file name defined (--gene).\033[0m" << std::endl;
			exit( -1 );
		}
	}

	if (simulate & (nFolds == 0)) {
		std::cout << "\033[33;1mNo number of folds specified. Using default setting: 3 (--nFolds).\033[0m" << std::endl;
		nFolds = 3;
	}

	if (seed == 0) {
		seed = (uint32_t) time( NULL );
		std::cout << "\033[33;1mNo seed specified. Generating a random seed: " << seed << " (--seed).\033[0m" << std::endl;
		srand( seed );
	}

	if (nThreads <= 0) {
		std::cout << "\033[33;1mNo threads specified. Executing in single thread mode (--nThrd).\033[0m" << std::endl;
		nThreads = 1;
	}

	if (verboseLevel > 3) {
		std::cout << "\033[33;1mNverbose-level higher than 3.\033[0m" << std::endl;
		verboseLevel = 3;
	}

	if (verboseLevel < 0) {
		std::cout << "\033[33;1mverbose-level lower than 0.\033[0m" << std::endl;
		verboseLevel = 0;
	}
}


void ArgHandle::displayHelp() {
	std::cout << " **** CosNET-GPU ****" << std::endl;
	std::cout << "( --- qui bisogna scrivere qualcosa tipo i nomi degli autori e una descrizione dell'applicazione --- )" << std::endl;

	std::cout << "Usage: " << std::endl;
	std::cout << "    " << argv[0] << " [options]" << std::endl;
	std::cout << std::endl;

	std::cout << "Options:" << std::endl;
	std::cout << "    " << "--help               Print this help." << std::endl;
	std::cout << "    " << "--out file.txt       Output file. Is a space-separated value text file." << std::endl;
	std::cout << "    " << "--statesOut file.txt Optional output file containing the states of each node at the end of execution" <<std::endl;
	std::cout << "    " << "                     of the Hopfield dynamics." << std::endl;
	std::cout << "    " << "--label file.txt     Labelling input file. Must be a space-separated value text file." << std::endl;
	std::cout << "    " << "                     Each 1 in the label file represent a minor class candidate, 0 otherwise." << std::endl;
	std::cout << "    " << "                     Parameter 'n' is assigned by reading this file." << std::endl;
	std::cout << "    " << "--data file.txt      Data matrix input file. Must be a space-separated value text file." << std::endl;
	std::cout << "    " << "                     Data is read row wise, where each row is a feature and each column" << std::endl;
	std::cout << "    " << "                     represents a sample." << std::endl;
	std::cout << "    " << "--fold file.txt      Optional fold input file. Must be a space-separated value text file." << std::endl;
	std::cout << "    " << "                     Folds in text file must be numbered from 0. If unspecified, random" << std::endl;
	std::cout << "    " << "                     generation of fold is enabled." << std::endl;
	std::cout << "    " << "--simulate P         Enable simulation of a random data / label / fold set. Positive examples are" << std::endl;
	std::cout << "    " << "                     generated with probability P (0 < P < 1). -m and -n parameters are required." << std::endl;
	std::cout << "    " << "-m M                 Number of features to be generated. Enabled only if --simulate option is specified." << std::endl;
	std::cout << "    " << "-n N                 Number of samples to be generated. Enabled only if --simulate option is specified." << std::endl;
	std::cout << "    " << "--nFolds N           Number of folds for cross validation [default = 3]. This option is disabled if" << std::endl;
	std::cout << "    " << "                     a fold file is specified." << std::endl;
	std::cout << "    " << "--nThrd              Number of threads" << std::endl;
	std::cout << "    " << "--seed N             Seed for random number generator." << std::endl;
	std::cout << "    " << "--verbose-level N    Level of verbosity: 0 = silent" << std::endl;
	std::cout << "    " << "                                         1 = only progress" << std::endl;
	std::cout << "    " << "                                         2 = rf and progress" << std::endl;
	std::cout << "    " << "                                         3 = complete" << std::endl;
	std::cout << std::endl << std::endl;
}

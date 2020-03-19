// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
// MCMC Colorer - Command line argument manager
// Alessandro Petrini, 2019-20
#include "ArgHandle.h"
#ifdef __unix
	#include <getopt.h>
#endif
#ifdef WIN32
	#include "getopt/getopt.h"
#endif

ArgHandle::ArgHandle( int argc, char **argv ) :
		graphFilename(""), outDir(""),
		simulate(false), prob(0.0), n(0),
		mcmccpu(false), mcmcgpu(false), lubygpu(false),
		nCol(0), numColRatio(0.0), tabooIteration(0), tailcut(false),
		repetitions(1), seed(0),
		verboseLevel(0),
		argc(argc), argv(argv) {
}

ArgHandle::~ArgHandle() {}

void ArgHandle::processCommandLine() {

	printLogo();

	char const *short_options = "g:o:s:n:1:2:3:k:r:t:l:R:S:v:h:M";
	const struct option long_options[] = {

		{ "graph",			required_argument, 0, 'g' },
		{ "outDir",			required_argument, 0, 'o' },

		{ "simulate",		required_argument, 0, 's' },
		{ "nodes",			required_argument, 0, 'n' },

		{ "mcmccpu",		no_argument,	   0, '1' },
		{ "mcmcgpu",		no_argument,	   0, '2' },
		{ "lubygpu",		no_argument,	   0, '3' },

		{ "nCol",			required_argument, 0, 'k' },
		{ "numColRatio",	required_argument, 0, 'r' },
		{ "tabooIteration", required_argument, 0, 't' },
		{ "tailcut",		no_argument,	   0, 'l' },

		{ "repet",			required_argument, 0, 'R' },
		{ "seed",			required_argument, 0, 'S' },

		{ "verbose-level",	required_argument, 0, 'v' },
		{ "help",			no_argument,	   0, 'h' },
		{ "cite-me",		no_argument,	   0, 'M' },

		{ 0, 0, 0, 0 }
	};

	while (1) {
		int option_index = 0;
		int c = getopt_long( argc, argv, short_options, long_options, &option_index );

		if (c == -1) {
			break;
		}

		switch (c) {
			case 'g':
				graphFilename = std::string( optarg );
				break;

			case 'o':
				outDir = std::string( optarg );
				break;

			case 's':
				simulate = true;
				try {
					double temp = std::stod( optarg );
					if ((temp < 0) | (temp > 1)) {
						std::cout << TXT_BIRED << "Simulation: probabilty of positive class must be 0 < prob < 1." << TXT_NORML << std::endl;
						exit( -1 );
					}
					else {
						prob = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "Argument missing: specify the probabilty for positive class." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 'n':
				try {
					int temp = std::stoi( optarg );
					if (temp < 1) {
						std::cout << TXT_BIRED << "n must be a positive integer." << TXT_NORML << std::endl;
						exit( -1 );
					}
					else {
						n = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "n must be a positive integer." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case '1':
				mcmccpu = true;
				break;

			case '2':
				mcmcgpu = true;
				break;

			case '3':
				lubygpu = true;
				break;


			case 'k':
				try {
					int temp = std::stoi( optarg );
					if (temp < 1) {
						std::cout << TXT_BIRED << "nCol must be a positive integer." << TXT_NORML << std::endl;
						exit( -1 );
					}
					else {
						nCol = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "nCol must be a positive integer." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 'r':
				try {
					double temp = std::stod( optarg );
					if ((temp < 1.0f) | (temp > 16.0f)) {
						std::cout << TXT_BIRED << "Color ratio must be 1.0 < numColRatio < 16.0." << TXT_NORML << std::endl;
						exit( -1 );
					}
					else {
						numColRatio = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "Argument missing: specify color ratio 1 <= numColRatio <= 16." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 't':
				try {
					int temp = std::stoi(optarg);
					if (temp < 1) {
						std::cout << TXT_BIRED << "tabooIteration must be a positive integer." << TXT_NORML << std::endl;
						exit(-1);
					}
					else {
						tabooIteration = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "tabooIteration must be a positive integer." << TXT_NORML << std::endl;
					exit(-1);
				}
				break;

			case 'l':
				tailcut = true;
				break;

			case 'R':
				try {
					int temp = std::stoi( optarg );
					if (temp < 1) {
						std::cout << TXT_BIRED << "repetitions must be a positive integer." << TXT_NORML << std::endl;
						exit( -1 );
					}
					else {
						repetitions = temp;
					}
				}
				catch (...) {
					std::cout << TXT_BIRED << "repetitions must be a positive integer." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 'S':
				try {
					int temp = std::stoi( optarg );
					seed = temp;
				}
				catch (...) {
					std::cout << TXT_BIRED << "seed argument must be integer." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 'v':
				try {
					int temp = std::stoi( optarg );
					verboseLevel = temp;
				}
				catch (...) {
					std::cout << TXT_BIRED << "verbose-level argument must be integer." << TXT_NORML << std::endl;
					exit( -1 );
				}
				break;

			case 'h':
				displayHelp();
				exit( 0 );
				break;

			case 'M':
				citeMe();
				exit(0);
				break;

		default:
			break;
		}
	}

	////////////////////

	if ((!simulate) && (graphFilename.empty())) {
		std::cout << TXT_BIRED << "Graph file undefined (--graph). Specify a graph file or enable simulation mode." << TXT_NORML << std::endl;
		exit(-1);
	}

	if ((!mcmccpu) && (!mcmcgpu) && (!lubygpu)) {
		std::cout << TXT_BIYLW << "No coloring algorithm specified: enabling MCMC CPU by default (--mcmccpu | --mcmcgpu | --lubygpu)" << TXT_NORML << std::endl;
		mcmccpu = true;
	}

	if (simulate && (n == 0)) {
		std::cout << TXT_BIRED << "Simualtion enabled: specify the number of nodes (-n)." << TXT_NORML << std::endl;
		exit( -1 );
	}

	if (simulate && ((prob < 0) | (prob > 1))) {
		std::cout << TXT_BIRED << "Simulation: probabilty of positive class must be 0 < prob < 1." << TXT_NORML << std::endl;
		exit( -1 );
	}

	if ((mcmccpu || mcmcgpu) && (nCol == 0)) {
		std::cout << TXT_BIYLW << "No number of colors specified (--nCol): enabling default value: maxDeg / numColRatio." << TXT_NORML << std::endl;
		//exit(-1);
	}

	if (numColRatio == 0.0) {
		std::cout << TXT_BIYLW << "Using default color ratio (1.0) (--numColRatio)" << TXT_NORML << std::endl;
		numColRatio = 1.0;
	}

	if (seed == 0) {
		seed = (uint32_t) time( NULL );
		std::cout << TXT_BIYLW << "No seed specified. Generating a random seed: " << seed << " (--seed)." << TXT_NORML << std::endl;
		srand( seed );
	}

	if (verboseLevel > 3) {
		std::cout << TXT_BIYLW << "verbose-level higher than 3." << TXT_NORML << std::endl;
		verboseLevel = 3;
	}

	if (verboseLevel < 0) {
		std::cout << TXT_BIYLW << "verbose-level lower than 0." << TXT_NORML << std::endl;
		verboseLevel = 0;
	}

	//////////////////
	if (!simulate) {
		std::vector<std::string> justFilename = split_str(graphFilename, "/\\");
		std::vector<std::string> nameNoExtV = split_str(justFilename[justFilename.size() - 1], ".");
		if (nameNoExtV.size() > 1) {
			std::vector<std::string>::const_iterator first = nameNoExtV.begin();
			std::vector<std::string>::const_iterator last  = nameNoExtV.end() - 1;
			graphName = join_str(std::vector<std::string>(first, last), ".");
		} else {
			graphName = nameNoExtV[0];
		}
	} else {
		graphName = std::to_string(n) + "_" + std::to_string(prob) + "_" + std::to_string(numColRatio);
	}

	if (outDir.empty()) {
		outDir = graphName + "_out";
		std::cout << TXT_BIYLW << "No output directory defined. Saving to: " << outDir << " (--outDir)." << TXT_NORML << std::endl;
		mkdir(outDir.c_str(), 0775);
	}

}


void ArgHandle::displayHelp() {
	std::cout << TXT_BIGRN << "Usage: " << TXT_NORML << std::endl;
	std::cout << "    " << argv[0] << " [options]" << std::endl;
	std::cout << std::endl;

	std::cout << TXT_BIGRN << "Options:" << TXT_NORML << std::endl;
	std::cout << "    " << "--help               Print this help." << std::endl;
	std::cout << TXT_BIYLW << "  Dataset" << TXT_NORML << std::endl;
	std::cout << "    " << "--graph file.txt     Input graph specified as a list of edges (mandatory if not in simulation mode)." << std::endl;
	std::cout << "    " << "--outDir             Output directory." << std::endl;
	std::cout << "    " << "--simulate P         Enable simulation of a random Erdos graph. Edges are generated with probability" << std::endl;
	std::cout << "    " << "                     P (0 < P < 1). -n parameter is mandatory." << std::endl;
	std::cout << "    " << "-n N                 Number of nodes to be generated. Enabled only if --simulate option is specified." << std::endl;
	std::cout << TXT_BIYLW << "  Coloring algorithm" << TXT_NORML << std::endl;
	std::cout << "    " << "--mcmccpu            Enables MCMC CPU colorer." << std::endl;
	std::cout << "    " << "--mcmcgpu            Enables MCMC GPU colorer." << std::endl;
	std::cout << "    " << "--lubygpu            Enables Luby GPU colorer." << std::endl;
	std::cout << TXT_BIYLW << "  Coloring options (only for MCMC CPU and MCMC GPU)" << TXT_NORML << std::endl;
	std::cout << "    " << "--nCol N             Number of colors (mandatory)" << std::endl;
	std::cout << "    " << "--numColRatio N.N    Optional divider for number of colors (default = 1.0, 1.0 <= numColRatio <= 16.0)" << std::endl;
	std::cout << "    " << "--tabooIterations N  Optional number of iteration for the taboo strategy" << std::endl;
	std::cout << "    " << "--tailcut            Enables tail cutting strategy (default = disabled)" << std::endl;
	std::cout << TXT_BIYLW << "  General options" << TXT_NORML << std::endl;
	std::cout << "    " << "--repet N            Number of repetitions for each coloring (optional, default = 1)." << std::endl;
	std::cout << "    " << "--seed N             Seed for random number generator(optional, default = random)." << std::endl << std::endl;
	std::cout << "Verbosity level is set in 'logger.conf' file: switch TRACE ENABLE to true for additional prints to the console." << std::endl;
	std::cout << std::endl << std::endl;
}

void ArgHandle::citeMe() {
	std::cout << std::endl << std::endl;
	std::cout << "This work can be cited by adding the following items to your bibliografy:" << std::endl << std::endl << std::endl;
	std::cout << "@inproceedings{colorerGbR2019," << std::endl;
	std::cout << "	author    = {Conte, Donatello and Grossi, Giuliano and Lanzarotti, Raffaella and Lin, Jianyi and Petrini, Alessandro}," << std::endl;
	std::cout << "	title     = {A parallel MCMC algorithm for the Balanced Graph Coloring problem}," << std::endl;
	std::cout << "	booktitle = {IAPR International workshop on Graph-Based Representation in Pattern Recognition, Tours, France}," << std::endl;
	std::cout << "	year      = {2019}," << std::endl;
	std::cout << "	month     = {Jul}," << std::endl;
	std::cout << "	day       = {19-21}" << std::endl;
	std::cout << "}" << std::endl << std::endl;
}

void ArgHandle::printLogo() {
	std::cout <<               "_________________________________________________________________________________________________________" << std::endl << std::endl;
	std::cout << "\033[38;5;215m  ███▄ ▄███▓ ▄████▄   ███▄ ▄███▓ ▄████▄      ▄████▄   ▒█████   ██▓     ▒█████   ██▀███  ▓█████  ██▀███  \e[0m" << std::endl;
	std::cout << "\033[38;5;215m ▓██▒▀█▀ ██▒▒██▀ ▀█  ▓██▒▀█▀ ██▒▒██▀ ▀█     ▒██▀ ▀█  ▒██▒  ██▒▓██▒    ▒██▒  ██▒▓██ ▒ ██▒▓█   ▀ ▓██ ▒ ██▒\e[0m" << std::endl;
	std::cout << "\033[38;5;216m ▓██    ▓██░▒▓█    ▄ ▓██    ▓██░▒▓█    ▄    ▒▓█    ▄ ▒██░  ██▒▒██░    ▒██░  ██▒▓██ ░▄█ ▒▒███   ▓██ ░▄█ ▒\e[0m" << std::endl;
	std::cout << "\033[38;5;216m ▒██    ▒██ ▒▓▓▄ ▄██▒▒██    ▒██ ▒▓▓▄ ▄██▒   ▒▓▓▄ ▄██▒▒██   ██░▒██░    ▒██   ██░▒██▀▀█▄  ▒▓█  ▄ ▒██▀▀█▄  \e[0m" << std::endl;
	std::cout << "\033[38;5;217m ▒██▒   ░██▒▒ ▓███▀ ░▒██▒   ░██▒▒ ▓███▀ ░   ▒ ▓███▀ ░░ ████▓▒░░██████▒░ ████▓▒░░██▓ ▒██▒░▒████▒░██▓ ▒██▒\e[0m" << std::endl;
	std::cout << "\033[38;5;217m ░ ▒░   ░  ░░ ░▒ ▒  ░░ ▒░   ░  ░░ ░▒ ▒  ░   ░ ░▒ ▒  ░░ ▒░▒░▒░ ░ ▒░▓  ░░ ▒░▒░▒░ ░ ▒▓ ░▒▓░░░ ▒░ ░░ ▒▓ ░▒▓░\e[0m" << std::endl;
	std::cout << "\033[38;5;218m ░  ░      ░  ░  ▒   ░  ░      ░  ░  ▒        ░  ▒     ░ ▒ ▒░ ░ ░ ▒  ░  ░ ▒ ▒░   ░▒ ░ ▒░ ░ ░  ░  ░▒ ░ ▒░\e[0m" << std::endl;
	std::cout << "\033[38;5;218m ░      ░   ░        ░      ░   ░           ░        ░ ░ ░ ▒    ░ ░   ░ ░ ░ ▒    ░░   ░    ░     ░░   ░ \e[0m" << std::endl;
	std::cout << "\033[38;5;218m        ░   ░ ░             ░   ░ ░         ░ ░          ░ ░      ░  ░    ░ ░     ░        ░  ░   ░     \e[0m" << std::endl;
	std::cout << "\033[38;5;218m            ░                   ░           ░                                                           \e[0m" << std::endl;
	std::cout <<               "_________________________________________________________________________________________________________" << std::endl << std::endl;
	std::cout <<               "                  PhuseLab / AnacletoLab - Universita' degli studi di Milano - 2019-20                  " << std::endl;
	std::cout <<               "                                   N. Aspes - G. Grossi - A. Petrini                                    " << std::endl;
	std::cout <<               "                                http://github.com/PhuseLab/MCMC_Colorer                                 " << std::endl << std::endl;
	std::cout <<               "                         Use '--cite-me' command line option for citation info                          " << std::endl << std::endl;
	std::cout <<               "                        '--help' for the complete  list of command line options                         " << std::endl;
	std::cout <<               "_________________________________________________________________________________________________________" << std::endl << std::endl;

}

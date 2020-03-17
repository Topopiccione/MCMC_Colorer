// MCMC Colorer - Misc utilities
// Alessandro Petrini, 2019-20
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

// ANSI console command for text coloring
#ifdef __unix
#define TXT_BICYA "\033[96;1m"
#define TXT_BIPRP "\033[95;1m"
#define TXT_BIBLU "\033[94;1m"
#define TXT_BIYLW "\033[93;1m"
#define TXT_BIGRN "\033[92;1m"
#define TXT_BIRED "\033[91;1m"
#define TXT_NORML "\033[0m"
#define TXT_COLMC "\033[92;1m"
#else
#define TXT_BICYA ""
#define TXT_BIPRP ""
#define TXT_BIBLU ""
#define TXT_BIYLW ""
#define TXT_BIGRN ""
#define TXT_BIRED ""
#define TXT_NORML ""
#define TXT_COLMC ""
#endif

extern bool g_traceLogEn;

enum cmode {
	MODE_MCMC_CPU	= 1,
	MODE_MCMC_GPU	= 2,
	MODE_LUBY_GPU 	= 4
};

void checkLoggerConfFile();

std::vector<std::string> split_str(const std::string s, const std::string delimiters);
std::string join_str(const std::vector<std::string> & v, const std::string delim);

// MCMC Colorer - Misc utilities
// Alessandro Petrini, 2019-20
#include "miscUtils.h"

void checkLoggerConfFile() {
	std::ifstream confFile( "logger.conf", std::ios::in );
	if (!confFile) {
		std::cout << TXT_BIYLW << "Logger configuration file not found (logger.conf). Creating one..." << std::endl;
		std::ofstream confFile( "logger.conf", std::ios::out );
		confFile << "* GLOBAL:" << std::endl;
		confFile << "    FORMAT               =  \"%datetime %msg\"" << std::endl;
		confFile << "    FILENAME             =  \"default_ParSMURFng.log\"" << std::endl;
		confFile << "    ENABLED              =  true" << std::endl;
		confFile << "    TO_FILE              =  true" << std::endl;
		confFile << "    TO_STANDARD_OUTPUT   =  true" << std::endl;
		confFile << "    SUBSECOND_PRECISION  =  6" << std::endl;
		confFile << "    PERFORMANCE_TRACKING =  true" << std::endl;
		confFile << "    MAX_LOG_FILE_SIZE    =  4194304	## 4MB" << std::endl;
		confFile << "    LOG_FLUSH_THRESHOLD  =  100 ## Flush after every 100 logs" << std::endl;
		confFile << "* DEBUG:" << std::endl;
		confFile << "    FORMAT               = \"%datetime{%d/%M} %func %msg\"" << std::endl;
		confFile << "* TRACE:" << std::endl;
		confFile << "    ENABLED              =  false" << std::endl;
		confFile.close();
	} else {
		confFile.close();
	}
}

std::vector<std::string> split_str(const std::string s, const std::string delimiters) {
	std::vector<std::string> toBeRet;
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

std::string join_str(const std::vector<std::string> & v, const std::string delim) {
	std::string outString;
	for (std::vector<std::string>::const_iterator p = v.begin(); p != v.end(); ++p) {
		outString += *p;
		if (p != v.end() - 1)
			outString += delim;
	}
	return outString;
}

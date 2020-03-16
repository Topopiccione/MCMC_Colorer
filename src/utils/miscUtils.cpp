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

import os
import json

def lineParser( logFile ):
	tempDict = {}
	for line in logFile:
		if 'Nodes:' in line:
			tempDict["nnodes"] = int( line.split( sep = ' ')[1] )
			tempDict["nedges"] = int( line.split( sep = ' ')[4] )
		if 'Max Deg:' in line:
			tempDict["maxDeg"] = int( line.split( sep = ' ')[2] )
			tempDict["minDeg"] = int( line.split( sep = ' ')[6] )
			tempDict["avgDeg"] = float( line.split( sep = ' ')[10] )
		if 'Edge probability' in line:
			tempDict["edgeProb"] = float( line.split( sep = ' ')[6] )
		if 'Iteration:' in line:
			tempDict["numIter"] = int( line.split( sep = ' ')[1] )
		if 'Iteration performed:' in line:
			tempDict["performedIter"] = int( line.split( sep = ' ')[2] )
		if 'Execution time:' in line:
			tempDict["execTime"] = float( line.split( sep = ' ')[2] )
		if 'Number of colors:' in line:
			tempDict["numColors"] = int( line.split( sep = ' ')[3] )
		if 'Used colors:' in line:
			tempDict["usedColors"] = int( line.split( sep = ' ')[2] )
		if 'Average number' in line:
			tempDict["avgNodesPerColor"] = float( line.split( sep = ' ')[7] )
		if 'Variance:' in line:
			tempDict["varNodesPerColor"] = float( line.split( sep = ' ')[1] )
		if 'StD:' in line:
			tempDict["stdNodesPerColor"] = float( line.split( sep = ' ')[1] )
	return tempDict

def mcmcGpuLineParser( logFile ):
	tempDict = {}
	iterCount = 0
	for line in logFile:
		if 'iteration_' in line:
			iterCount = iterCount + 1
		if 'time ' in line:
			tempDict["execTime"] = float( line.split( sep = ' ')[1] )
		if 'numCol' in line:
			tempDict["numColors"] = int( line.split( sep = ' ')[1] )
		if 'end_used_colors' in line:
			tempDict["usedColors"] = int( line.split( sep = ' ')[1] )
		if 'end_average' in line:
			tempDict["avgNodesPerColor"] = float( line.split( sep = ' ')[1] )
		if 'end_variance' in line:
			tempDict["varNodesPerColor"] = float( line.split( sep = ' ')[1] )
		if 'end_standard_deviation' in line:
			tempDict["stdNodesPerColor"] = float( line.split( sep = ' ')[1] )

	tempDict["performedIter"] = iterCount
	return tempDict

def parseDirs():
	basePath = '.\\expMCMC'
	subDir = os.listdir( basePath )
	totResLuby = {}
	totResMCMC_CPU = {}
	totResMCMC_GPU = {}
	for currDir in subDir:
		if os.path.isfile( basePath + '\\' + currDir ):
			continue
		filesInDir = os.listdir( basePath + '\\' + currDir )
		lubyRes = []
		mcmcCpuRes = []
		mcmcGpuRes = []
		for currLog in filesInDir:
			numNodes = currLog.split( sep='-' )[0]
			prob = currLog.split( sep='-' )[1]

			if 'LUBY' in currLog:
				with open( basePath + '\\' + currDir + '\\' + currLog ) as logFile:
					tempItem = lineParser( logFile )
					tempItem["algo"] = "LUBY"
					lubyRes.append( tempItem )
			if 'MCMC' in currLog:
				with open( basePath + '\\' + currDir + '\\' + currLog ) as logFile:
					tempItem = lineParser( logFile )
					tempItem["algo"] = "MCMC_CPU"
					mcmcCpuRes.append( tempItem )
			if 'resultsFile' in currLog:
				with open( basePath + '\\' + currDir + '\\' + currLog ) as logFile:
					tempItem = mcmcGpuLineParser( logFile )
					tempItem["numIter"] = int( (currLog.split( sep = '-')[3]).split( sep = '.')[0] )
					tempItem["algo"] = "MCMC_GPU"
					mcmcGpuRes.append( tempItem )

		if lubyRes:
			totResLuby[str(numNodes) + '-' + str(prob)] = lubyRes
		if mcmcCpuRes:
			totResMCMC_CPU[str(numNodes) + '-' + str(prob)] = mcmcCpuRes
		if mcmcGpuRes:
			totResMCMC_GPU[str(numNodes) + '-' + str(prob)] = mcmcGpuRes

	with open( basePath + '\\outLuby.json', "w") as outFile:
		json.dump( totResLuby, outFile )
	with open( basePath + '\\outMCMC_CPU.json', "w") as outFile:
		json.dump( totResMCMC_CPU, outFile )
	with open( basePath + '\\outMCMC_GPU.json', "w") as outFile:
		json.dump( totResMCMC_GPU, outFile )

def avgCalc( exp, item ):
	itemAccum = 0
	for expRun in exp:
		itemAccum += expRun[item]
	return itemAccum / len( exp )

def statMakerFromJson():
	basePath = '.\\expMCMC'
	with open( basePath + "\\outLuby.json", "r" ) as read_file:
		dataLuby = json.load( read_file )
	with open( basePath + "\\outMCMC_CPU.json", "r") as read_file:
		dataMcmcCpu = json.load( read_file )
	with open( basePath + "\\outMCMC_GPU.json", "r") as read_file:
		dataMcmcGpu = json.load( read_file )

	results = {}

	for exp in dataLuby:
		lubyDict = {}
		graphDict = {}
		graphDict["nNodes"] = dataLuby[exp][0]["nnodes"]
		graphDict["nEdges"] = dataLuby[exp][0]["nedges"]
		graphDict["edgeProb"] = dataLuby[exp][0]["edgeProb"]
		lubyDict["numColors"] = avgCalc( dataLuby[exp], "numColors")
		lubyDict["execTime"] = avgCalc( dataLuby[exp], "execTime")
		lubyDict["avgNodesPerColor"] = avgCalc( dataLuby[exp], "avgNodesPerColor")
		lubyDict["varNodesPerColor"] = avgCalc( dataLuby[exp], "varNodesPerColor")
		lubyDict["stdNodesPerColor"] = avgCalc( dataLuby[exp], "stdNodesPerColor")

		results[exp]["graph"] = graphDict
		results[exp]["Luby"] = lubyDict

	for exp in dataMcmcCpu:
		mcmcDict = {}
		mcmcDict["numColors"] = avgCalc( dataMcmcCpu[exp], "numColors")
		mcmcDict["usedColors"] = avgCalc( dataMcmcCpu[exp], "usedColors")
		mcmcDict["execTime"] = avgCalc( dataMcmcCpu[exp], "execTime")
		mcmcDict["avgNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "avgNodesPerColor")
		mcmcDict["varNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "varNodesPerColor")
		mcmcDict["stdNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "stdNodesPerColor")
		mcmcDict["performedIter"] = avgCalc( dataMcmcCpu[exp], "performedIter")

		results[exp]["MCMC_CPU"] = mcmcDict

	for exp in dataMcmcGpu:
		mcmcDict = {}
		mcmcDict["numColors"] = avgCalc( dataMcmcGpu[exp], "numColors")
		mcmcDict["usedColors"] = avgCalc( dataMcmcGpu[exp], "usedColors")
		mcmcDict["execTime"] = avgCalc( dataMcmcGpu[exp], "execTime")
		mcmcDict["avgNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "avgNodesPerColor")
		mcmcDict["varNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "varNodesPerColor")
		mcmcDict["stdNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "stdNodesPerColor")
		mcmcDict["performedIter"] = avgCalc( dataMcmcGpu[exp], "performedIter")

		results[exp]["MCMC_GPU"] = mcmcDict

	with open( basePath + '\\finalRes.json', "w") as outFile:
		json.dump( results, outFile )

if __name__ == '__main__':
	parseDirs()
	statMakerFromJson()

import os
import json

def clusterParser( logFile ):
	cluster = []
	for line in logFile:
		if 'Number of colors:' in line or 'Average number of nodes' in line or 'end_used_colors' in line:
			break
		itms = line.split( sep = ' ' )
		nn = int( itms[1] )
		cluster.append(nn)
	return line, cluster

def lineParser( logFile ):
	tempDict = {}
	for line in logFile:
		if 'Color histogram:' in line:
			line, tempDict["colorClusters"] = clusterParser( logFile )
		if 'Max deg:' in line:
			tempDict["maxDeg"] = int( line.split( sep = ' ')[2] )
			tempDict["minDeg"] = int( line.split( sep = ' ')[6] )
			tempDict["avgDeg"] = float( line.split( sep = ' ')[10] )
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
			tempDict["repet"] = int( line.split( sep = ' ')[1] )
		if 'Iteration performed:' in line:
			tempDict["performedIter"] = int( line.split( sep = ' ')[2] )
		if 'Execution time:' in line:
			tempDict["execTime"] = float( line.split( sep = ' ')[2] )
		if 'Number of colors:' in line:
			tempDict["numColors"] = int( line.split( sep = ' ')[3] )
		if 'Used colors:' in line:
			tempDict["usedColors"] = int( line.split( sep = ' ')[7] )
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
		if 'time ' in line:
			tempDict["execTime"] = float( line.split( sep = ' ')[1] )
			line, tempDict["colorClusters"] = clusterParser( logFile )
		if 'iteration_' in line:
			iterCount = iterCount + 1
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
	s = os.sep
	basePath = '.' + s + 'expMCMC'
	subDir = os.listdir( basePath )
	totResLuby = {}
	totResMCMC_CPU = {}
	totResMCMC_GPU = {}
	for currDir in subDir:
		if os.path.isfile( basePath + s + currDir ):
			continue
		filesInDir = os.listdir( basePath + s + currDir )
		lubyRes = []
		mcmcCpuRes = []
		mcmcGpuRes = []
		for currLog in filesInDir:
			numNodes = currLog.split( sep='-' )[0]
			prob = currLog.split( sep='-' )[1]

			if 'LUBY' in currLog:
				with open( basePath + s + currDir + s + currLog ) as logFile:
					tempItem = lineParser( logFile )
					tempItem["algo"] = "LUBY"
					lubyRes.append( tempItem )
			if 'MCMC' in currLog:
				with open( basePath + s + currDir + s + currLog ) as logFile:
					tempItem = lineParser( logFile )
					tempItem["algo"] = "MCMC_CPU"
					mcmcCpuRes.append( tempItem )
			if 'resultsFile' in currLog:
				with open( basePath + s + currDir + s + currLog ) as logFile:
					tempItem = mcmcGpuLineParser( logFile )
					tempItem["repet"] = int( (currLog.split( sep = '-')[3]).split( sep = '.')[0] )
					tempItem["algo"] = "MCMC_GPU"
					mcmcGpuRes.append( tempItem )

		if lubyRes:
			totResLuby[str(numNodes) + '-' + str(prob)] = lubyRes
		if mcmcCpuRes:
			totResMCMC_CPU[str(numNodes) + '-' + str(prob)] = mcmcCpuRes
		if mcmcGpuRes:
			totResMCMC_GPU[str(numNodes) + '-' + str(prob)] = mcmcGpuRes

	with open( basePath + s + 'outLuby.json', "w") as outFile:
		json.dump( totResLuby, outFile )
	with open( basePath + s + 'outMCMC_CPU.json', "w") as outFile:
		json.dump( totResMCMC_CPU, outFile )
	with open( basePath + s + 'outMCMC_GPU.json', "w") as outFile:
		json.dump( totResMCMC_GPU, outFile )

def avgCalc( exp, item ):
	itemAccum = 0
	for expRun in exp:
		itemAccum += expRun[item]
	return itemAccum / len( exp )

def statMakerFromJson():
	s = os.sep
	basePath = '.' + s + 'expMCMC'
	with open( basePath + s + "outLuby.json", "r" ) as read_file:
		dataLuby = json.load( read_file )
	with open( basePath + s + "outMCMC_CPU.json", "r") as read_file:
		dataMcmcCpu = json.load( read_file )
	with open( basePath + s + "outMCMC_GPU.json", "r") as read_file:
		dataMcmcGpu = json.load( read_file )

	results = {}

	for exp in dataLuby:
		lubyAvg = {}
		graphDict = {}
		graphDict["nNodes"] = dataLuby[exp][0]["nnodes"]
		graphDict["nEdges"] = dataLuby[exp][0]["nedges"]
		graphDict["edgeProb"] = dataLuby[exp][0]["edgeProb"]
		graphDict["maxDeg"] = dataLuby[exp][0]["maxDeg"]
		graphDict["minDeg"] = dataLuby[exp][0]["minDeg"]
		graphDict["avgDeg"] = dataLuby[exp][0]["avgDeg"]
		lubyAvg["numColors"] = avgCalc( dataLuby[exp], "numColors")
		lubyAvg["execTime"] = avgCalc( dataLuby[exp], "execTime")
		lubyAvg["avgNodesPerColor"] = avgCalc( dataLuby[exp], "avgNodesPerColor")
		lubyAvg["varNodesPerColor"] = avgCalc( dataLuby[exp], "varNodesPerColor")
		lubyAvg["stdNodesPerColor"] = avgCalc( dataLuby[exp], "stdNodesPerColor")

		results[exp] = {}
		results[exp]["graph"] = graphDict
		results[exp]["Luby"] = {}
		results[exp]["Luby"]["Exp"] = dataLuby[exp]
		results[exp]["Luby"]["Avg"] = lubyAvg
		for ee in results[exp]["Luby"]["Exp"]:
			ee.pop("nnodes")
			ee.pop("nedges")
			ee.pop("maxDeg")
			ee.pop("minDeg")
			ee.pop("avgDeg")
			ee.pop("edgeProb")

	for exp in dataMcmcCpu:
		mcmcDict = {}
		mcmcDict["numColors"] = avgCalc( dataMcmcCpu[exp], "numColors")
		mcmcDict["usedColors"] = avgCalc( dataMcmcCpu[exp], "usedColors")
		mcmcDict["execTime"] = avgCalc( dataMcmcCpu[exp], "execTime")
		mcmcDict["avgNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "avgNodesPerColor")
		mcmcDict["varNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "varNodesPerColor")
		mcmcDict["stdNodesPerColor"] = avgCalc( dataMcmcCpu[exp], "stdNodesPerColor")
		mcmcDict["performedIter"] = avgCalc( dataMcmcCpu[exp], "performedIter")

		results[exp]["MCMC_CPU"] = {}
		results[exp]["MCMC_CPU"]["Exp"] = dataMcmcCpu[exp]
		results[exp]["MCMC_CPU"]["Avg"] = mcmcDict
		for ee in results[exp]["MCMC_CPU"]["Exp"]:
			ee.pop("nnodes")
			ee.pop("nedges")
			ee.pop("maxDeg")
			ee.pop("minDeg")
			ee.pop("avgDeg")
			ee.pop("edgeProb")

	for exp in dataMcmcGpu:
		mcmcDict = {}
		mcmcDict["numColors"] = avgCalc( dataMcmcGpu[exp], "numColors")
		mcmcDict["usedColors"] = avgCalc( dataMcmcGpu[exp], "usedColors")
		mcmcDict["execTime"] = avgCalc( dataMcmcGpu[exp], "execTime")
		mcmcDict["avgNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "avgNodesPerColor")
		mcmcDict["varNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "varNodesPerColor")
		mcmcDict["stdNodesPerColor"] = avgCalc( dataMcmcGpu[exp], "stdNodesPerColor")
		mcmcDict["performedIter"] = avgCalc( dataMcmcGpu[exp], "performedIter")

		results[exp]["MCMC_GPU"] = {}
		results[exp]["MCMC_GPU"]["Exp"] = dataMcmcGpu[exp]
		results[exp]["MCMC_GPU"]["Avg"] = mcmcDict

	with open( basePath + s + 'finalRes.json', "w") as outFile:
		json.dump( results, outFile )

if __name__ == '__main__':
	parseDirs()
	statMakerFromJson()

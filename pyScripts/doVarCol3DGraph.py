import json
import sys
import os
#from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def calcQualcosa( exp, mode ):
	print( "qualcosa " + mode )
	return 0

def readDataFromJson( baseDir ):
	s = os.sep
	basePath = baseDir
	subDir = os.listdir( basePath )
	finalDict = {}
	for currDir in subDir:
		#print( basePath + s + currDir + s + "finalRes.json" )
		density = int( currDir )
		with open( basePath + s + currDir + s + "finalRes.json", "r" ) as read_file:
			dataFromJson = json.load( read_file )
			tempDict = {}
			for colorRatio in dataFromJson:
				for exp in dataFromJson["colorRatio"]:
					tempItem = {
						"n": exp["nNodes"],
						"p": exp["edgeProb"],
						"colorRatio": float( colorRatio ),
						"Luby": calcQualcosa( exp, 0 ),
						"MCPU": calcQualcosa( exp, 1 ),
						"MGPU": calcQualcosa( exp, 2 )
					}
					tempDict[str(colorRatio)] = tempItem
		finalDict[str(density)] = tempDict

	return finalDict



	fig = plt.figure()
	ax = plt.axes( projection='3d' )
	plt.show()






if __name__ == '__main__':
	filename = sys.argv[1]
	dataFromDirs = readDataFromJson( filename )

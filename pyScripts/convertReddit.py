import os
import sys

def convertReddit(infilename, outfilename):
	with open(outfilename, "w") as outF:
		with open(infilename, "r") as inF:
			for line in inF:
				sss = line.strip('\n').split(sep=',')
				outF.write(sss[0] + ' ' + sss[1] + ' 0.1\n')
				inF.readline()

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python3 testSelfArcs.py <inputFile> [outputFile]")
		exit(-1)
	infilename = sys.argv[1]
	if len(sys.argv) > 2:
		outfilename = sys.argv[2]
	else:
		ll = infilename.split(sep=os.sep)
		tt = ll[-1].split(sep='.')
		tout = tt[0] + '_tempconv'
		tt[0] = tout
		newtt = ".".join(tt)
		ll[-1] = newtt
		outfilename = os.sep.join(ll)
	convertReddit(infilename, outfilename)

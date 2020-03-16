import os
import sys

def removeSelfArcs(infilename, outfilename):
	cnt = 0
	with open(outfilename, "w") as outF:
		with open(infilename, "r") as inF:
			l = inF.readline()		# Jump header
			outF.write(l)
			for line in inF:
				sss = line.split(sep=' ')
				if sss[0] != sss[1]:
					outF.write(line)
				else:
					cnt = cnt + 1
	print('stripped {} self arcs'.format(cnt))

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
		tout = tt[0] + '_purged'
		tt[0] = tout
		newtt = ".".join(tt)
		ll[-1] = newtt
		outfilename = os.sep.join(ll)
	# print(infilename)
	# print(outfilename)
	removeSelfArcs(infilename, outfilename)

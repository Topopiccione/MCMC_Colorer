#!/bin/bash

EXECNAME="MCMC_Colorer"
DENSITY_LIST='200 250 300 350'
GRAPHNODES_LIST='100000 200000 300000'
NUMCOLRATIO_LIST='1.0'

for dens in $DENSITY_LIST
do
	echo "Density = $dens"
	if [ ! -d "$dens" ]; then
		mkdir $dens
	fi

	cd $dens

	for graph_nodes in $GRAPHNODES_LIST
	do
		probab=$(bc <<< "scale=6; $dens/ $graph_nodes")
		for colRatio in $NUMCOLRATIO_LIST
		do
			commandLine="../$EXECNAME --simulate $probab -n $graph_nodes --numColRatio $colRatio --repet 4 --tabooIteration 4"
			$commandLine
		done
	done

	cd ..

done

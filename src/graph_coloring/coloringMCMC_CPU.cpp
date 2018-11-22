#include "coloringMCMC_CPU.h"
#include "utils/dbg.h"
#include "utils/miscUtils.h"

extern dbg * g_debugger;

template<typename nodeW, typename edgeW>
ColoringMCMC_CPU<nodeW, edgeW>::ColoringMCMC_CPU(Graph<nodeW, edgeW>* g, ColoringMCMCParams params, uint32_t seed) :
	Colorer<nodeW, edgeW>(g), str(g->getStruct()), nNodes(g->getStruct()->nNodes), nCol(params.nCol), lambda(params.lambda),
	epsilon(params.epsilon), ratioFreezed(params.ratioFreezed), seed(seed) {

	std::cout << TXT_BIGRN << "** MCMC CPU colorer **" << TXT_NORML << std::endl;

	LOG(TRACE) << TXT_COLMC << "Creating ColorerMCMC with parameters: nCol= " << nCol << " - lambda= " << lambda
		<< " - epsilon= " << epsilon << " - ratioFreezed= " << ratioFreezed << " - seed= " << seed << TXT_NORML;

	C = std::vector<uint32_t>(nNodes);
	Cstar = std::vector<uint32_t>(nNodes);
	LOG(TRACE) << TXT_COLMC << "C: allocated " << C.size() << " x " << sizeof(uint32_t) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "Cstar: allocated " << Cstar.size() << " x " << sizeof(uint32_t) << TXT_NORML;

	q = std::vector<float>(nNodes);
	qstar = std::vector<float>(nNodes);
	LOG(TRACE) << TXT_COLMC << "q: allocated " << q.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "qstar: allocated " << qstar.size() << " x " << sizeof(float) << TXT_NORML;

	p = std::vector<float>(nCol);
	pstar = std::vector<float>(nCol);
	LOG(TRACE) << TXT_COLMC << "p: allocated " << p.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "pstar: allocated " << pstar.size() << " x " << sizeof(float) << TXT_NORML;

	nodeProbab = std::vector<float>(nNodes);
	LOG(TRACE) << TXT_COLMC << "nodeProbab: allocated " << nodeProbab.size() << " x " << sizeof(float) << TXT_NORML;

	freezed = std::vector<bool>(nNodes);
	LOG(TRACE) << TXT_COLMC << "freezed: allocated " << freezed.size() << " x " << sizeof(bool) << TXT_NORML;

	freeColors = std::vector<bool>(nCol);
	LOG(TRACE) << TXT_COLMC << "freeColors: allocated " << freeColors.size() << " x " << sizeof(bool) << TXT_NORML;

	Cviols = std::vector<bool>(nNodes);
	Cstarviols = std::vector<bool>(nNodes);
	LOG(TRACE) << TXT_COLMC << "Cviols: allocated " << Cviols.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "Cstarviols: allocated " << Cstarviols.size() << " x " << sizeof(bool) << TXT_NORML;

	colorIdx = std::vector<size_t>( nCol );

	// Starting the RNGs
	gen = std::default_random_engine(seed);
	unifInitColors = std::uniform_int_distribution<uint32_t>(0, nCol - 1);
	unifDistr = std::uniform_real_distribution<float>(0, 1);
	//bernieFreeze = std::bernoulli_distribution(ratioFreezed);

	// Begin with a random coloration (colors range = [0, nCol-1])
	size_t idx = 0;
	////////////// Baseline
	//std::for_each(std::begin(C), std::end(C), [&](uint32_t &val) {val = unifInitColors(gen); });
	////////////// Non-uniform probabilities: linear
	// Filling p with color distribution
	// divider = nCol * (nCol + 1);
	// std::for_each(std::begin(p), std::end(p), [&](float &val) {
	// 	val = 2.0f * (float)(nCol - idx) / divider;
	// 	idx++;
	// } );
	// //Extract in advance all the experiment probabilities
	// std::for_each(std::begin(nodeProbab), std::end(nodeProbab), [&](float &val) {val = unifDistr(gen); });
	// // Run extract_new_color on each node (q vector in ignored)
	// for (idx = 0; idx < nNodes; idx++)
	// 	extract_new_color(idx, p, nodeProbab, q, C);
	////////////// Non-uniform probabilities: negative exp
	expLambda = 0.1f;
	auto expEval = [&] (size_t i) {return exp(-expLambda * (float(i)));};
	std::for_each( std::begin(p), std::end(p), [&](float &val) { val = expEval(idx); idx++; } );
	divider = std::accumulate( std::begin(p), std::end(p), 0.0f );
	idx = 0;
	std::for_each( std::begin(p), std::end(p), [&](float &val) { val /= divider; } );
	//Extract in advance all the experiment probabilities
	std::for_each(std::begin(nodeProbab), std::end(nodeProbab), [&](float &val) {val = unifDistr(gen); });
	// Run extract_new_color on each node (q vector in ignored)
	for (idx = 0; idx < nNodes; idx++)
		extract_new_color(idx, p, nodeProbab, q, C);
	//////////////

}

template<typename nodeW, typename edgeW>
ColoringMCMC_CPU<nodeW, edgeW>::~ColoringMCMC_CPU() {
	LOG(TRACE) << TXT_COLMC << "Cstarviols: freeing " << Cstarviols.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "Cviols: freeing " << Cviols.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "nodeProbab: freeing " << nodeProbab.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "freeColors: freeing " << freeColors.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "freezed: freeing " << freezed.size() << " x " << sizeof(bool) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "qstar: freeing " << qstar.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "q: freeing " << q.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "pstar: freeing " << pstar.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "p: freeing " << p.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "Cstar: freeing " << Cstar.size() << " x " << sizeof(uint32_t) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "C: freeing " << C.size() << " x " << sizeof(uint32_t) << TXT_NORML;
}

template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::run() {
	LOG(TRACE) << TXT_COLMC << "Starting MCMC coloring..." << TXT_NORML;
	auto bernie = [&](float p) {return ((double)rand() / (RAND_MAX)) >= p ? 0 : 1; };

	//std::vector<size_t> colorIdx( nCol );
	std::vector<size_t> histBins( nCol );
	//size_t ii = 0;
	//std::for_each( std::begin(colorIdx), std::end(colorIdx), [&](size_t &val) {val = ii++;} );

	size_t iter = 0;
	size_t maxiter = 250;

	// Count the number of violations on the extracted coloring
	// Remember: 1: node is in a current violation state, 0: is not
	Cviol = violation_count(C, Cviols);
	LOG(TRACE) << TXT_BIBLU << "Initial violations: " << Cviol << TXT_NORML;

	// Stay in the loop until there are no more violations
	while (Cviol != 0) {
		LOG(TRACE) << TXT_BIRED << "iteration " << iter << TXT_NORML;
		// Extract in advance all the experiment probabilities
		std::for_each(std::begin(nodeProbab), std::end(nodeProbab), [&](float &val) {val = unifDistr(gen); });

		///////// Reordering color depending on the histogram
		// std::fill( std::begin(histBins), std::end(histBins), 0 );
		// size_t ii = 0;
		// std::for_each( std::begin(colorIdx), std::end(colorIdx), [&](size_t &val) {val = ii++;} );
		// std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBins[val]++;} );
		// std::sort( std::begin(colorIdx), std::end(colorIdx), [&](int i,int j) {return histBins[i] > histBins[j]; } );
		///////// No reorder
		size_t ii = 0;
		std::for_each( std::begin(colorIdx), std::end(colorIdx), [&](size_t &val) {val = ii++;} );
		/////////

		Cviol = violation_count(C, Cviols);
		LOG(TRACE) << TXT_BIBLU << "C violations: " << Cviol << TXT_NORML;

		show_histogram();

		if ((g_traceLogEn) & (Cviol < 40)) {
			size_t idx = 0;
			std::string logString;
			logString = "Violating nodes: ";
			std::for_each(Cviols.begin(), Cviols.end(), [&](bool val) {if (val) logString = logString + std::to_string(idx) + " "; idx++; });
			LOG(TRACE) << TXT_BIBLU << logString.c_str() << TXT_NORML;
		}

		// Managed freezing nodes.
		// Remember: 0 = evaluate new color, 1 = freezed
		//float freezingProb = 0.5f - exp( (-(int32_t)Cviol) / (float) nNodes ) * 0.5f;
		//float freezingProb = 0.2f + exp( (-(int32_t)Cviol) / (float) nNodes ) / 2.5f;
		float freezingProb = 0;
		LOG(TRACE) << TXT_BIBLU << "freezingProb: " << freezingProb << TXT_NORML;
		for (size_t i = 0; i < freezed.size(); i++) {
			if (!Cviols[i])
				freezed[i] = bernie(freezingProb);
			else
				freezed[i] = 0;
		}
		if (g_traceLogEn) {
			size_t tempVal = std::count(std::begin(freezed), std::end(freezed), 1);
			LOG(TRACE) << TXT_BIBLU << "Number of freezed nodes: " << tempVal << TXT_NORML;
		}

		// reset stats...
		Zvcomp_max = Zvcomp_avg = 0;
		Zvcomp_min = nCol + 1;

		// Iternal loop 1: building Cstar
		for (size_t i = 0; i < nNodes; i++) {
			if (freezed[i]) {
				Cstar[i] = C[i];
				q[i] = 1.0f;
				continue;
			}

			// Build list of free / occupied colors (and count them)
			size_t Zvcomp = count_free_colors(i, C, freeColors);
			size_t Zv = nCol - Zvcomp;

			// Fill vect p.
			fill_p(i, Zv);

			// consider nodeProbab, build the CdF of p, get the new color and put it in Cstar[i]
			extract_new_color(i, p, nodeProbab, q, Cstar);

			// update stats
			Zvcomp_avg += Zvcomp;
			Zvcomp_max = (Zvcomp > Zvcomp_max) ? Zvcomp : Zvcomp_max;
			Zvcomp_min = (Zvcomp < Zvcomp_min) ? Zvcomp : Zvcomp_min;
		}

		// Log stats
		LOG(TRACE) << TXT_BIBLU << "Zv max: " << Zvcomp_max << " - Zv min: " << Zvcomp_min << " - Zv avg: "
			<< Zvcomp_avg / (float)nNodes << TXT_NORML;

		// Count the violation of the new coloring Cstar; also fill the Cstarviols vector
		Cstarviol = violation_count(Cstar, Cstarviols);
		LOG(TRACE) << TXT_BIBLU << "Cstar violations: " << Cstarviol << TXT_NORML;

		// Internal loop 2: building C
		for (size_t i = 0; i < nNodes; i++) {
			if (freezed[i]) {
				qstar[i] = 1.0f;
				continue;
			}
			// Build list of free / occupied colors (and count them)
			size_t Zvcomp = count_free_colors(i, Cstar, freeColors);
			size_t Zv = nCol - Zvcomp;

			// No need to do the experiment, just evaluate qstar[i]
			fill_qstar(i, Zv, Cstar, C, freeColors, Cstarviols, q);
		}

		// Apply log() to q and qstar; then, cumulate
		std::for_each(std::begin(q), std::end(q), [](float & val) {val = log(val); });
		std::for_each(std::begin(qstar), std::end(qstar), [](float & val) {val = log(val); });
		float sumlogq = std::accumulate(std::begin(q), std::end(q), 0.0f);
		float sumlogqstar = std::accumulate(std::begin(qstar), std::end(qstar), 0.0f);

		// evaluate alpha
		alpha = lambda * ((int64_t)Cviol - (int64_t)Cstarviol) - sumlogq + sumlogqstar;
		LOG(TRACE) << TXT_BIBLU << "Cviol: " << Cviol << " - Cviolstar: " << Cstarviol << " - sumlogq: " << sumlogq
			<< " - sumlogqstar: " << sumlogqstar << TXT_NORML;
		LOG(TRACE) << TXT_BIBLU << "alpha: " << alpha << TXT_NORML;

		if (g_debugger->check_F12keypress())
			g_debugger->stop_and_debug();

		// Consider min(alpha, 0); execute an experiment against alpha to accept the new coloring
		auto minAlpha = std::min(alpha, 0.0f);
		if (alpha != 0) {
			if (bernie(minAlpha)) {
				LOG(TRACE) << "Rejecting coloration!" << std::endl;
			}
			else {
				std::swap(C, Cstar);
				Cviol = Cstarviol;
			}
		}

		iter++;
		if (iter > maxiter) {
			LOG(TRACE) << TXT_BIRED << "Maximum iteration reached: exiting..." << TXT_NORML;
			break;
		}
	}
}


// Arguments:
//		currentColoring -> the coloring array being processed; can be C or Cstar
//		violations -> boolean array stating if the current color of a node is currently violating an admissible coloration
// Returns: total number of nodes whose color is in a violation state
template<typename nodeW, typename edgeW>
size_t ColoringMCMC_CPU<nodeW, edgeW>::violation_count(const std::vector<uint32_t> & currentColoring, std::vector<bool> & violations) {
	size_t viol = 0;
	for (size_t i = 0; i < nNodes; i++) {
		violations[i] = 0;
		size_t nodeViolations = 0;
		// Get the node color
		const uint32_t nodeColor = currentColoring[i];
		// Get the index and the number of the neighbours of node i
		const size_t nodeDeg = str->cumulDegs[i + 1] - str->cumulDegs[i];
		const size_t neighIdx = str->cumulDegs[i];
		const node * neighPtr = str->neighs + neighIdx;

		// Scan all the neighbours color and count the violations
		nodeViolations = std::count_if(neighPtr, neighPtr + nodeDeg,
			[&](node neigh) {return nodeColor == currentColoring[neigh]; });

		if (nodeViolations > 0) {
			violations[i] = 1;
			viol++;
		}
	}
	return viol;
}


// This is executed for each node
// Arguments:
//		currentNode: the current node under processing
//		currentColoring: the coloring vector that the current node color has to be tested against; can be C or Cstar
// Returns:
//		freeColors: a boolean array (size == nCol) which states is the i-th color is free (1) or not (0)
//		Also returns the number of free colors
template<typename nodeW, typename edgeW>
size_t ColoringMCMC_CPU<nodeW, edgeW>::count_free_colors(const size_t currentNode, const std::vector<uint32_t> & currentColoring,
	std::vector<bool> & freeColors) {

	// Resets the freeColors array: all colors are free at the beginning
	std::fill(std::begin(freeColors), std::end(freeColors), 1);

	// Get node current color
	const uint32_t nodeColor = currentColoring[currentNode];
	// Get node deg and neighs
	const size_t nodeDeg = str->cumulDegs[currentNode + 1] - str->cumulDegs[currentNode];
	const size_t neighIdx = str->cumulDegs[currentNode];
	const node * neighPtr = str->neighs + neighIdx;

	// Scan the neighborhood, and set 0 to all conflicting colors
	for (size_t i = 0; i < nodeDeg; i++) {
		node neigh = neighPtr[i];
		freeColors[currentColoring[neigh]] = 0;
	}

	// Returns the number of free colors
	return std::count(std::begin(freeColors), std::end(freeColors), 1);
}


// This is executed for each node
// Arguments:
//		currentNode: guess what
//		Zv: number of conflicting colors for of the current node
// Returns:
//		nothing, but modifies p vector
template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::fill_p(const size_t currentNode, const size_t Zv) {
	size_t idx = 0;
	const size_t Zvcomp = nCol - Zv;
	const uint32_t currentColor = C[currentNode];

	// Scan the p vector (size = nCol) and fill with probabilities
	// Discriminate wheter current node color is in conflict or not
	if (Cviols[currentNode] == 1) { // Current color is in a violation state
		// Handle the special case of nodes conflicting and not having free colors to choose from
		auto nFreeColors = std::accumulate(std::begin(freeColors), std::end(freeColors), 0);
		if (nFreeColors == 0) {
			std::for_each(std::begin(p), std::end(p), [&](float &val) {
				if (idx == currentColor)
					val = 1.0f - (nCol - 1) * epsilon;
				else
					val = epsilon;
				idx++;
			});
			return;
		}
		////////// Baseline
		// std::for_each(std::begin(p), std::end(p), [&](float &val) {
		// 	if (freeColors[idx])
		// 		val = (1.0f - epsilon * Zv) / (float)Zvcomp;
		// 	else
		// 		val = epsilon;
		// 	idx++;
		// });
		////////// Non-uniform probabilities: linear
		// float reminder = 0;
		// size_t nOccup = 0;
		// // Start filling the probabilites and accumulating the reminder
		// idx = 0;
		// std::for_each(std::begin(p), std::end(p), [&](float &val) {
		// 	if (freeColors[colorIdx[idx]])
		// 		//val = 2.0f * (float)(idx + 1) / divider;
		// 		val = 2.0f * (float)(nCol - colorIdx[idx]) / divider;
		// 	else {
		// 		val = epsilon;
		// 		//reminder += (2.0f * (float)(idx + 1) / divider) - epsilon;
		// 		reminder += (2.0f * (float)(nCol - colorIdx[idx]) / divider) - epsilon;
		// 		nOccup++;
		// 	}
		// 	idx++;
		// });
		// // Redistributing the reminder on the free colors
		// idx = 0;
		// std::for_each(std::begin(p), std::end(p), [&](float &val) {
		// 	if (freeColors[colorIdx[idx]])
		// 		val += (reminder / (float) (nCol - nOccup));
		// 	idx++;
		// });
		/////////// Non-uniform probabilities: negative exponential
		float reminder = 0;
		size_t nOccup = 0;
		idx = 0;
		auto expEval = [&] (size_t i) {return exp(-expLambda * (float(i)));};
		// Start filling the probabilites and accumulating the reminder
		idx = 0;
		std::for_each(std::begin(p), std::end(p), [&](float &val) {
			if (freeColors[colorIdx[idx]])
				val = 2.0f * expEval(idx) / divider;
			else {
				val = epsilon;
				reminder += (2.0f * expEval(idx) / divider) - epsilon;
				nOccup++;
			}
			idx++;
		});
		// Redistributing the reminder on the free colors
		idx = 0;
		std::for_each(std::begin(p), std::end(p), [&](float &val) {
			if (freeColors[colorIdx[idx]])
				val += (reminder / (float) (nCol - nOccup));
			idx++;
		});

	}
	else { // Current color is NOT in a violation state
		std::for_each(std::begin(p), std::end(p), [&](float &val) {
			if (colorIdx[idx] == currentColor)
				val = 1.0f - (nCol - 1) * epsilon;
			else
				val = epsilon;
			idx++;
		});
	}
	////////////////
}


// This is executed for each node
// Arguments:
//		currentNode: guess what
//		pVect: vector of probability (size = nCol) assembled by fill_p()
//		experimentVect: vector of probability (size = nNodes) containing the experiment
// Returns:
//		qVect: modified in the currentNode position with p[extractedColor]
//		newColoring: modified in the currentNode position with the new extracted color. Must be Cstar.
template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::extract_new_color(const size_t currentNode, const std::vector<float> & pVect,
	const std::vector<float> & experimentVect, std::vector<float> & qVect, std::vector<uint32_t> & newColoring) {

	float experimentThrsh = experimentVect[currentNode];
	float cdf = 0;
	size_t idx;

	// Walk through pVect while building the cumulative distribution function
	// Exit from the cycle as soon as cdf gets > experiemntThrs. Save the index that is the new color for Cstar
	for (idx = 0; idx < pVect.size(); idx++) {
		cdf += pVect[idx];
		if (cdf > experimentThrsh)
			break;
	}

	qVect[currentNode] = pVect[idx];
	newColoring[currentNode] = idx;
}


template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::fill_qstar(const size_t currentNode, const size_t Zv, const std::vector<uint32_t> & newColoring,
	const std::vector<uint32_t> & oldColoring, const std::vector<bool> & freeCols, const std::vector<bool> & newColoringViols,
	std::vector<float> & qVect) {

	const size_t Zvcomp = nCol - Zv;
	const uint32_t currentColor = newColoring[currentNode];

	if (newColoringViols[currentNode] == 1) { // New color is in a violation state
		if (freeCols[currentColor])
			qstar[currentNode] = (1.0f - epsilon * Zv) / (float)Zvcomp;
		else
			qstar[currentNode] = epsilon;
	}
	else { // New color is NOT in a violation state
		if (newColoring[currentNode] == oldColoring[currentNode])
			qstar[currentNode] = 1.0f - (nCol - 1) * epsilon;
		else
			qstar[currentNode] = epsilon;
	}
}

template<typename nodeW, typename edgeW>
void ColoringMCMC_CPU<nodeW, edgeW>::show_histogram() {
	std::vector<size_t> histBins(nCol, 0);
	std::vector<size_t> histBinsOcc(nCol, 0);
	// Initial check: each value should be 0 <= val < nCol
	size_t idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { if ((val < 0) | (val >= nCol)) std::cout << "Error in color " << idx << ". Exiting..." << std::endl; idx++;} );

	// Filling histogram bins
	idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBinsOcc[val] += (Cviols[idx++] == 1);} );
	idx = 0;
	std::for_each( std::begin(C), std::end(C), [&](uint32_t val) { histBins[val] += (Cviols[idx++] == 0);} );

	// finding the correct scaler, so that the maximum bin get printed 80 times (ignoring rounding errors...)
	std::vector<size_t> tempVec(nCol);
	idx = 0;
	std::for_each( std::begin(tempVec), std::end(tempVec), [&](size_t &val) { val = histBinsOcc[idx] + histBins[idx]; idx++;} );
	auto max_val = std::max_element( std::begin(tempVec), std::end(tempVec) );
	size_t scaler = *max_val / 80;

	// Printing the histogram.
	for (size_t i = 0; i < histBins.size(); i++) {
		std::cout << i << ": ";

		size_t howMany = histBins[i] / scaler;
		std::cout << TXT_BIGRN;
		for (size_t j = 0; j < howMany; j++)
			std::cout << "*";

		howMany = histBinsOcc[i] / scaler;
		std::cout << TXT_BIRED;
		for (size_t j = 0; j < howMany; j++)
			std::cout << "*";

		std::cout << TXT_NORML << std::endl;
	}
	std::cout << "Each '*' = " << scaler << " nodes" << std::endl;
}


/////////////////////
template class Graph<float, float>;
template class ColoringMCMC_CPU<float, float>;

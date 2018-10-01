#include "coloringMCMC_CPU.h"

template<typename nodeW, typename edgeW>
ColoringMCMC_CPU<nodeW, edgeW>::ColoringMCMC_CPU( Graph<nodeW, edgeW>* g, ColoringMCMCParams params, uint32_t seed ) :
		Colorer<nodeW, edgeW>( g ), nNodes( g->getStruct()->nNodes ), nCol( params.nCol ), lambda( params.lambda ),
		epsilon( params.epsilon ), ratioFreezed( params.ratioFreezed ), seed( seed ) {

	LOG(TRACE) << TXT_COLMC << "Creating ColorerMCMC with parameters: nCol= " << nCol << " - lambda= " << lambda
		<< " - epsilon= " << epsilon << " - ratioFreezed= " << ratioFreezed << " - seed= " << seed << TXT_NORML;

	C = std::vector<uint32_t>( nNodes );
	Cstar = std::vector<uint32_t>( nNodes );
	LOG(TRACE) << TXT_COLMC << "C: allocated " << C.size() << " x " << sizeof(uint32_t) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "Cstar: allocated " << Cstar.size() << " x " << sizeof(uint32_t) << TXT_NORML;

	q = std::vector<float>( nNodes );
	qstar = std::vector<float>( nNodes );
	LOG(TRACE) << TXT_COLMC << "q: allocated " << q.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "qstar: allocated " << qstar.size() << " x " << sizeof(float) << TXT_NORML;

	p = std::vector<float>( nNodes );
	pstar = std::vector<float>( nNodes );
	LOG(TRACE) << TXT_COLMC << "p: allocated " << p.size() << " x " << sizeof(float) << TXT_NORML;
	LOG(TRACE) << TXT_COLMC << "pstar: allocated " << pstar.size() << " x " << sizeof(float) << TXT_NORML;

	freezed = std::vector<bool>( nNodes );
	LOG(TRACE) << TXT_COLMC << "freezed: allocated " << freezed.size() << " x " << sizeof(bool) << TXT_NORML;

	freeColors = std::vector<bool>( nCol );
	LOG(TRACE) << TXT_COLMC << "freeColors: allocated " << freeColors.size() << " x " << sizeof(bool) << TXT_NORML;

	nodeProbab = std::vector<float>( nNodes );
	LOG(TRACE) << TXT_COLMC << "nodeProbab: allocated " << nodeProbab.size() << " x " << sizeof(float) << TXT_NORML;

	// Starting the RNGs
	gen				= std::default_random_engine( seed );
	unifInitColors	= std::uniform_int_distribution<uint32_t>( 0, nCol - 1 );
	unifDistr		= std::uniform_real_distribution<float>( 0, 1 );
	bernieFreeze	= std::bernoulli_distribution( ratioFreezed );

	// Begin with a random coloration
	// Colors range = [0, nCol-1]
	std::for_each( std::begin(C), std::end(C), [&](uint32_t &val) {val = unifInitColors( gen );} );

}

template<typename nodeW, typename edgeW>
ColoringMCMC_CPU<nodeW, edgeW>::~ColoringMCMC_CPU() {
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

	// Stay in the loop until there are no more violations
	while (Cstarviol != 0) {
		// Reset variables...
		std::fill( std::begin(freeColors), std::end(freeColors), 0 );

		// Extract in advance all the experiment probabilities
		std::for_each( std::begin(nodeProbab), std::end(nodeProbab), [&] (float &val) {val = unifDistr(gen);} );

		// Iternal loop 1: building Cstar
		for (size_t i = 0; i < nNodes; i++) {
			// Build list of free / occupied colors (and count them)
			/// for_each... freeColors = is_color_free(...); // 0 = occupied, 1 = free
			size_t Az = std::count( std::begin(freeColors), std::end(freeColors), 0 );
			size_t Azcomp = nCol - Az;

			// Fill vect p. Depends on C, freeColors
			size_t idx = 0;
			/// fill_p(...)

			// consider nodeProbab[i], build the CdF of p, get the new color and put it in Cstar[i]
			// Save qi] for later use
		}

		// Internal loop 2: building C
		for (size_t i = 0; i < nNodes; i++) {
			// Build list of free / occupied colors (and count them)
			/// for_each... freeColors = is_color_free(...); // 0 = occupied, 1 = free
			size_t Azstar = std::count( std::begin(freeColors), std::end(freeColors), 0 );
			size_t Azstarcomp = nCol - Azstar;

			// No need to do the experiment, just evaluate qstar[i]
		}

		// Count the number of conflicts on both C and Cstar
		// ...

		// Cumulate q and qstar
		// ...

		// evaluate alpha
		// ...




	}



}



/////////////////////
template class ColoringMCMC_CPU<float, float>;

/*
 * AdaptiveMixtureModelSegmetation.h
 *
 *  Created on: Dec 1, 2012
 *      Author: avo
 */

#ifndef ADAPTIVEMIXTUREMODELSEGMETATION_H_
#define ADAPTIVEMIXTUREMODELSEGMETATION_H_

#include "filter.hpp"

class AdaptiveMixtureModelSegmetation : public Filter {

	enum Input
	{
		WorldCoordinates,
	};

	unsigned int n_view;

public:
	AdaptiveMixtureModelSegmetation();
	virtual ~AdaptiveMixtureModelSegmetation();

	void init();
	void execute();
};

#endif /* ADAPTIVEMIXTUREMODELSEGMETATION_H_ */

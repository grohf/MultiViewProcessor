/*
 * DFPFHEstimator.h
 *
 *  Created on: Sep 25, 2012
 *      Author: avo
 */

#ifndef DFPFHESTIMATOR_H_
#define DFPFHESTIMATOR_H_

#include "feature.hpp"

class DFPFHEstimator : public Feature {
public:
	DFPFHEstimator();
	virtual ~DFPFHEstimator();

	void init();
	void execute();
};


#endif /* DFPFHESTIMATOR_H_ */

/*
 * SVDEstimatorCPU.h
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#ifndef SVDESTIMATORCPU_H_
#define SVDESTIMATORCPU_H_

#include "feature.hpp"

class SVDEstimator_CPU : public Feature {
public:
	SVDEstimator_CPU();
	virtual ~SVDEstimator_CPU();

	void init();
	void execute();
};

#endif /* SVDESTIMATORCPU_H_ */

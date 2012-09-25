/*
 * FPFHEstimator2.h
 *
 *  Created on: Sep 25, 2012
 *      Author: avo
 */

#ifndef FPFHESTIMATOR2_H_
#define FPFHESTIMATOR2_H_

#include "feature.hpp"

template<unsigned int featureBins>
class FPFHEstimator2 : public Feature {
public:
	FPFHEstimator2();
	virtual ~FPFHEstimator2();

	void init();
	void execute();
};

#endif /* FPFHESTIMATOR2_H_ */

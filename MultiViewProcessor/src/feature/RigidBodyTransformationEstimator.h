/*
 * RigidBodyTransformationEstimator.h
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#ifndef RIGIDBODYTRANSFORMATIONESTIMATOR_H_
#define RIGIDBODYTRANSFORMATIONESTIMATOR_H_

#include "feature.hpp"

class RigidBodyTransformationEstimator : public Feature {

	dim3 grid;
	dim3 block;

public:
	RigidBodyTransformationEstimator() { }
	virtual ~RigidBodyTransformationEstimator() { }

	void init();
	void execute();
};

#endif /* RIGIDBODYTRANSFORMATIONESTIMATOR_H_ */

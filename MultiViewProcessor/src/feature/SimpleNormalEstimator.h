/*
 * SimpleNormalEstimator.h
 *
 *  Created on: Dec 7, 2012
 *      Author: avo
 */

#ifndef SIMPLENORMALESTIMATOR_H_
#define SIMPLENORMALESTIMATOR_H_

#include "feature.hpp"

class SimpleNormalEstimator : public Feature {

	unsigned int n_view;

	enum Target
	{
		Normals
	};

	enum Input
	{
		WorldCoordinates
	};

public:
	SimpleNormalEstimator(unsigned int n_view);
	virtual ~SimpleNormalEstimator();

	void init();
	void execute();

	DeviceDataInfoPtr getNormals()
	{
		return getTargetData(Normals);
	}
	void setWorldCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,WorldCoordinates);
	}
};

#endif /* SIMPLENORMALESTIMATOR_H_ */

/*
 * NormalPCAEstimator.h
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#ifndef NORMALPCAESTIMATOR_H_
#define NORMALPCAESTIMATOR_H_

#include "feature.hpp"

class NormalPCAEstimator : public Feature {

	dim3 grid;
	dim3 block;

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

	NormalPCAEstimator(unsigned int n_view_);
	virtual ~NormalPCAEstimator();

	void execute();
	void init();

	DeviceDataInfoPtr getNormals()
	{
		return getTargetData(Normals);
	}
	void setWorldCoordinates(DeviceDataInfoPtr ddip){
		addInputData(ddip,WorldCoordinates);
	}
};

#endif /* NORMALPCAESTIMATOR_H_ */

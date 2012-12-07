/*
 * GlobalFineRegistrationEstimator.h
 *
 *  Created on: Dec 7, 2012
 *      Author: avo
 */

#ifndef GLOBALFINEREGISTRATIONESTIMATOR_H_
#define GLOBALFINEREGISTRATIONESTIMATOR_H_

#include "feature.hpp"

class GlobalFineRegistrationEstimator : public Feature {

	enum Input
	{
		WorldCoordinates,
		Normals,
		GlobalTransformationMatrices,
		SensorInfoListV2,
	};

	enum Target
	{
		FittedGlobalTransformationMatrices,
	};

	unsigned int n_view;

public:
	GlobalFineRegistrationEstimator(unsigned int n_view);
	virtual ~GlobalFineRegistrationEstimator();

	void init();
	void execute();

	void setPointCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,WorldCoordinates);
	}

	void setPointNormals(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,Normals);
	}

	void setTransformationmatrices(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,GlobalTransformationMatrices);
	}

	void setSensorInfoList(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,SensorInfoListV2);
	}

	DeviceDataInfoPtr getTransformationmatrices()
	{
		return getTargetData(FittedGlobalTransformationMatrices);
	}


};

#endif /* GLOBALFINEREGISTRATIONESTIMATOR_H_ */

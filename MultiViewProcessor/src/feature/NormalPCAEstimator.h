/*
 * NormalPCAEstimator.h
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#ifndef NORMALPCAESTIMATOR_H_
#define NORMALPCAESTIMATOR_H_

#include <thrust/host_vector.h>
#include "feature.hpp"

class NormalPCAEstimator : public Feature {

	dim3 grid;
	dim3 block;
	float dist_thresh;

	unsigned int n_view;
	unsigned int outputlevel;

	enum Target
	{
		Normals
	};

	enum Input
	{
		WorldCoordinates
	};

public:

	NormalPCAEstimator(unsigned int n_view_,float dist_thresh_,unsigned int outputlevel=0);
	virtual ~NormalPCAEstimator();

	void execute();
	void init();

	DeviceDataInfoPtr getNormals()
	{
		return getTargetData(Normals);
	}
	void setWorldCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,WorldCoordinates);
	}


	void TestNormalCreater(thrust::host_vector<float4>& views,thrust::host_vector<float4>& normals);
};

#endif /* NORMALPCAESTIMATOR_H_ */

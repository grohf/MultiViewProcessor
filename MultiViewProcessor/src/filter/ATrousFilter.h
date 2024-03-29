/*
 * ATrousFilter.h
 *
 *  Created on: Aug 25, 2012
 *      Author: avo
 */

#ifndef ATROUSFILTER_H_
#define ATROUSFILTER_H_

#include <thrust/host_vector.h>
#include "filter.hpp"


class ATrousFilter : public Filter {

	dim3 grid;
	dim3 block;
	unsigned int radi;

	unsigned int atrous_radi;
	unsigned int atrous_length;

	unsigned int iterations;
	float sigma_depth;
	float sigma_intensity;

	unsigned int n_view;
	int outputlevel;
	enum Input
	{
		WorldCoordinates,
		SensorInfoList,
		PointIntensity,
	};

	enum Target
	{
		FilteredWorldCoordinates,
	};

public:
	ATrousFilter(unsigned int n_view_, unsigned int iteration_, float sigma_depth_, float sigma_intensity_, int outputlevel = 0) :
		iterations(iteration_), sigma_depth(sigma_depth_), sigma_intensity(sigma_intensity_), outputlevel(outputlevel)
	{
		n_view = n_view_;
		radi = 2;


		//	DeviceDataParamFunction f = shrinkData;
		//	addParamChanger(f);
	};
	virtual ~ATrousFilter(){};

	void init();
	void execute();

	void setInput2DPointCloud(DeviceDataInfoPtr ddiPtr)
	{
		addFilterInput(ddiPtr,WorldCoordinates);
	}

	void setInputSensorInfo(DeviceDataInfoPtr ddiPtr)
	{
		addInputData(ddiPtr,SensorInfoList);
	}

	void setPointIntensity(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,PointIntensity);
	}

	DeviceDataInfoPtr getFilteredWorldCoordinates()
	{
		return getTargetData(FilteredWorldCoordinates);
	}

	void TestAtrousFilter(thrust::host_vector<float4>& views);

private:
	void initAtrousConstants();
};

#endif /* ATROUSFILTER_H_ */

/*
 * ATrousFilterAdvanced.h
 *
 *  Created on: Dec 1, 2012
 *      Author: avo
 */

#ifndef ATROUSFILTERADVANCED_H_
#define ATROUSFILTERADVANCED_H_

#include "filter.hpp"

class ATrousFilterAdvanced : public Filter {

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

	unsigned int iterations;
	float sigma_depth;
	float sigma_intensity;

	unsigned int n_view;

public:
	ATrousFilterAdvanced(unsigned int n_view, unsigned int iterations, float sigma_depth, float sigma_intensity);
	virtual ~ATrousFilterAdvanced();

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

};

#endif /* ATROUSFILTERADVANCED_H_ */

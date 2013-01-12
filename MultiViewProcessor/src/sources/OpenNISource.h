/*
 * OpenNISource.h
 *
 *  Created on: Jan 12, 2013
 *      Author: avo
 */

#ifndef OPENNISOURCE_H_
#define OPENNISOURCE_H_

#include "source.h"

class OpenNISource : public Source {

	enum Target
	{
		ImageDepth,
		ImageRGB,
		PointXYZI,
		PointRGBA,
		PointIntensity,
		SensorInfoList,
	};

public:
	OpenNISource(unsigned int n_view,int outputlevel=0);
	virtual ~OpenNISource();

	void init();
	void loadFrame();

	DeviceDataInfoPtr getWorldCoordinates()
	{
		return getTargetData(PointXYZI);
	}

	DeviceDataInfoPtr getIntensity()
	{
		return getTargetData(PointIntensity);
	}

	DeviceDataInfoPtr getSensorV2InfoList()
	{
		return getTargetData(SensorInfoList);
	}



private:
	dim3 grid,block;
	unsigned int n_view;
	char *baseDir;

	int outputlevel;
};

#endif /* OPENNISOURCE_H_ */

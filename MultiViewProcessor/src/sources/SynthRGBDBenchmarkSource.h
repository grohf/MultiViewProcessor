/*
 * SynthRGBDBenchmarkSource.h
 *
 *  Created on: Oct 26, 2012
 *      Author: avo
 */

#ifndef SYNTHRGBDBENCHMARKSOURCE_H_
#define SYNTHRGBDBENCHMARKSOURCE_H_

#include "source.h"

class SynthRGBDBenchmarkSource : public Source {
public:

	enum Target
	{
		ImageDepth,
		ImageRGB,
		PointXYZI,
		PointRGBA,
		PointIntensity,
		SensorInfoList,
		GroundTruthTransformations,
	};

	SynthRGBDBenchmarkSource(unsigned int n_view_,char *baseDir_,bool transform=false);
	virtual ~SynthRGBDBenchmarkSource();

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

	DeviceDataInfoPtr getGroundTruthTransformation()
	{
		return getTargetData(GroundTruthTransformations);
	}

private:
	dim3 grid,block;
	unsigned int n_view;
	char *baseDir;
	bool transform;
};

#endif /* SYNTHRGBDBENCHMARKSOURCE_H_ */

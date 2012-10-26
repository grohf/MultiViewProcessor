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
	};

	SynthRGBDBenchmarkSource(unsigned int n_view_);
	virtual ~SynthRGBDBenchmarkSource();

	void init();
	void loadFrame();

	DeviceDataInfoPtr getWorldCoordinates()
	{
		return getTargetData(PointXYZI);
	}

private:
	dim3 grid,block;
	unsigned int n_view;
};

#endif /* SYNTHRGBDBENCHMARKSOURCE_H_ */

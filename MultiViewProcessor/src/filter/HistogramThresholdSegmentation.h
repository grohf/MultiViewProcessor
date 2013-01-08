/*
 * HistogramThresholdSegmentation.h
 *
 *  Created on: Dec 2, 2012
 *      Author: avo
 */

#ifndef HISTOGRAMTHRESHOLDSEGMENTATION_H_
#define HISTOGRAMTHRESHOLDSEGMENTATION_H_

#include "filter.hpp"

class HistogramThresholdSegmentation : public Filter {

	enum Input
	{
		WorldCoordinates,
	};

	unsigned int n_view;
	unsigned int outputlevel;

public:
	HistogramThresholdSegmentation(unsigned int n_view,unsigned int outputlevel = 0);
	virtual ~HistogramThresholdSegmentation();

	void init();
	void execute();

	void setPointCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,WorldCoordinates);
	}

};

#endif /* HISTOGRAMTHRESHOLDSEGMENTATION_H_ */

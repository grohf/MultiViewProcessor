/*
 * TruncateThresholdFilter.h
 *
 *  Created on: Sep 17, 2012
 *      Author: avo
 */

#ifndef TRUNCATETHRESHOLDFILTER_H_
#define TRUNCATETHRESHOLDFILTER_H_

#include "filter.hpp"

class TruncateThresholdFilter : public Filter {

	unsigned int n_view;
	float min;
	float max;

	dim3 grid;
	dim3 block;

public:
	TruncateThresholdFilter(unsigned int n_view_, float min_, float max_)
	{
		n_view = n_view_;
		min = min_;
		max = max_;

	}
	virtual ~TruncateThresholdFilter()
	{

	}

	void execute();
	void init();

	void setWorldCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,0);
	}

};

#endif /* TRUNCATETHRESHOLDFILTER_H_ */

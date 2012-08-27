/*
 * ATrousFilter.h
 *
 *  Created on: Aug 25, 2012
 *      Author: avo
 */

#ifndef ATROUSFILTER_H_
#define ATROUSFILTER_H_

#include "filter.hpp"


class ATrousFilter : public Filter {

	dim3 grid;
	dim3 block;
	unsigned int radi;

	unsigned int atrous_radi;
	unsigned int atrous_length;

public:
	ATrousFilter(unsigned int radi_=2) {
		radi = radi_;
		//	DeviceDataParamFunction f = shrinkData;
		//	addParamChanger(f);
	};
	virtual ~ATrousFilter(){};

	void init();
	void execute();

	void setInput2DPointCloud(DeviceDataInfoPtr ddiPtr)
	{
		addFilterInput(ddiPtr,0);
	}

	DeviceDataInfoPtr getFilteredWorldCoordinates()
	{
		return getTargetData(0);
	}

private:
	void initAtrousConstants();
};

#endif /* ATROUSFILTER_H_ */

/*
 * manipulator.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef MANIPULATOR_H_
#define MANIPULATOR_H_

#include "module.h"
#include "device_data.h"
#include <vector>

class Manipulator : public Module, public DeviceDataRequester
{

public:
	virtual ~Manipulator() {}

	virtual void init() = 0;

	virtual void execute() = 0;

	virtual void addSourceData(DeviceDataInfoPtr dInfoPtr, unsigned int idx)
	{
		sourceList[idx] = dInfoPtr;
	}

	virtual DeviceDataPtr getDeviceDataPointer(DeviceDataInfoPtr dInfoPtr)
	{
		return dInfoPtr->ptr;
	}

	//TODO:
	virtual void addTarget() = 0;

protected:
	std::vector<DeviceDataInfoPtr> sourceList;
	std::vector<DeviceDataInfoPtr> targetList;
};


#endif /* MANIPULATOR_H_ */

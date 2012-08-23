/*
 * source.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef SOURCE_H_
#define SOURCE_H_

#include <module.h>
#include <device_data.h>
#include <vector>

class Source : public Module, public DeviceDataRequester
{
public:

	enum Status
	{
		Realized = 0
	};

	Status status;

	/* Timebase */

	virtual void addTargetData(DeviceDataInfoPtr dInfoPtr, unsigned int idx)
	{
		targetList[idx] = dInfoPtr;
	}

protected:
	std::vector<DeviceDataInfoPtr> targetList;
};


#endif /* SOURCE_H_ */

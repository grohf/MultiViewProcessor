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

class Source : public Module, public DeviceDataRequester, public TargetLister
{
public:

	enum Status
	{
		Realized = 0
	};

	Status status;

	/* Timebase */


	virtual void init() = 0;

	void execute()
	{
		loadFrame();
	}

	virtual void loadFrame() = 0;


};


#endif /* SOURCE_H_ */

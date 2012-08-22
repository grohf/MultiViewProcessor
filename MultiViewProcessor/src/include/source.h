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

class Source : public Module, public DeviceDataRequester
{
	enum Status
	{
		Realized = 0
	};

	Status status;

	/* Timebase */


};


#endif /* SOURCE_H_ */

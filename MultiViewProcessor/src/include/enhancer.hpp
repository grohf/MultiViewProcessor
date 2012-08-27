/*
 * enhancer.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef ENHANCER_H_
#define ENHANCER_H_

#include "module.h"
#include "device_data.h"
#include <vector>
#include <boost/shared_ptr.hpp>


class Enhancer : public Module, public DeviceDataRequester, public InputLister<>, public TargetLister<>
{

public:
	virtual ~Enhancer() {}

	virtual void init() = 0;

	virtual void execute() = 0;

protected:

};

typedef boost::shared_ptr<Enhancer> EnhancerPtr;


#endif /* ENHANCER_H_ */

/*
 * feature.hpp
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef FEATURE_HPP_
#define FEATURE_HPP_

#include "enhancer.hpp"
#include <device_data.h>
#include <vector>

class Feature;
typedef boost::shared_ptr<Feature> FeaturePtr;

class Feature : public Enhancer
{
public:

	virtual void init() = 0;
	virtual void execute() = 0;

	Feature()
	{
		ptr.reset(this);
	}

	FeaturePtr getFeaturePtr()
	{
		return ptr;
	}

private:
	FeaturePtr ptr;
	std::vector<DeviceDataParams> paramList;
};


#endif /* FEATURE_HPP_ */

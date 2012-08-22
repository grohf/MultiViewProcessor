/*
 * feature.hpp
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef FEATURE_HPP_
#define FEATURE_HPP_

#include "manipulator.hpp"
#include <device_data.h>
#include <vector>

class Feature : public Manipulator
{
public:
	void setTargetDeviceDataParams(DeviceDataParams params)
	{

	}

private:
	std::vector<DeviceDataParams> paramList;
};


#endif /* FEATURE_HPP_ */

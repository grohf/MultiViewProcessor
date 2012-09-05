/*
 * TransformCam2World.h
 *
 *  Created on: Sep 4, 2012
 *      Author: avo
 */

#ifndef TRANSFORMCAM2WORLD_H_
#define TRANSFORMCAM2WORLD_H_

#include "filter.hpp"

class TransformCam2World : public Filter {
public:

	enum Input
	{
		CoordsAsCamCoords,
	};

	enum Target
	{
		CoordsAsWorldCoords,
	};

	void init();
	void execute();

	TransformCam2World()
	{

	}

	virtual ~TransformCam2World()
	{

	}

	void setInputAsCamCoords(DeviceDataInfoPtr ddiPtr)
	{

		addTargetData(ddiPtr,CoordsAsWorldCoords);
	}

	void setInputRGB(DeviceDataInfo ddiPtr)
	{

	}
};

#endif /* TRANSFORMCAM2WORLD_H_ */

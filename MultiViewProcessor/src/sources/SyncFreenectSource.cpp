/*
 * SyncFreenectSource.cpp
 *
 *  Created on: Aug 23, 2012
 *      Author: avo
 */

#include "SyncFreenectSource.h"

struct SyncFreenectLoader
{

};

void
SyncFreenectSource::init()
{

}

SyncFreenectSource::SyncFreenectSource()
{
	DeviceDataParams pointXYZIParams;
	pointXYZIParams.elements = 640*480;
	pointXYZIParams.element_size = sizeof(float4);
	pointXYZIParams.dataType = Point4D;
	pointXYZIParams.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(pointXYZIParams),PointXYZI);

	DeviceDataParams pointRGBAParams;
	pointRGBAParams.elements = 640*480;
	pointRGBAParams.element_size = sizeof(uchar4);
	pointRGBAParams.dataType = Point4D;
	pointRGBAParams.elementType = UCHAR4;

	addTargetData(addDeviceDataRequest(pointRGBAParams),PointRGBA);
}

SyncFreenectSource::~SyncFreenectSource()
{
	// TODO Auto-generated destructor stub
}


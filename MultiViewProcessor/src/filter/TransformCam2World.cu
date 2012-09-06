/*
 * TransformCam2World.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: avo
 */

#include "device_data.h"

#include "TransformCam2World.h"

namespace device
{

	struct CamToWorldTransformer
	{
		float4 *coords;
		SensorInfo *sinfolist;
		unsigned int sensor;

		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZ(int cx,int cy, float wz, float &wx, float &wy) const
		{

			double factor = (sinfolist[sensor].pix_size * wz)  / sinfolist[sensor].dist;

			wx = (float) ((cx - 320) * factor);
			wy = (float) ((cy - 240) * factor);

		}

		__device__ __forceinline__ void
		operator () () const
		{
			int x = blockIdx.x*blockDim.x + threadIdx.x;
			int y = blockIdx.y*blockDim.y + threadIdx.y;
//			int z = blockIdx.z;

			float wz = coords[y*640+x].z;
			float wx,wy;

			convertCXCYWZtoWXWYWZ(x,y,wz,wx,wy);

//			float4 f4 = make_float4(wx,wy,wz,((r+r+r+b+g+g+g+g)>>3));
//			xyzi[z*640*480+y*640+x] = f4;



		}
	};
}


void TransformCam2World::init()
{

}

void TransformCam2World::execute()
{

}


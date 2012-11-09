/*
 * device_utils.hpp
 *
 *  Created on: Nov 8, 2012
 *      Author: avo
 */

#ifndef DEVICE_UTILS_HPP_
#define DEVICE_UTILS_HPP_

namespace device
{
	__device__ __forceinline__ float
	dotf43(const float4& v1,const float4& v2)
	{
	  return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
	}

	__device__ __forceinline__ float4
	minusf43(const float4& v1, const float4& v2)
	{
	  return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0);
	}

	__device__ __host__ __forceinline__ float4
	crossf43(const float4& v1, const float4& v2)
	{
	  return make_float4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x,0);
	}

	__device__ __forceinline__ float
	lengthf43(const float4& v)
	{
	  return sqrt(dotf43(v, v));
	}

	__device__ __forceinline__ float
	kldivergence(float *cur,float *mean,int bins)
	{
		float div = 0.f;

		for(int i=0;i<bins;i++)
		{
			div += (cur[i]-mean[i])*__logf(cur[i]/mean[i]);
		}
		return div;
	}

	__device__ __forceinline__ float
	euclideandivergence(float *cur,float *mean,int bins)
	{
		float div = 0.f;

		for(int i=0;i<bins;i++)
		{
			float d = (cur[i]-mean[i]);
			div += d*d;
		}
		return sqrtf(div);
	//    	return div;
	}
}


#endif /* DEVICE_UTILS_HPP_ */

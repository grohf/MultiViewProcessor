/*
 * TruncateThresholdFilter.cpp
 *
 *  Created on: Sep 17, 2012
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>

#include "TruncateThresholdFilter.h"
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <stdio.h>


namespace device
{
	struct ThresholdTruncator
	{
		float4* pos;
		float min;
		float max;

		__device__ __forceinline__ void
		operator () () const
		{
			int sx = blockIdx.x*blockDim.x+threadIdx.x;
			int sy = blockIdx.y*blockDim.y+threadIdx.y;

			int off = blockIdx.z*640*480+sy*640+sx;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf(" %f %f %f \n",pos[off].x,pos[off].y,pos[off].z);

			float4 wc = pos[off];
			if(wc.z < min || wc.z > max)
				pos[off].z = 0;
		}

	};
	__global__ void filterTruncateThreshold(ThresholdTruncator tt) { tt (); }

}
device::ThresholdTruncator truncator;


template <typename T>
struct clamp : public thrust::unary_function<T,T>
{
    T lo, hi;

    __host__ __device__
    clamp(T _lo, T _hi) : lo(_lo), hi(_hi) {}

    __host__ __device__
    T operator()(T x)
    {
        if (x < lo)
            return lo;
        else if (x < hi)
            return x;
        else
            return hi;
    }
};

void TruncateThresholdFilter::init()
{

	truncator.pos = (float4 *)getInputDataPointer(0);
	truncator.min = min;
	truncator.max = max;

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);
}


void TruncateThresholdFilter::execute()
{

	device::filterTruncateThreshold<<<grid,block>>>(truncator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

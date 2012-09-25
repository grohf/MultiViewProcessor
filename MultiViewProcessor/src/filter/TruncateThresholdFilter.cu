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

#include "point_info.hpp"
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
			SegmentationPointInfo spi;
			spi = Foreground;
			if(wc.z < min || wc.z > max)
				spi = Background;

			device::setSegmentationPointInfo(pos[off].w,spi);
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

	size_t uc4s = 640*480*sizeof(uchar4);
	char path[50];
	float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
	checkCudaErrors(cudaMemcpy(h_f4_depth,truncator.pos,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

//	uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
//	for(int i=0;i<640*480;i++)
//	{
//		unsigned char g = h_f4_depth[i].z/20;
//		h_uc4_depth[i] = make_uchar4(g,g,g,128);
//
//		if(!device::isValid(h_f4_depth[i].w)) h_uc4_depth[i].x = 255;
//
//		if(device::isReconstructed(h_f4_depth[i].w)) h_uc4_depth[i].y = 255;
//	}
//
//	sprintf(path,"/home/avo/pcds/src_depth_valid_map%d.ppm",0);
//	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);



	uchar4 *h_uc4_depth2 = (uchar4 *)malloc(uc4s);
	for(int i=0;i<640*480;i++)
	{
		unsigned char g = h_f4_depth[i].z/20;
		h_uc4_depth2[i] = make_uchar4(g,g,g,128);

		if(device::isForeground(h_f4_depth[i].w)) h_uc4_depth2[i].x = 255;

		if(device::isBackground(h_f4_depth[i].w)) h_uc4_depth2[i].y = 255;

		if(!device::isSegmented(h_f4_depth[i].w)) h_uc4_depth2[i].z = 255;
	}

	sprintf(path,"/home/avo/pcds/src_segmented_map%d.ppm",0);
	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth2,640,480);

}

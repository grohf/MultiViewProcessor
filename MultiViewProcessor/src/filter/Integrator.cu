/*
 * Integrator.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "point_info.hpp"
#include "../sink/pcd_io.h"

#include "Integrator.h"

namespace device
{
	struct IntegratorFilter
	{
		enum
		{
			dx = 32,
			dy = 32,
		};

		float4* input_pos;
		float* input_transformations;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int sx = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int sy = blockIdx.y*blockDim.y+threadIdx.y;

			float4 pos = input_pos[blockIdx.z*640*480+sy*640+sx];
			float4 tp = make_float4(0,0,0,0);

			__shared__ float tm[12];
			if(threadIdx.y==0 && threadIdx.x<12)
			{
				tm[threadIdx.x] = input_transformations[blockIdx.z*12+threadIdx.x];
//				tm[sx] = 1.f;
			}
			__syncthreads();
//
//			if(blockIdx.z==0 && blockIdx.x==5 && blockIdx.y==5 && threadIdx.y==0 && threadIdx.x==0)
//			{
//				printf("kernelTransform: ");
//				for(int i=0;i<12;i++)
//				{
//					printf("%f | ",tm[i]);
//				}
//				printf("\n");
//			}

			if(!isReconstructed(pos.w) && isForeground(pos.w) && isValid(pos.w))
//			if(isForeground(pos.w))
			{
				tp.x = tm[0] * pos.x + tm[1] * pos.y + tm[2] * pos.z + tm[9];
				tp.y = tm[3] * pos.x + tm[4] * pos.y + tm[5] * pos.z + tm[10];
				tp.z = tm[6] * pos.x + tm[7] * pos.y + tm[8] * pos.z + tm[11];
			}

			input_pos[blockIdx.z*640*480+sy*640+sx] = tp;
		}

	};

	__global__ void integrateViews(const IntegratorFilter inte){ inte(); }

}

device::IntegratorFilter integratorFilter;

void Integrator::init()
{
	integratorFilter.input_pos = (float4 *)getInputDataPointer(Positions);
	integratorFilter.input_transformations = (float *)getInputDataPointer(MinTransformations);
}

unsigned int maxFrames = 300;
float4 *buffer;

unsigned int c=0;

void Integrator::execute()
{
	thrust::device_ptr<float> ptr_bestTransform = thrust::device_pointer_cast(integratorFilter.input_transformations);
	thrust::device_vector<float> d_bestTransform(ptr_bestTransform,ptr_bestTransform+n_view*12);
	thrust::host_vector<float> h_bestTransform = d_bestTransform;

	for(int v=0;v<n_view;v++)
	{
		for(int i=0;i<12;i++)
		{
			printf("%f | ",h_bestTransform[v*12+i]);
		}
		printf("\n");
	}

	dim3 integrateBlock(integratorFilter.dx,integratorFilter.dy);
	dim3 integrateGrid(640/integratorFilter.dx,480/integratorFilter.dy,n_view);


	device::integrateViews<<<integrateGrid,integrateBlock>>>(integratorFilter);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if(c<maxFrames)
	{
		checkCudaErrors(cudaMemcpy(&buffer[c*n_view*640*480],integratorFilter.input_pos,n_view*640*480*sizeof(float4),cudaMemcpyDeviceToHost));
		c++;
	}

	printf("integrator done \n");

//	thrust::device_ptr<float4> ptr_pos = thrust::device_pointer_cast(integratorFilter.input_pos);
//	thrust::device_vector<float4> d_pos(ptr_pos,ptr_pos+n_view*640*480);
//	thrust::host_vector<float4> h_pos = d_pos;
//
//
//
////	for(int v=0;v<n_view;v++)
////	{
////		for(int p=0;p<640*480;p++)
////		{
////
////		}
////	}
//
//	char path[100];
//	for(int i=0;i<n_view;i++)
//	{
//		sprintf(path,"/home/avo/pcds/mt/besttransformed_%d.pcd",i);
//		host::io::PCDIOController pcdIOCtrl;
//		float4 *f4p = h_pos.data();
//		pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
//	}
//
//	printf("write imgs \n");
//
//	for(int v=0;v<n_view;v++)
//	{
////		float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
////		checkCudaErrors(cudaMemcpy(h_f4_depth,integratorFilter.input_pos+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
////
//
//
//		size_t uc4s = 640*480*sizeof(uchar4);
//		uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
//
//
//		for(int i=0;i<640*480;i++)
//		{
//			unsigned char g = h_pos[v*640*480+i].z/20;
//			h_uc4_depth[i] = make_uchar4(g,g,g,128);
//
////			if(!device::isValid(h_f4_depth[i].w)) h_uc4_depth[i].x = 255;
////
////			if(device::isReconstructed(h_f4_depth[i].w)) h_uc4_depth[i].y = 255;
//		}
//
//		sprintf(path,"/home/avo/pcds/mt/besttranstest%d.ppm",v);
//		sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);
//	}

}

void Integrator::flush()
{
	printf("start flushing.. \n");
	char path[100];
	for(int k=0;k<c;k++)
	{
		printf("---> %d / %d \n",k,c);
		for(int i=0;i<n_view;i++)
		{
			sprintf(path,"/home/avo/pcds/seqs/13/t_%d_%d.pcd",k,i);
			host::io::PCDIOController pcdIOCtrl;
			float4 *f4p = &(buffer[k*n_view*640*480+i*640*480]);
			pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
		}
	}
}

Integrator::Integrator(unsigned int n_view) : n_view(n_view)
{
	// TODO Auto-generated constructor stub
	buffer = (float4*)malloc(maxFrames*n_view*640*480*sizeof(float4));
}

Integrator::~Integrator()
{
	// TODO Auto-generated destructor stub
}


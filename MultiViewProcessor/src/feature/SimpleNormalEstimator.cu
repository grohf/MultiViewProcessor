/*
 * SimpleNormalEstimator.cpp
 *
 *  Created on: Dec 7, 2012
 *      Author: avo
 */

#include "SimpleNormalEstimator.h"

#include "utils.hpp"
#include "point_info.hpp"
#include <helper_cuda.h>
#include <helper_image.h>

namespace device
{
	struct SimpleNormaleEstimation
	{
		enum
		{
			dx = 32,
			dy = 32,
		};

		float4 *input_pos;
		float4 *output_normal;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
			unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int z = blockIdx.z;

			if(y+1>=480 || x+1>640)
				return;

			float3 m = fetchToFloat3(input_pos[z*640*480+y*640+x]);
			float3 p = fetchToFloat3(input_pos[z*640*480+(y+1)*640+x])-m;
			float3 q = fetchToFloat3(input_pos[z*640*480+y*640+(x+1)])-m;

			float3 n = cross(q,p);
			n = n * (1.f / norm(n));

			if(dot(n,m)>0.f)
				n *= -1.f;

			output_normal[blockIdx.z*640*480+y*640+x] = make_float4(n.x,n.y,n.z,0);
		}
	};
	__global__ void estimateNormals(const SimpleNormaleEstimation ne) {ne (); }

	struct SobleNormaleEstimation
	{
		enum
		{
			dx = 32,
			dy = 32,
			r = 1,

			kx = 2*r+dx,
			ky = 2*r+dy,

			Min = 0,
			Max = 15000,
		};

		float4 *input_pos;
		float4 *output_normal;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float4 shm[kx*ky];

			const int oy = blockIdx.y*blockDim.y-r;
			const int ox = blockIdx.x*blockDim.x-r;

			int sx,sy,off;
			for(off=threadIdx.y*blockDim.x+threadIdx.x;off<kx*ky;off+=blockDim.x*blockDim.y){
				sy = off/kx;
				sx = off - sy*kx;

				sy = oy + sy;
				sx = ox + sx;

				if(sx < 0) 		sx	=	0;
				if(sx > 639) 	sx 	= 639;
				if(sy < 0)		sy 	= 	0;
				if(sy > 479)	sy 	= 479;

				shm[off]=input_pos[blockIdx.z*640*480+sy*640+sx];
			}
			__syncthreads();

			off = (threadIdx.y+1)*kx+threadIdx.x+1;
		  sx = threadIdx.x + blockIdx.x * blockDim.x;
		  sy = threadIdx.y + blockIdx.y * blockDim.y;
			if(shm[off].z<=Min)
			{
				float4 f4 = make_float4(0,0,0,-2);
				output_normal[blockIdx.z*640*480+sy*640+sx] = f4;
				return;
			}

			if(shm[off].z>Max)
			{
				float4 f4 = make_float4(0,0,0,-3);
				output_normal[blockIdx.z*640*480+sy*640+sx] = f4;
				return;
			}

			float3 p;
			off = (threadIdx.y+0)*kx+threadIdx.x+0;
			p 	= 	fetchToFloat3(shm[off]) * (-1.f/8.f);

			off = (threadIdx.y+1)*kx+threadIdx.x+0;
			p = p +	fetchToFloat3(shm[off]) * (-1.f/4.f);

			off = (threadIdx.y+2)*kx+threadIdx.x+0;
			p = p + fetchToFloat3(shm[off]) * (-1.f/8.f);

			off = (threadIdx.y+0)*kx+threadIdx.x+2;
			p = p +	fetchToFloat3(shm[off]) * (1.f/8.f);

			off = (threadIdx.y+1)*kx+threadIdx.x+2;
			p = p + fetchToFloat3(shm[off])	* (1.f/4.f);

			off = (threadIdx.y+2)*kx+threadIdx.x+2;
			p = p + fetchToFloat3(shm[off])	* (1.f/8.f);


			float3 q;
			off = (threadIdx.y+0)*kx+threadIdx.x+0;
			q = 	fetchToFloat3(shm[off]) * (-1.f/8.f);

			off = (threadIdx.y+0)*kx+threadIdx.x+1;
			q = q + fetchToFloat3(shm[off]) * (-1.f/4.f);

			off = (threadIdx.y+0)*kx+threadIdx.x+2;
			q = q + fetchToFloat3(shm[off]) * (-1.f/8.f);

			off = (threadIdx.y+2)*kx+threadIdx.x+0;
			q = q + fetchToFloat3(shm[off]) * (1.f/8.f);

			off = (threadIdx.y+2)*kx+threadIdx.x+1;
			q = q +	fetchToFloat3(shm[off]) * (1.f/4.f);

			off = (threadIdx.y+2)*kx+threadIdx.x+2;
			q = q + fetchToFloat3(shm[off]) * (1.f/8.f);

			float3 n = cross(q,p);
			float l = norm(n);
			if(l>0)
				n = n * (1.f/l);


			float4 f4 = make_float4(n.x,n.y,n.z,(l>0)?l:-1.f);
			output_normal[blockIdx.z*640*480+sy*640+sx] = f4;

		}

	};
	__global__ void estimateSobleNormals(const SobleNormaleEstimation sne) {sne (); }



}
device::SimpleNormaleEstimation simpleNormalEstimator;
device::SobleNormaleEstimation sobleNormalEstimator;



void
SimpleNormalEstimator::init()
{
	simpleNormalEstimator.input_pos = (float4 *)getInputDataPointer(WorldCoordinates);
	simpleNormalEstimator.output_normal = (float4 *)getTargetDataPointer(Normals);

	sobleNormalEstimator.input_pos = (float4 *)getInputDataPointer(WorldCoordinates);
	sobleNormalEstimator.output_normal = (float4 *)getTargetDataPointer(Normals);
}

void
SimpleNormalEstimator::execute()
{
	dim3 grid(640/simpleNormalEstimator.dx,480/simpleNormalEstimator.dy,n_view);
	dim3 block(simpleNormalEstimator.dx,simpleNormalEstimator.dy);

//	device::estimateNormals<<<grid,block>>>(simpleNormalEstimator);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());

	device::estimateSobleNormals<<<grid,block>>>(sobleNormalEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	char path[50];
	for(int v=0;v<n_view;v++)
	{
		float4 *h_f4_normals = (float4 *)malloc(640*480*sizeof(float4));
//		checkCudaErrors(cudaMemcpy(h_f4_normals,simpleNormalEstimator.output_normal+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_f4_normals,sobleNormalEstimator.output_normal+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));


		float4 *h_f4_pos = (float4 *)malloc(640*480*sizeof(float4));
		checkCudaErrors(cudaMemcpy(h_f4_pos,sobleNormalEstimator.input_pos+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

		uchar4 *h_uc4 = (uchar4 *)malloc(640*480*sizeof(uchar4));
		for(int i=0;i<640*480;i++)
		{
			if(h_f4_normals[i].w >= 0 && device::isForeground(h_f4_pos[i].w)) //&& !device::isReconstructed(h_f4_pos[i].w)
			{
				h_uc4[i].x = (h_f4_normals[i].x+1.0f)*127.f;
				h_uc4[i].y = (h_f4_normals[i].y+1.0f)*127.f;
				h_uc4[i].z = (h_f4_normals[i].z+1.0f)*127.f;
				h_uc4[i].w = 127.f;
			}
			else
			{
				h_uc4[i].x = 0;
				h_uc4[i].y = 0;
				h_uc4[i].z = 0;
				h_uc4[i].w = 127.f;
			}

		}

		sprintf(path,"/home/avo/pcds/soble_normals_%d.ppm",v);
		sdkSavePPM4ub(path,(unsigned char *)h_uc4,640,480);
	}
	printf("Simple Normals done! \n");

}

SimpleNormalEstimator::SimpleNormalEstimator(unsigned int n_view) : n_view(n_view)
{
	DeviceDataParams params;
	params.elements = 640*480*n_view;
	params.element_size = sizeof(float4);
	params.dataType = Point4D;
	params.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(params),Normals);

}

SimpleNormalEstimator::~SimpleNormalEstimator()
{
	// TODO Auto-generated destructor stub
}


/*
 * ATrousFilter.cpp
 *
 *  Created on: Aug 25, 2012
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>

#include "ATrousFilter.h"

//Test
#include "../sink/pcd_io.h"

namespace device
{
	namespace constant
	{
		__constant__ unsigned int atrous_radi;
		__constant__ unsigned int atrous_length;
		__constant__ unsigned int level2step_size[] = { 1, 2, 4, 8, 16, 32 };
		__constant__ float atrous_coef[64];
		__constant__ float atrous_sum;

		__constant__ unsigned int lx;
		__constant__ unsigned int ly;

		__constant__ double sensorInfo_pix_size;
		__constant__ double sensorInfo_dist;
	}
}


namespace device
{
	struct ATrousEnhancer128
	{

		unsigned int modi;
		unsigned int level;

		float4* input;
		float4* output;

		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float4 shm[36*28];

			int off,sx,sy;



			if(modi & 8){


				sx = blockIdx.x*blockDim.x+threadIdx.x;
				sy = blockIdx.y*blockDim.y+threadIdx.y;

				off = blockIdx.z*640*480+sy*640+sx;

				if(off >= 640*480)
					printf("off to big!! x: %d | y: %d | z: %d \n",sx,sy,blockIdx.z);

				float4 inp = input[off];

//				if(blockIdx.x == 3 && blockIdx.y==3 && threadIdx.x == 3 && threadIdx.y == 3)
//				{
//					printf("x: %f | y: %f | z: %f | w: %f \n",inp.x,inp.y,inp.z,inp.w);
//				}

				output[blockIdx.z*640*480+sy*640+sx] = inp;

				return;
			}



			/* ---------- LOAD SHM ----------- */
	//			int oy = blockIdx.y*blockDim.y-constant::atrous_radi*constant::level2step_size[level];
	//			int ox = blockIdx.x*blockDim.x-constant::atrous_radi*constant::level2step_size[level];

			int ox,oy;

			sy = blockIdx.y*blockDim.y/constant::level2step_size[level];
			oy = sy-constant::atrous_radi*constant::level2step_size[level] + blockIdx.y*blockDim.y-sy;

			sx = blockIdx.x*blockDim.x/constant::level2step_size[level];
			ox = sx-constant::atrous_radi*constant::level2step_size[level] + blockIdx.x*blockDim.x-sx;

//			if(blockIdx.x == 1 && blockIdx.y==1 && threadIdx.x == 3 && threadIdx.y == 3)
//			{
//				printf("ox: %d | oy: %d \n",ox,oy);
//				printf("lx: %d | ly: %d \n",constant::lx,constant::ly);
//			}

//			off = threadIdx.y*blockDim.x+threadIdx.x;
//			sy = off/constant::lx;
//			sx = off - sy*constant::lx;
//
//			sy = oy + sy * constant::level2step_size[level];
//			sx = ox + sx * constant::level2step_size[level];
//
//			if(sx < 0) 		sx	=	0;
//			if(sx > 639) 	sx 	= 639;
//			if(sy < 0)		sy 	= 	0;
//			if(sy > 479)	sy 	= 479;
//
//	//			surf3Dread<uchar4>(&u4tmp,surf::surfRef1,(ox+sx)*sizeof(uchar4),(oy+sy),blockIdx.z*2,cudaBoundaryModeClamp);
//	//			shm_luma[off] = (u4tmp.x+u4tmp.x+u4tmp.x+u4tmp.z+u4tmp.y+u4tmp.y+u4tmp.y+u4tmp.y)>>3;
//
////			surf3Dread<float4>(&shm[off],surf::surfRef,(ox+sx)*sizeof(float4),(oy+sy),blockIdx.z,cudaBoundaryModeClamp);
//			shm[off]=input[blockIdx.z*640*480+sy*640+sx];
//
//			off += blockDim.x*blockDim.y;
//			if(off<constant::lx*constant::ly){
//				sy = off/constant::lx;
//				sx = off - sy*constant::lx;
//
//				sy = oy + sy * constant::level2step_size[level];
//				sx = ox + sx * constant::level2step_size[level];
//
//				if(sx < 0) 		sx	=	0;
//				if(sx > 639) 	sx 	= 639;
//				if(sy < 0)		sy 	= 	0;
//				if(sy > 479)	sy 	= 479;
//
//	//				surf3Dread<uchar4>(&u4tmp,surf::surfRef1,(ox+sx)*sizeof(uchar4),(oy+sy),blockIdx.z*2,cudaBoundaryModeClamp);
//	//				shm_luma[off] = (u4tmp.x+u4tmp.x+u4tmp.x+u4tmp.z+u4tmp.y+u4tmp.y+u4tmp.y+u4tmp.y)>>3;
//
////				surf3Dread<float4>(&shm[off],surf::surfRef,(ox+sx)*sizeof(float4),(oy+sy),blockIdx.z,cudaBoundaryModeClamp);
//				shm[off]=input[blockIdx.z*640*480+sy*640+sx];
//			}
//			__syncthreads();


			for(off = threadIdx.y*blockDim.x+threadIdx.x;off<(constant::lx*constant::ly);off+=blockDim.x*blockDim.y)
			{
				sy = off/constant::lx;
				sx = off - sy*constant::lx;

				sy = oy + sy * constant::level2step_size[level];
				sx = ox + sx * constant::level2step_size[level];

				if(sx < 0) 		sx	=	0;
				if(sx > 639) 	sx 	= 639;
				if(sy < 0)		sy 	= 	0;
				if(sy > 479)	sy 	= 479;

				shm[off]=input[blockIdx.z*640*480+sy*640+sx];
			}
			__syncthreads();

			/* ---------- CONV ----------- */

			off = (threadIdx.y+constant::atrous_radi)*constant::lx+threadIdx.x+constant::atrous_radi;

			float sum = 0.0f;
			float sum_weight = 0.0f;

			float mid_depth = shm[off].z;
			float mid_luma = shm[off].w;

			float cur_depth;

			float weight;
			for(sy=0;sy<constant::atrous_length;sy++)
			{
				for(sx=0;sx<constant::atrous_length;sx++)
				{
					off = (threadIdx.y+sy)*constant::lx+threadIdx.x+sx;
					cur_depth = shm[off].z;

					weight = constant::atrous_coef[sy*constant::atrous_length+sx];
					if(cur_depth==0.0f) weight = 0.0f;

					if(mid_depth != 0.0f && (modi & 2) )
					{
						weight *= __expf(-0.5f*(abs(mid_depth-cur_depth)/(10.0f * 10.0f)));
					}

					if(modi & 4)
					{
						 weight *= __expf(-0.5f*(abs(mid_luma-shm[off].w)/(5.0f * 5.0f)));
					}

					sum += weight * cur_depth;
					sum_weight += weight;
				}
			}

			/* ---------- SAVE ----------- */

			sx = blockIdx.x*blockDim.x + threadIdx.x;
			sy = blockIdx.y*blockDim.y + threadIdx.y;
	//
			off = (sum_weight/constant::atrous_sum)*255.0f;
			off = (off << 16 | off << 8 | off) & 0x00ffffff;
//	//
//	////			uchar4 u4 = make_uchar4(off,off,off,0);
//	////			surf3Dwrite<uchar4>(u4,surf::surfOutRef,sx*sizeof(uchar4),sy,blockIdx.z);
//	//
			off = (threadIdx.y+constant::atrous_radi)*constant::lx+threadIdx.x+constant::atrous_radi;
			shm[off].z = (sum_weight>0)?(sum/sum_weight):0;
//			surf3Dwrite<float4>(shm[off],surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
			output[blockIdx.z*640*480+sy*640+sx] = shm[off];

//				float4 f4t = make_float4(0,0,500,0);
//				surf3Dwrite<float4>(f4t,surf::surfRefBuffer,sx*sizeof(float4),sy,0);
		}
	};

	__global__ void filterAtrousKernel(const ATrousEnhancer128 ate){ ate (); }

	struct CoordsUpdater
	{

		float4* pos;

		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZ(int cx,int cy, float wz, float &wx, float &wy) const
		{

			double factor = (constant::sensorInfo_pix_size * wz)  / constant::sensorInfo_dist;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf("factor: %f ref_pix_size: %f ref_dist: %f \n",factor,constant::sensorInfo_pix_size,constant::sensorInfo_dist);

			wx = (float) ((cx - 320) * factor);
			wy = (float) ((cy - 240) * factor);

		}

		__device__ __forceinline__ void
		operator () () const
		{
			int sx = blockIdx.x*blockDim.x+threadIdx.x;
			int sy = blockIdx.y*blockDim.y+threadIdx.y;

			int off = blockIdx.z*640*480+sy*640+sx;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf(" %f %f %f \n",pos[off].x,pos[off].y,pos[off].z);

			float4 wc = pos[off];
			convertCXCYWZtoWXWYWZ(sx,sy,wc.z,wc.x,wc.y);
			pos[off] = wc;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf(" %f %f %f \n",pos[off].x,pos[off].y,pos[off].z);

		}
	};
	__global__ void updateCoords(const CoordsUpdater cu){ cu (); }

}

device::ATrousEnhancer128 atrousfilter;
device::CoordsUpdater coordsUpdater;

void ATrousFilter::execute()
{

	device::filterAtrousKernel<<<grid,block>>>(atrousfilter);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

//	atrousfilter.level = 1;
//	atrousfilter.input = (float4 *)getTargetDataPointer(0);
//	atrousfilter.output = (float4 *)getInputDataPointer(0);
//
//	device::filterAtrousKernel<<<grid,block>>>(atrousfilter);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	atrousfilter.level = 2;
//	atrousfilter.input = (float4 *)getInputDataPointer(0);
//	atrousfilter.output = (float4 *)getTargetDataPointer(0);
//
//	device::filterAtrousKernel<<<grid,block>>>(atrousfilter);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());

	device::updateCoords<<<grid,block>>>(coordsUpdater);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


//	size_t uc4s = 640*480*sizeof(uchar4);
////	uchar4 *h_uc4_rgb = (uchar4 *)malloc(uc4s);
////	checkCudaErrors(cudaMemcpy(h_uc4_rgb,loader.rgba,640*480*sizeof(uchar4),cudaMemcpyDeviceToHost));
////
//	char path[50];
////	sprintf(path,"/home/avo/pcds/src_rgb%d.ppm",0);
////	sdkSavePPM4ub(path,(unsigned char*)h_uc4_rgb,640,480);
////
//	float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
//	checkCudaErrors(cudaMemcpy(h_f4_depth,atrousfilter.output,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
//
//	uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
//	for(int i=0;i<640*480;i++)
//	{
//		unsigned char g = h_f4_depth[i].z/20;
//		h_uc4_depth[i] = make_uchar4(g,g,g,128);
//	}
//
//	sprintf(path,"/home/avo/pcds/src_depth_atrous%d.ppm",0);
//	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);

	/* Test
		char path[50];

		for(int i=0;i<n_view;i++)
		{
			float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
			checkCudaErrors(cudaMemcpy(h_f4_depth,coordsUpdater.pos+i*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

			sprintf(path,"/home/avo/pcds/src_wc_points_atfiltered_%d.pcd",i);
			host::io::PCDIOController pcdIOCtrl;
			pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
		}
	 */

}


void ATrousFilter::init()
{

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);

	atrousfilter.modi = 1 | 2 | 4;
	atrousfilter.level = 0;
	atrousfilter.input = (float4 *)getInputDataPointer(0);
	atrousfilter.output = (float4 *)getTargetDataPointer(0);

//	printf("atrous_radi: %d \n",atrous_radi);
//	cudaMemcpyToSymbol(device::constant::atrous_radi,&atrous_radi,sizeof(unsigned int));
//	cudaMemcpyToSymbol(device::constant::atrous_length,&atrous_length,sizeof(unsigned int));

	initAtrousConstants();

	SensorInfo *d_sinfo = (SensorInfo *)getInputDataPointer(1);
	cudaMemcpyToSymbol(device::constant::sensorInfo_pix_size,&(d_sinfo->pix_size),sizeof(double),0,cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(device::constant::sensorInfo_dist,&(d_sinfo->dist),sizeof(double),0,cudaMemcpyDeviceToDevice);

	SensorInfo h_sinfo;
	checkCudaErrors(cudaMemcpy(&h_sinfo,d_sinfo,sizeof(SensorInfo),cudaMemcpyDeviceToHost));

//	printf("host: pix_size: %f | dist %f \n",h_sinfo.pix_size,h_sinfo.dist);


	coordsUpdater.pos = (float4 *)getTargetDataPointer(0);
}

void ATrousFilter::initAtrousConstants()
{
	using namespace device;
//	printf("radi %d \n",radi);
//	radi = 2;
	atrous_radi = radi;
	atrous_length = 2*atrous_radi+1;
	float atrous_coef[atrous_length*atrous_length];
	cudaMemcpyToSymbol(device::constant::atrous_radi,&atrous_radi,sizeof(unsigned int));
	cudaMemcpyToSymbol(device::constant::atrous_length,&atrous_length,sizeof(unsigned int));



	int as = 1;
	for(;as<atrous_length;as++){
		atrous_coef[as]=1.0f;
		unsigned int ai = as-1;
		for(;ai>0;ai--)
			atrous_coef[ai]=atrous_coef[ai]+atrous_coef[ai-1];
		atrous_coef[0]=1.0f;
	}


	as = atrous_length-1;
	for(;as>=0;as--){
		float mul = atrous_coef[as]/(atrous_coef[atrous_length/2]*atrous_coef[atrous_length/2]);
		unsigned int ai = 0;
		for(;ai<atrous_length;ai++)
			atrous_coef[as*atrous_length+ai]=mul*atrous_coef[ai];
	}

	float atrous_sum = 0.0f;
	int ati;
	for(ati=0;ati<atrous_length*atrous_length;ati++){
		atrous_sum += atrous_coef[ati];
	}
	cudaMemcpyToSymbol(constant::atrous_coef,&atrous_coef,sizeof(float)*atrous_length*atrous_length);
	cudaMemcpyToSymbol(constant::atrous_sum,&atrous_sum,sizeof(float));

	unsigned int lx = block.x+2*atrous_radi;
	unsigned int ly = block.y+2*atrous_radi;

	cudaMemcpyToSymbol(constant::lx,&lx,sizeof(unsigned int));
	cudaMemcpyToSymbol(constant::ly,&ly,sizeof(unsigned int));
//
//	shmSize = (block.x+atrous_radi*2)*(block.y+atrous_radi*2)*sizeof(uchar4)*2;
}





//void shrinkData(DeviceDataParams &params)
//{
//	params.elements /= 2;
//}


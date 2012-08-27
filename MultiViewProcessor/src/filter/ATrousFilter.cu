/*
 * ATrousFilter.cpp
 *
 *  Created on: Aug 25, 2012
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "ATrousFilter.h"

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
//				off = threadIdx.y*blockDim.x+threadIdx.x;

//				surf3Dread<float4>(&shm[off],surf::surfRef,sx*sizeof(float4),sy,blockIdx.z,cudaBoundaryModeClamp);
//				surf3Dwrite<float4>(shm[off],surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);

				off = blockIdx.z*640*480+sy*640+sx;
				if(off >= 640*480)
					printf("off to big!! x: %d | y: %d | z: %d \n",sx,sy,blockIdx.z);

				float4 inp = input[off];

				if(blockIdx.x == 3 && blockIdx.y==3 && threadIdx.x == 3 && threadIdx.y == 3)
				{
					printf("x: %f | y: %f | z: %f | w: %f \n",inp.x,inp.y,inp.z,inp.w);
				}

				output[blockIdx.z*640*480+sy*640+sx] = input[blockIdx.z*640*480+sy*640+sx];

				return;
			}
//
//
//
//			/* ---------- LOAD SHM ----------- */
//	//			int oy = blockIdx.y*blockDim.y-constant::atrous_radi*constant::level2step_size[level];
//	//			int ox = blockIdx.x*blockDim.x-constant::atrous_radi*constant::level2step_size[level];
//
//			int ox,oy;
//
//			sy = blockIdx.y*blockDim.y/constant::level2step_size[level];
//			oy = sy-constant::atrous_radi*constant::level2step_size[level] + blockIdx.y*blockDim.y-sy;
//
//			sx = blockIdx.x*blockDim.x/constant::level2step_size[level];
//			ox = sx-constant::atrous_radi*constant::level2step_size[level] + blockIdx.x*blockDim.x-sx;
//
//
//			off = threadIdx.y*blockDim.x+threadIdx.x;
//			sy = off/constant::lx;
//			sx = off - sy*constant::lx;
//
//			sy *= constant::level2step_size[level];
//			sx *= constant::level2step_size[level];
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
//			shm[off]=input[blockIdx.z*640*480+(oy+sy)*640+(ox+sx)];
//
//			off += blockDim.x*blockDim.y;
//			if(off<constant::lx*constant::ly){
//				sy = off/constant::lx;
//				sx = off - sy*constant::lx;
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
//				shm[off]=input[blockIdx.z*640*480+(oy+sy)*640+(ox+sx)];
//			}
//			__syncthreads();
//
//
//			/* ---------- CONV ----------- */
//
//			off = (threadIdx.y+constant::atrous_radi)*constant::lx+threadIdx.x+constant::atrous_radi;
//
//			float sum = 0.0f;
//			float sum_weight = 0.0f;
//
//			float mid_depth = shm[off].z;
//			float mid_luma = shm[off].w;
//
//			float cur_depth;
//
//			float weight;
//			for(sy=0;sy<constant::atrous_length;sy++)
//			{
//				for(sx=0;sx<constant::atrous_length;sx++)
//				{
//					off = (threadIdx.y+sy)*constant::lx+threadIdx.x+sx;
//					cur_depth = shm[off].z;
//
//					weight = constant::atrous_coef[sy*constant::atrous_length+sx];
//					if(cur_depth==0.0f) weight = 0.0f;
//
//					if(mid_depth != 0.0f && (modi & 2) )
//					{
//						weight *= __expf(-0.5f*(abs(mid_depth-cur_depth)/(10.0f * 10.0f)));
//					}
//
//					if(modi & 4)
//					{
//						 weight *= __expf(-0.5f*(abs(mid_luma-shm[off].w)/(5.0f * 5.0f)));
//					}
//
//					sum += weight * cur_depth;
//					sum_weight += weight;
//				}
//			}
//
//			/* ---------- SAVE ----------- */
//
//			sx = blockIdx.x*blockDim.x + threadIdx.x;
//			sy = blockIdx.y*blockDim.y + threadIdx.y;
//	//
//			off = (sum_weight/constant::atrous_sum)*255.0f;
//			off = (off << 16 | off << 8 | off) & 0x00ffffff;
//	//
//	////			uchar4 u4 = make_uchar4(off,off,off,0);
//	////			surf3Dwrite<uchar4>(u4,surf::surfOutRef,sx*sizeof(uchar4),sy,blockIdx.z);
//	//
//			off = (threadIdx.y+constant::atrous_radi)*constant::lx+threadIdx.x+constant::atrous_radi;
//			shm[off].z = (sum_weight>0)?(sum/sum_weight):0;
////			surf3Dwrite<float4>(shm[off],surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
//			output[blockIdx.z*640*480+sy*640+sx] = shm[off];
//
//	//			float4 f4t = make_float4(0,0,500,0);
//	//			surf3Dwrite<float4>(f4t,surf::surfRefBuffer,sx*sizeof(float4),sy,0);
		}
	};

	__global__ void filterAtrousKernel(const ATrousEnhancer128 ate){ ate (); }
}

device::ATrousEnhancer128 atrousfilter;

void ATrousFilter::execute()
{

	device::filterAtrousKernel<<<grid,block>>>(atrousfilter);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}


void ATrousFilter::init()
{

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,1);

	atrousfilter.modi = 1 | 2 | 4 | 8;
	atrousfilter.level = 0;
	atrousfilter.input = (float4 *)getInputDataPointer(0);
	atrousfilter.output = (float4 *)getTargetDataPointer(0);

	printf("pointer xyzi atrous in: %d \n",atrousfilter.input);

}

void ATrousFilter::initAtrousConstants()
{
	using namespace device;

	unsigned int atrous_radi = radi;
	unsigned int atrous_length = 2*atrous_radi+1;
	float atrous_coef[atrous_length*atrous_length];
	cudaMemcpyToSymbol(constant::atrous_radi,&atrous_radi,sizeof(unsigned int));
	cudaMemcpyToSymbol(constant::atrous_length,&atrous_length,sizeof(unsigned int));


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


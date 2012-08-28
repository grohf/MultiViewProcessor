/*
 * FPFH.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#include "FPFH.h"
#include <helper_cuda.h>
#include <helper_image.h>
#include "../sink/pcd_io.h"
#include "utils.hpp"


namespace device
{
	namespace constant
	{
		__constant__ float radius;
	}
}

namespace device
{
		struct SPFHEstimator
		{

			enum
			{
				kxr = 3,
				kyr = kxr,

				kx = 32+kxr*2,
				ky = 24+kyr*2,

				kl = 2*kxr+1,

				nbins = 16
			};

			float4 *input_pos;
			float4 *input_normals;

			float *output_bins;

			unsigned int view;
			float radius;

		    __device__ __forceinline__ float
		    dotf43(const float4& v1,const float4& v2) const
		    {
		      return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
		    }

		    __device__ __forceinline__ float4
		    minusf43(const float4& v1, const float4& v2) const
		    {
		      return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0);
		    }

		    __device__ __host__ __forceinline__ float4
		    crossf43(const float4& v1, const float4& v2) const
		    {
		      return make_float4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x,0);
		    }

		    __device__ __forceinline__ float
		    lengthf43(const float4& v) const
		    {
		      return sqrt(dotf43(v, v));
		    }

			__device__ __forceinline__ void
			operator () () const
			{
				__shared__ float4 shm_pos[kx*ky];
				__shared__ float4 shm_normal[kx*ky];
//				__shared__ unsigned int shm_bins[kx*ky*16];


				int sx,sy,off;

				/* ---------- LOAD SHM ----------- */
				const int oy = blockIdx.y*blockDim.y-kyr;
				const int ox = blockIdx.x*blockDim.x-kxr;


				for(off=threadIdx.y*blockDim.x+threadIdx.x;off<kx*ky;off+=blockDim.x*blockDim.y)
				{
					sy = off/kx;
					sx = off - sy*kx;

					sy = oy + sy;
					sx = ox + sx;

					if(sx < 0) 		sx	=	0;
					if(sx > 639) 	sx 	= 639;
					if(sy < 0)		sy 	= 	0;
					if(sy > 479)	sy 	= 479;

					shm_pos[off] = input_pos[view*640*480+sy*640+sx];
					shm_normal[off] = input_normals[view*640*480+sy*640+sx];

				}
				__syncthreads();


				sx = threadIdx.x + blockIdx.x * blockDim.x;
				sy = threadIdx.y + blockIdx.y * blockDim.y;
				unsigned int mid_global_off = sy*640+sx;

				unsigned int mid_off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
				float4 mid_pos = shm_pos[mid_off];
				float4 mid_normal = shm_normal[mid_off];

				unsigned int point_count = 0;
				int mid_bin[16];
				for(int i=0;i<16;i++)
					mid_bin[i]=0;

				if(mid_pos.z==0)
				{
					for(int i=0;i<16;i++)
						mid_bin[i]=-1;

					return;
				}

//				for(sy=0;sy<(kl+1)/2;sy++)
				for(sy=0;sy<kl;sy++)
				{
					for(sx=0;sx<kl ;sx++)
					{
						off = (threadIdx.y+sy)*kx+threadIdx.x+sx;
						float4 cur_pos = shm_pos[off];
						float4 cur_normal = shm_normal[off];

						if(lengthf43(minusf43(mid_pos,cur_pos))>radius && mid_off==off && cur_pos.z==0)
						{
								continue;
						}

						point_count++;

						float4 ps,pt,ns,nt;

						if( dotf43(mid_normal, minusf43(cur_pos,mid_pos) ) <= dotf43(cur_normal, minusf43(mid_pos,cur_pos) ) )
						{
							ps = mid_pos;
							ns = mid_normal;

							pt = cur_pos;
							nt = cur_normal;
						}
						else
						{
							ps = cur_pos;
							ns = cur_normal;

							pt = mid_pos;
							nt = mid_normal;
						}

						float4 u = ns;
						float4 v = crossf43(minusf43(pt,ps),u);
						float4 w = crossf43(u,v);

						unsigned int idx = 0;

						if(dotf43(v,nt)>=0)
							idx += 1;

						float f2 = lengthf43( minusf43(pt,ps));
						if(f2 >= 20)
							idx += 2;

						if(dotf43(u,minusf43(pt,ps))/f2 >= 0)
							idx += 4;

						if(atan2f(dotf43(w,nt),dotf43(u,nt)) >= 0)
							idx += 8;

						mid_bin[idx]++;

//						if( !( (oy+sy) < 0 || (oy+sy) > 479 || (ox+sx) < 0 || (ox+sx) > 639) )
//						{
//							atomicAdd(&output_bins[((oy+sy)*640+(ox+sx))*16+idx],1.0f);
//						}
//							atomicAdd(&output_bins[mid_global_off*16+idx],1.0f);

					}
				}

				for(int i=0;i<16;i++)
					output_bins[mid_global_off*16+i] = ((float)mid_bin[i])/((float)point_count);


				if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
				{
					for(int i=0;i<16;i++)
						printf("%d | ",mid_bin[i]);

					printf("\n");

					for(int i=0;i<16;i++)
						printf("%f | ",output_bins[mid_global_off*16+i]);


				}

//				for(int i=0;i<16;i++)
//					output_bins[mid_global_off*16+i] =
//							(point_count>0)
//							?output_bins[mid_global_off*16+i]/(float)point_count
//							:-1.0f;



//			if(blockIdx.x==10 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y==10)
//			{
//				for(int i=0;i<16;i++)
//					printf("%f | ",output_bins[mid_global_off*16+i]);
//
//				printf("\n");
//			}

			}
		};
		__global__ void computeSPFH(const SPFHEstimator spfhe){ spfhe(); }


		struct SPFHNormalizer
		{
			float *inline_bins;

			__device__ __forceinline__ void
			operator () () const
			{
				int sx = threadIdx.x + blockIdx.x * blockDim.x;
				int sy = threadIdx.y + blockIdx.y * blockDim.y;

				int off = (sy*640+sx)*16;

				if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
				{
					for(int i=0;i<16;i++)
						printf("%f | ",inline_bins[off+i]);

					printf("\n");
				}

				float sum = 0.0;
				for(int i=0;i<16;i++)
					sum += inline_bins[off+i];

				for(int i=0;i<16;i++)
					inline_bins[off+i] /= sum;


				if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
				{
					for(int i=0;i<16;i++)
						printf("%f | ",inline_bins[off+i]);

					printf("\n");
				}
			}
		};
		__global__ void normalizeSPFH(const SPFHNormalizer spfhn){ spfhn(); }


		struct FPFHEstimator
		{
			enum
			{
				kxr = 3,
				kyr = kxr,

				kx = 32+kxr*2,
				ky = 24+kyr*2,

				kl = 2*kxr+1,

				bins=4
			};


			float *spfh_input;
			float *fpfh_output;

			float4 *input_pos;

//			unsigned int bins;
			unsigned int view;


		    __device__ __forceinline__ float4
		    minusf43(const float4& v1, const float4& v2) const
		    {
		      return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0);
		    }

		    __device__ __forceinline__ float
		    dotf43(const float4& v1,const float4& v2) const
		    {
		      return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
		    }

		    __device__ __forceinline__ float
		    lengthf43(const float4& v) const
		    {
		      return sqrt(dotf43(v, v));
		    }

			__device__ __forceinline__ void
			operator () () const
			{
//				__shared__ float shm[kx*ky*bins];
//				__shared__ float4 shm_pos[kx*ky];

				int sx,sy,off,global_off;

				/* ---------- LOAD SHM ----------- */
				const int oy = blockIdx.y*blockDim.y-kyr;
				const int ox = blockIdx.x*blockDim.x-kxr;
//
//
//				for(off=threadIdx.y*blockDim.x+threadIdx.x;off<kx*ky;off+=blockDim.x*blockDim.y)
//				{
//					sy = off/kx;
//					sx = off - sy*kx;
//
//					sy = oy + sy;
//					sx = ox + sx;
//
//					if(sx < 0) 		sx	=	0;
//					if(sx > 639) 	sx 	= 639;
//					if(sy < 0)		sy 	= 	0;
//					if(sy > 479)	sy 	= 479;
//
//					shm_pos[off] = input_pos[sy*640+sx];
//				}
//				__syncthreads();


				sx = threadIdx.x + blockIdx.x * blockDim.x;
				sy = threadIdx.y + blockIdx.y * blockDim.y;
				unsigned int mid_global_off = sy*640+sx;

//				unsigned int mid_off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;

				float mid_bins[16];
				for(int i=0;i<16;i++)
					mid_bins[i] = 0.0f;

				float4 mid_pos = input_pos[view*640*480+mid_global_off];

				unsigned int point_count = 0;
				for(sy=(oy<0)?(-oy):0;sy<kl;sy++)
				{
					for(sx=(ox<0)?(-ox):0;sx<kl ;sx++)
					{
						float4 cur_pos = input_pos[view*640*480+(oy+sy)*640+(ox+sx)];
						float weight = 1.0f/lengthf43(minusf43(mid_pos,cur_pos));
						for(int i=0;i<16;i++)
						{
							mid_bins[i] += weight * spfh_input[(sy*kx+sx)*16+i];
							point_count++;
						}
					}
				}

				for(int i=0;i<16;i++)
					mid_bins[i] /= (float)point_count;

			}

		};


}

device::SPFHEstimator spfhEstimator1;
device::SPFHEstimator spfhEstimator2;

//device::SPFHNormalizer spfhNormailizer1;
//device::SPFHNormalizer spfhNormailizer2;

device::FPFHEstimator fpfhEstimator1;
device::FPFHEstimator fpfhEstimator2;

void FPFH::init()
{
	spfhEstimator1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	spfhEstimator1.input_normals = (float4 *)getInputDataPointer(PointNormal);
	spfhEstimator1.output_bins = (float *)getTargetDataPointer(SPFHistogram1);

	spfhEstimator2.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	spfhEstimator2.input_normals = (float4 *)getInputDataPointer(PointNormal);
	spfhEstimator2.output_bins = (float *)getTargetDataPointer(SPFHistogram2);


//	spfhNormailizer1.inline_bins = (float *)getTargetDataPointer(SPFHistogram1);
//
//	spfhNormailizer2.inline_bins = (float *)getTargetDataPointer(SPFHistogram2);


	fpfhEstimator1.spfh_input = (float *)getTargetDataPointer(SPFHistogram1);
	fpfhEstimator1.fpfh_output = (float *)getTargetDataPointer(FPFHistogram1);

	fpfhEstimator2.spfh_input = (float *)getTargetDataPointer(SPFHistogram2);
	fpfhEstimator2.fpfh_output = (float *)getTargetDataPointer(FPFHistogram2);



	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y);


}

void FPFH::execute()
{

	float radii[] = {1.5f,2.0f,2.5f};

//	device::SPFHEstimator *cur_estimator = &spfhEstimator1;

	for(int v=0;v<1;v++)
	{
		spfhEstimator1.view = v;
		spfhEstimator2.view = v;

//		radius = radii[0];
//		cudaMemcpy(&device::constant::radius,&radius,sizeof(float),cudaMemcpyHostToDevice);

		spfhEstimator1.radius=radii[0];
		device::computeSPFH<<<grid,block>>>(spfhEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

//		device::normalizeSPFH<<<grid,block>>>(spfhNormailizer1);
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());



		for(int ri=0;ri<1;ri++)
		{

		}

	}

//	sp

}



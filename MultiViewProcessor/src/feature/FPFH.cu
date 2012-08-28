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

			uint16_t *output_bins;

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
				__shared__ unsigned int shm_bins[kx*ky*16];


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

					shm_pos[off] = input_pos[blockIdx.z*640*480+sy*640+sx];
					shm_normal[off] = input_normals[blockIdx.z*640*480+sy*640+sx];

				}
				__syncthreads();

				unsigned int mid_off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
				float4 mid_pos = shm_pos[off];
				float4 mid_normal = shm_normal[off];

				for(sy=0;sy<(kl+1)/2;sy++)
				{
					for(sx=0;sx<kl && mid_off!=off ;sx++)
					{
						off = (threadIdx.y+sy)*kx+threadIdx.x+sx;
						float4 cur_pos = shm_pos[off];
						float4 cur_normal = shm_normal[off];
//						unsigned int (*cur_bins)[16] = &shm_bins[off];

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


						atomicInc(&shm_bins[mid_off*16+idx],1);
						atomicInc(&shm_bins[off*16+idx],1);
					}
				}

				sx = threadIdx.x + blockIdx.x * blockDim.x;
				sy = threadIdx.y + blockIdx.y * blockDim.y;

				//TODO COPY all 16 values
				for(int i=0;i<16;i++)
					output_bins[(blockDim.z*640*480+sy*640+sx)*16+i] = shm_bins[mid_off*16+i];


			}
		};


		struct FPFHEstimator
		{
			enum
			{
				kxr = 3,
				kyr = kxr,

				kx = 32+kxr*2,
				ky = 24+kyr*2,

				kl = 2*kxr+1
			};

			uint16_t *bins_spf;
//			unsigned int

		};



void FPFH::init()
{

}

void FPFH::execute()
{

}

FPFH::FPFH()
{
	// TODO Auto-generated constructor stub

}

FPFH::~FPFH()
{
	// TODO Auto-generated destructor stub
}


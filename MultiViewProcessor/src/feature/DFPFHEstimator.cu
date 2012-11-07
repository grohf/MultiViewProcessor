/*
 * FPFHEstimator2.cpp
 *
 *  Created on: Sep 25, 2012
 *      Author: avo
 */

#include "DFPFHEstimator.h"
#include <helper_cuda.h>
#include <helper_image.h>

#include <thrust/remove.h>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "point_info.hpp"


#include "../sink/pcd_io.h"
#include "utils.hpp"

namespace device
{

	struct SDPFHEstimator
	{
		enum
		{
			points_per_block = 32,
			WARP_SIZE = 32,
			dx = points_per_block * WARP_SIZE,
			dy = 1,
			features = 3,
			bins_per_feature = 11,
			bins = features * bins_per_feature,

			rx = 2,
			ry = rx,
			rxl = 2*rx+1,
			ryl = 2*ry+1,

		};

		float4 *input_pos;
		float4 *input_normals;



		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float4 shm_pos[points_per_block];
			__shared__ float4 shm_normal[points_per_block];
			__shared__ float shm_histo[points_per_block*bins];

			__shared__ unsigned int shm_off[points_per_block*2];

			unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
			unsigned int wid = tid / WARP_SIZE;
			unsigned int wtid = tid - wid * WARP_SIZE;

//			unsigned int oy = (blockIdx.x * points_per_block + wid)/640;
//			unsigned int ox = (blockIdx.x * points_per_block + wid) - oy*640;

			if(tid<points_per_block)
			{
				shm_off[tid] = (blockIdx.x * points_per_block + tid)/640;
				shm_off[2*tid] = (blockIdx.x * points_per_block + tid) - shm_off[tid]*640;

				shm_pos[tid] = input_pos[shm_off[tid*2]*640+shm_off[tid]];
				shm_normal[tid] = input_normals[shm_off[tid*2]*640+shm_off[tid]];
			}

			for(int j=wtid;j<rxl*ryl;j+=WARP_SIZE)
			{

			}




		}
	};

}


void
DFPFHEstimator::init()
{

}


void
DFPFHEstimator::execute()
{

}





DFPFHEstimator::DFPFHEstimator()
{
	// TODO Auto-generated constructor stub

}

DFPFHEstimator::~DFPFHEstimator()
{
	// TODO Auto-generated destructor stub
}


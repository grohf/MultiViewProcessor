/*
 * GlobalFineRegistrationEstimator.cpp
 *
 *  Created on: Dec 7, 2012
 *      Author: avo
 */

#include "GlobalFineRegistrationEstimator.h"


namespace device
{
	struct SparseAbCreater : public FeatureBaseKernel
	{
		enum
		{
			dx = 1024,
		};

		float4 *input_pos;
		float4 *input_normal;
		float *input_tm;
		SensorInfoV2 *sinfo;

		unsigned int *output_rowInfo;
		float *output_sparseA;
		float *output_b;

		unsigned int n_view;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_tm_src[TMatrixDim];
			__shared__ float shm_tm_target[TMatrixDim];
			unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			if(tid==0)
			{
				unsigned int x = blockIdx.z;
				unsigned int y = 0;
				unsigned int i = 1;
				while(x >= n_view - i)
				{
					x -= n_view-i;
					i++;
					y++;
				}
				view_src = y;
				view_target = x+y+1;
			}
			__syncthreads();

			if(tid<TMatrixDim)
			{
				shm_tm_src[tid] = input_tm[view_src*TMatrixDim+tid];
				shm_tm_target[tid] = input_tm[view_target*TMatrixDim+tid];
			}
			__syncthreads();



		}
	};
	__global__ void computeSparseAb(const SparseAbCreater A) { A (); }
}

void
GlobalFineRegistrationEstimator::init()
{

}

void
GlobalFineRegistrationEstimator::execute()
{

}

GlobalFineRegistrationEstimator::GlobalFineRegistrationEstimator(unsigned int n_view) : n_view(n_view)
{
	// TODO Auto-generated constructor stub

}

GlobalFineRegistrationEstimator::~GlobalFineRegistrationEstimator()
{
	// TODO Auto-generated destructor stub
}


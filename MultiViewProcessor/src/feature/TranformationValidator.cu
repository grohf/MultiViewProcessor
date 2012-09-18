/*
 * TranformationValidator.cpp
 *
 *  Created on: Sep 18, 2012
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>

#include "TranformationValidator.h"


namespace device
{
	struct TransforamtionErrorEstimator
	{
		enum
		{
			dx = 32,
			dy = 24,
			WARP_SIZE = 32,
		};

		float4 *pos;

		unsigned int matrices_offset;
		float *transformation_matrices;

		int *transformationMetaData;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ unsigned int shm_matrix_dimx;
			__shared__ unsigned int shm_length;
			__shared__ unsigned int shm_off;

			__shared__ int idx[32];
			__shared__ float shm_tm[32*12];

			if(threadIdx.x==0){
				shm_length = transformationMetaData[0];
				shm_off = 0;
			}

			while(shm_off*32<shm_length)
			{

			}

			for(int tid = threadIdx.y*blockDim.x+threadIdx.x;tid<32*12;tid+=blockDim.x*blockDim.y)
			{
				unsigned int wid = tid/WARP_SIZE;
				unsigned int wtid = tid - wid*WARP_SIZE;

				shm_tm[wid*32+tid] = 0;//transformationMetaData[]

			}



//			float4 local_pos = pos[]

			if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
				printf("%d \n",transformationMetaData[0]);


		}

	};
	__global__ void computeTransformationError(TransforamtionErrorEstimator tee) { tee(); }
}

device::TransforamtionErrorEstimator transformationErrorestimator;

void
TranformationValidator::init()
{
	transformationErrorestimator.pos = (float4 *)getInputDataPointer(WorldCoords);
	transformationErrorestimator.transformation_matrices = (float *)getInputDataPointer(TransforamtionMatrices);
	transformationErrorestimator.transformationMetaData = (int *)getInputDataPointer(TransformationInfoList);

	block = dim3(transformationErrorestimator.dx,transformationErrorestimator.dy);
	grid = dim3(640/block.x,480/block.y);

}

void TranformationValidator::execute()
{

	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}


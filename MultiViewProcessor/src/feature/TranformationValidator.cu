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
#include <math.h>


#include "utils.hpp"

#include "TranformationValidator.h"

#ifndef MAXVIEWS
#define MAXVIEWS 8
#endif

namespace device
{
	namespace constant
	{
		__constant__ unsigned int idx2x[(MAXVIEWS*MAXVIEWS-1)/2];
		__constant__ unsigned int idx2y[(MAXVIEWS*MAXVIEWS-1)/2];
	}


	struct TransforamtionErrorEstimator
	{
		enum
		{
			dx = 32,
			dy = 24,
			WARP_SIZE = 32,
		};

		float4 *coords;
		float4 *normals;
		SensorInfo *sinfo;

		unsigned int matrix_dim_x;
		float *transformation_matrices;
		int *transformationMetaData;

		float *output_errorList;
		unsigned int *output_errorListIdx;

		float dist_threshold;
		float angle_threshold;

//	    __device__ __forceinline__ float
//	    dotf43(const float4& v1,const float4& v2)
//	    {
//	      return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
//	    }
//
//	    __device__ __forceinline__ float
//	    lengthf43(const float4& v)
//	    {
//	      return sqrt(dotf43(v, v));
//	    }

		__device__ __forceinline__ void
		operator () () const
		{
//			__shared__ unsigned int shm_matrix_dimx;
			__shared__ unsigned int shm_length;
//			__shared__ unsigned int shm_off;

			__shared__ int shm_idx[32];
			__shared__ float shm_tm[32*12];

			__shared__ float shm_error_buffer[dx*dy];

			if(threadIdx.x==0){
				shm_length = transformationMetaData[0];
//				shm_off = 0;
			}
			__syncthreads();

			unsigned int sx = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int sy = blockIdx.y*blockDim.y+threadIdx.y;

			//TODO match blockidx.z to view_src
			float4 pos = coords[blockIdx.z*640*480+sy*640+sx];
			float4 normal = normals[blockIdx.z*640*480+sy*640+sx];

			unsigned int tid = threadIdx.y*blockDim.x+threadIdx.x;

			for(int soff=0;soff<shm_length;soff+=32)
//			for(int soff=0;soff<32;soff+=32)
			{
				if(blockIdx.x==1 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y == 5)
				{
					printf("round: %d \n",soff/32);
//					printf("%d \n",transformationMetaData[0]);
//					printf("matrix_dim_x: %d",matrix_dim_x);
				}


				if(tid<32 && (tid+soff<shm_length) )
				{
					shm_idx[tid] = transformationMetaData[1+blockIdx.z*matrix_dim_x+tid+soff];
				}
				__syncthreads();

				for(int t = tid;t<32*12;t+=blockDim.x*blockDim.y)
				{
					unsigned int wid = t/WARP_SIZE;
					unsigned int wtid = t - wid*WARP_SIZE;

					if(wid+soff<shm_length)
					{
						unsigned int idx = shm_idx[wtid];
						shm_tm[wid*32+wtid] = transformation_matrices[(blockIdx.z*12+wid)*matrix_dim_x+idx];
					}
				}
				__syncthreads();

//				if(blockIdx.x==1 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y == 5)
//				{
//					printf("(%d) | ",shm_idx[3]);
//					for(int i=0;i<12;i++)
//					{
//					 	printf("%f |",shm_tm[i*32+3]);
//					}
//					printf("\n");
//				}

				for(int i=0; (i<32) || (soff+i < shm_length) ;i++)
				{
					shm_error_buffer[tid] = 0.f;

					if(pos.z == 0)
					{

						float3 pos_m,n_m;
						pos_m.x = shm_tm[0*32+i]*pos.x + shm_tm[1*32+i]*pos.y+ shm_tm[2*32+i]*pos.z + shm_tm[9*32+i];
						pos_m.y = shm_tm[3*32+i]*pos.x + shm_tm[4*32+i]*pos.y+ shm_tm[5*32+i]*pos.z + shm_tm[10*32+i];
						pos_m.z = shm_tm[6*32+i]*pos.x + shm_tm[7*32+i]*pos.y+ shm_tm[8*32+i]*pos.z + shm_tm[11*32+i];

						n_m.x = shm_tm[0*32+i]*normal.x + shm_tm[1*32+i]*normal.y+ shm_tm[2*32+i]*normal.z;
						n_m.y = shm_tm[3*32+i]*normal.x + shm_tm[4*32+i]*normal.y+ shm_tm[5*32+i]*normal.z;
						n_m.z = shm_tm[6*32+i]*normal.x + shm_tm[7*32+i]*normal.y+ shm_tm[8*32+i]*normal.z;


						int cx = (pos_m.x*120.f)/(0.2048f*pos_m.z)+320;
						int cy = (pos_m.y*120.f)/(0.2048f*pos_m.z)+240;

						if(cx>0 && cx<640 && cy>0 && cy<480)
						{

							float4 pos_d = coords[1*640*480+cy*640+cx];
							float4 n_d = normals[1*640*480+cy*640+cx];

							if( pos_d.z == 0)
								continue;

							float3 dv = make_float3(pos_m.x-pos_d.x,pos_m.y-pos_d.y,pos_m.z-pos_d.z);
							float angle = n_m.x * n_d.x + n_m.y * n_d.y + n_m.z * n_d.z;

							if( norm(dv) < dist_threshold && (((angle>=0)?angle:-angle) < angle_threshold) )
							{
								double error = dv.x*n_d.x * dv.y*n_d.y * dv.z*n_d.z;
								shm_error_buffer[tid] = error * error;
							}
						}
					}
					__syncthreads();

					float sum = 0.f;
					if(dx*dy>=1024) { if(tid < 512) { shm_error_buffer[tid] = sum = sum + shm_error_buffer[tid+512];} __syncthreads(); }
					if(dx*dy>=512) 	{ if(tid < 256) { shm_error_buffer[tid] = sum = sum + shm_error_buffer[tid+256];} __syncthreads(); }
					if(dx*dy>=256) 	{ if(tid < 128) { shm_error_buffer[tid] = sum = sum + shm_error_buffer[tid+128];} __syncthreads(); }
					if(dx*dy>=128) 	{ if(tid < 64) 	{ shm_error_buffer[tid] = sum = sum + shm_error_buffer[tid+64];	} __syncthreads(); }
					if(tid < 32)
					{
						volatile float *smem = shm_error_buffer;
						if(dx*dy>=64) 	{ smem[tid] = sum = sum + smem[tid+32]; }
						if(dx*dy>=32) 	{ smem[tid] = sum = sum + smem[tid+16]; }
						if(dx*dy>=16) 	{ smem[tid] = sum = sum + smem[tid+8]; 	}
						if(dx*dy>=8) 	{ smem[tid] = sum = sum + smem[tid+4]; 	}
						if(dx*dy>=4) 	{ smem[tid] = sum = sum + smem[tid+2]; 	}
						if(dx*dy>=2) 	{ smem[tid] = sum = sum + smem[tid+1]; 	}

						if(tid==0)
						{
//							output_errorList[(blockIdx.z*matrix_dim_x+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = smem[0];
//							output_errorList[(blockIdx.z*matrix_dim_x+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = blockIdx.y*gridDim.x+blockIdx.x;
							output_errorList[(blockIdx.z*matrix_dim_x+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = 1;
//							output_errorList[(i)*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x] = smem[0];
						}
					}

				}

				__syncthreads();


//				if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y == 0)
//					shm_off++;
//
//				__syncthreads();
			}



//			float4 local_pos = pos[]

//			if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
//				printf("%d \n",transformationMetaData[0]);


		}

	};
	__global__ void computeTransformationError(TransforamtionErrorEstimator tee) { tee(); }


	template<unsigned int blockSize>
	struct ErrorListSumCalculator
	{
		enum
		{
			dx = blockSize,
		};

		float *errorList;
		int *listMetaData;

		unsigned int n;
		unsigned int n_rsac;

	    __device__ __forceinline__ void
		operator () () const
		{
	    	__shared__ float shm[dx];
	    	__shared__ unsigned int shm_length;


	    	unsigned int tid = threadIdx.x;
	    	unsigned int i = tid;

	    	if(tid==0)
	    		shm_length = listMetaData[0];

	    	__syncthreads();
	    	if(blockIdx.x>shm_length)
	    	{
	    		errorList[(blockIdx.y*n_rsac+blockIdx.x)*n] = numeric_limits<float>::max();
				return;
	    	}

	    	float sum = 0.f;
	    	while(i<n)
	    	{
	    		sum += errorList[(blockIdx.y*n_rsac+blockIdx.x)*n+i];
	    		if( (i+dx) < n )
	    			sum += errorList[(blockIdx.y*n_rsac+blockIdx.x)*n+i+dx];

	    		i += dx*2;
	    	}

	    	shm[tid] = sum;
	    	__syncthreads();


	    	if(dx>=1024){ if(tid<512) 	{shm[tid] = sum = sum + shm[tid + 512];	} __syncthreads(); }
	    	if(dx>=512)	{ if(tid<256) 	{shm[tid] = sum = sum + shm[tid + 256];	} __syncthreads(); }
	    	if(dx>=256)	{ if(tid<128) 	{shm[tid] = sum = sum + shm[tid + 128];	} __syncthreads(); }
	    	if(dx>=128)	{ if(tid<64) 	{shm[tid] = sum = sum + shm[tid + 64];	} __syncthreads(); }

	    	if(tid<32)
	    	{
	    		volatile float *smem = shm;
	    		if(dx>=64) 	{ smem[tid] = sum = sum + smem[tid + 32]; 	}
	    		if(dx>=32) 	{ smem[tid] = sum = sum + smem[tid + 16]; 	}
	    		if(dx>=16) 	{ smem[tid] = sum = sum + smem[tid + 8];	}
	    		if(dx>=8) 	{ smem[tid] = sum = sum + smem[tid + 4]; 	}
	    		if(dx>=4)	{ smem[tid] = sum = sum + smem[tid + 2]; 	}
	    		if(dx>=2) 	{ smem[tid] = sum = sum + smem[tid + 1]; 	}

	    		if(tid==0)
	    			errorList[(blockIdx.y*n_rsac)*n+blockIdx.x] = sum;
	    	}


		}
	};
	__global__ void computeErrorListSum(ErrorListSumCalculator<256> elsc) { elsc(); }


	template<unsigned int bx>
	struct ErrorListMinimumPicker
	{
		enum
		{
			dx = bx,
		};

		float *errorList;

		float *output_minimumError;
		unsigned int *output_minimumErrorIdx;

		unsigned int n;
		unsigned int n_rsac;


		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_error[dx];
			__shared__ unsigned int shm_idx[dx];

	    	unsigned int tid = threadIdx.x;
	    	unsigned int i = tid;

	    	float min = 0.f;
	    	while(i<n)
	    	{
	    		float tmp = errorList[blockIdx.x*n_rsac*n+i];
	    		if(tmp<min)
	    			min = tmp;

	    		if((i+dx) < n )
	    		{
	    			tmp = errorList[blockIdx.x*n_rsac*n+i+dx];
	    			if(tmp<min)
	    				min = tmp;
	    		}

	    		i += dx*2;
	    	}

	    	shm_error[tid] = min;
	    	__syncthreads();

	    	if(dx>=1024){ if(tid<512) 	{ float tmp = shm_error[tid + 512]; if(tmp<min) shm_error[tid] = min = tmp; } __syncthreads(); }
	    	if(dx>=512)	{ if(tid<256) 	{ float tmp = shm_error[tid + 512]; if(tmp<min) shm_error[tid] = min = tmp; } __syncthreads(); }
	    	if(dx>=1024){ if(tid<512) 	{ float tmp = shm_error[tid + 512]; if(tmp<min) shm_error[tid] = min = tmp; } __syncthreads(); }


		}


	};
}

device::TransforamtionErrorEstimator transformationErrorestimator;
device::ErrorListSumCalculator<256> errorListSumcalculator256;

void
TranformationValidator::init()
{
	transformationErrorestimator.coords = (float4 *)getInputDataPointer(WorldCoords);
	transformationErrorestimator.normals = (float4 *)getInputDataPointer(Normals);

	//TODO: set to constant memory??
	transformationErrorestimator.sinfo = (SensorInfo *)getInputDataPointer(SensorInfoList);

	transformationErrorestimator.transformation_matrices = (float *)getInputDataPointer(TransforamtionMatrices);
	transformationErrorestimator.transformationMetaData = (int *)getInputDataPointer(TransformationInfoList);

	transformationErrorestimator.output_errorList = (float *)getTargetDataPointer(ErrorList);
	transformationErrorestimator.output_errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);
//	transformationErrorestimator.matrix_dim_x = ((n_views-1)*n_views)/2 * n_rsac;
	transformationErrorestimator.matrix_dim_x = n_rsac;

	transformationErrorestimator.dist_threshold = 300;
	transformationErrorestimator.dist_threshold = 0.5f;


	errorListSumcalculator256.errorList = (float *)getTargetDataPointer(ErrorList);
	errorListSumcalculator256.listMetaData = (int *)getInputDataPointer(TransformationInfoList);
	errorListSumcalculator256.n = grid.x * grid.y;
	errorListSumcalculator256.n_rsac = n_rsac;

}

void TranformationValidator::execute()
{

	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

//	int load = n_rsac;
//	float *h_tmp = (float *)malloc(load*grid.x*grid.y*sizeof(float));
//	checkCudaErrors( cudaMemcpy(h_tmp,transformationErrorestimator.output_errorList,load*grid.x*grid.y*sizeof(float),cudaMemcpyDeviceToHost));
//
//	for(int j=0;j<load;j++)
//	{
//		printf("j: %d -> ",j);
//		for(int i=0;i<5;i++)
//		{
//			printf("%f + ",h_tmp[j*grid.x*grid.y+i]);
//		}
//		printf("\n");
//	}

	device::computeErrorListSum<<<n_rsac,errorListSumcalculator256.dx>>>(errorListSumcalculator256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int load = 5;
	float *h_tmp2 = (float *)malloc(load*sizeof(float));
	checkCudaErrors( cudaMemcpy(h_tmp2,transformationErrorestimator.output_errorList,load*sizeof(float),cudaMemcpyDeviceToHost));

	for(int j=0;j<load;j++)
	{
		printf("(%d/%f) | ",j,h_tmp2[j]);
	}
	printf("\n");


}

TranformationValidator::TranformationValidator(unsigned int n_views,unsigned int n_rsac)
: n_views(n_views), n_rsac(n_rsac)
{

	block = dim3(transformationErrorestimator.dx,transformationErrorestimator.dy);
	grid = dim3(640/block.x,480/block.y);

	DeviceDataParams errorListParams;
	errorListParams.elements = ((n_views-1)*n_views)/2 * n_rsac * grid.x * grid.y;
	errorListParams.element_size = sizeof(float);
	errorListParams.elementType = FLOAT1;
	errorListParams.dataType = Point;
	addTargetData(addDeviceDataRequest(errorListParams),ErrorList);

	printf("elements: %d \n",errorListParams.elements);

	DeviceDataParams errorListIdxParams;
	errorListIdxParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
	errorListIdxParams.element_size = sizeof(unsigned int);
	errorListIdxParams.elementType = UINT1;
	errorListIdxParams.dataType = Indice;
	addTargetData(addDeviceDataRequest(errorListIdxParams),ErrorListIndices);

}


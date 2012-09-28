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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "point_info.hpp"

#include "utils.hpp"

#include "TranformationValidator.h"

#ifndef MAXVIEWS
#define MAXVIEWS 8
#endif

namespace device
{
	namespace constant
	{
		__constant__ unsigned int idx2d[(MAXVIEWS*MAXVIEWS-1)/2];
		__constant__ unsigned int idx2m[(MAXVIEWS*MAXVIEWS-1)/2];
	}


	struct TransforamtionErrorEstimator
	{
		enum
		{
			dx = 32,
			dy = 16,
			WARP_SIZE = 32,

			dbx = 9,
			dby = 12,
			dbtid = 24,
		};

		float4 *coords;
		float4 *normals;
		SensorInfo *sinfo;

		unsigned int n_rsac;
		float *transformation_matrices;
		int *transformationMetaData;

		float *output_errorTable;
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


			unsigned int tid = threadIdx.y*blockDim.x+threadIdx.x;

			if(blockIdx.x==0 && blockIdx.y==0 && tid == 0)
				printf("inkernel! \n");

			if(tid==0){
				shm_length = transformationMetaData[0];
				if(blockIdx.x==0 && blockIdx.y==0) printf("length_inkernel %d \n",shm_length);
			}
			__syncthreads();

			unsigned int sx = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int sy = blockIdx.y*blockDim.y+threadIdx.y;

			//TODO match blockidx.z to view_src
			float4 pos = coords[blockIdx.z*640*480+sy*640+sx];
			float4 normal = normals[blockIdx.z*640*480+sy*640+sx];


			for(int soff=0;soff<shm_length;soff+=32)
//			for(int soff=0;soff<32;soff+=32)
			{
//				if(blockIdx.x==1 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y == 5)
//				{
//					printf("round: %d \n",soff/32);
////					printf("%d \n",transformationMetaData[0]);
////					printf("matrix_dim_x: %d",matrix_dim_x);
//				}


				if(tid<32 && (tid+soff<shm_length) )
				{
					shm_idx[tid] = transformationMetaData[1+blockIdx.z*n_rsac+tid+soff];
				}
				__syncthreads();

				if(blockIdx.x==0 && blockIdx.y==0 && tid < 32)
				{
					output_errorListIdx[blockIdx.z*n_rsac+soff+tid] = shm_idx[tid];
				}

				for(int t = tid;t<32*12;t+=blockDim.x*blockDim.y)
				{
					unsigned int wid = t/WARP_SIZE;
					unsigned int wtid = t - wid*WARP_SIZE;

					if(wid+soff<shm_length)
					{
						unsigned int idx = shm_idx[wtid];
						shm_tm[wid*32+wtid] = transformation_matrices[(blockIdx.z*12+wid)*n_rsac+idx];
					}
				}
				__syncthreads();

//				if(blockIdx.x==1 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y == 5)
//				if(blockIdx.x==10 && blockIdx.y==10 && tid == 10)
//				{
//					for(int i=0;i<12;i++)
//					{
//					 	printf("%f |",shm_tm[i*32+0]);
//					}
//					printf("\n");
//				}

				for(int i=0; (i<32) && (soff+i < shm_length) ;i++)
//				for(int i=0; i<32 ;i++)
				{
					shm_error_buffer[tid] = 0.f;

					if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0) printf("pos: %f %f %f \n",pos.x,pos.y,pos.z);
					if(pos.z != 0)
					{

						float3 pos_m,n_m;
						pos_m.x = shm_tm[0*32+i]*pos.x + shm_tm[1*32+i]*pos.y + shm_tm[2*32+i]*pos.z + shm_tm[9*32+i];
						pos_m.y = shm_tm[3*32+i]*pos.x + shm_tm[4*32+i]*pos.y + shm_tm[5*32+i]*pos.z + shm_tm[10*32+i];
						pos_m.z = shm_tm[6*32+i]*pos.x + shm_tm[7*32+i]*pos.y + shm_tm[8*32+i]*pos.z + shm_tm[11*32+i];

						n_m.x = shm_tm[0*32+i]*normal.x + shm_tm[1*32+i]*normal.y+ shm_tm[2*32+i]*normal.z;
						n_m.y = shm_tm[3*32+i]*normal.x + shm_tm[4*32+i]*normal.y+ shm_tm[5*32+i]*normal.z;
						n_m.z = shm_tm[6*32+i]*normal.x + shm_tm[7*32+i]*normal.y+ shm_tm[8*32+i]*normal.z;

//						if(blockIdx.x==10 && blockIdx.y==11 && tid == 0 && i==0)
//						{
//							for(int d=0;d<12;d++)
//								printf("%f | ",shm_tm[d*32+i]);
//							printf("\n");
//							printf("pos_m: %f %f %f \n",pos_m.x,pos_m.y,pos_m.z);
//						}
						if(pos_m.z == 0 || isReconstructed(pos.w))
							continue;

						float cxf = ((pos_m.x*120.f)/(0.2048f*pos_m.z))+320+1;
						float cyf = ((pos_m.y*120.f)/(0.2048f*pos_m.z))+240+1;

						int cx = cxf;
						int cy = cyf;

						if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
							printf("%f %f -> %d %d | %f \n",cxf,cyf,cx,cy,(cxf-cx));

						cx += ((cxf-(float)cx)>=0.5)?1:0;
						cy += ((cyf-(float)cy)>=0.5)?1:0;

//						if(blockIdx.x==10 && blockIdx.y==11 && tid == 0 && i==0)
//							printf("%d %d \n",cx,cy);

						if(cx>0 && cx<640 && cy>0 && cy<480)
						{

							float4 pos_d = coords[1*640*480+cy*640+cx];
							float4 n_d = normals[1*640*480+cy*640+cx];

							if( pos_d.z == 0 || isReconstructed(pos_d.w))
								continue;

							float3 dv = make_float3(pos_m.x-pos_d.x,pos_m.y-pos_d.y,pos_m.z-pos_d.z);
							float angle = n_m.x * n_d.x + n_m.y * n_d.y + n_m.z * n_d.z;

//							float length =
							if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
							{
								printf("pos_m: %f %f %f (%d/%d)\n",pos_m.x,pos_m.y,pos_m.z,sx,sy);
								printf("pos_d: %f %f %f (%d/%d)\n",pos_d.x,pos_d.y,pos_d.z,cx,cy);
								printf("dv_length: %f dist_tresh: %f \n",norm(dv),dist_threshold);
							}

							//TODO: Bring angle back
							if( norm(dv) < dist_threshold)// && (((angle>=0)?angle:-angle) < angle_threshold) )
							{
								double error = dv.x*n_d.x + dv.y*n_d.y + dv.z*n_d.z;
								shm_error_buffer[tid] = error * error;
								if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
								{
									printf("dv: %f %f %f \n",dv.x,dv.y,dv.z);
									printf("nd: %f %f %f \n",n_d.x,n_d.y,n_d.z);
									printf("error: %f error^2: %f \n",error,error*error);
								}
//								shm_error_buffer[tid] = 0;// * error;
							}
						}
					}

					__syncthreads();

					if(blockIdx.x==dbx && blockIdx.y==dby && i==0 && tid==dbtid)
					{
						printf("errorList inBlock: ");
						for(int io=0;io<dx*dy;io++)
						{
							if(shm_error_buffer[io]>1.f)printf("(%d/%f) | ",io,shm_error_buffer[io]);
						}
						printf("\n");
					}

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
							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = sum;

							if(blockIdx.x==dbx && blockIdx.y==dby && i==0)
								printf("(%d) block_sum: %f \n",i,sum);

//							output_errorList[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = blockIdx.y*gridDim.x+blockIdx.x;
//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = i/(gridDim.x*gridDim.y);
//							output_errorList[(i)*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x] = smem[0];
//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = soff + i;
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

		float *input_errorTable;

		float *output_errorList;
		int *listMetaData;

		unsigned int gxy;
		unsigned int n_rsac;

	    __device__ __forceinline__ void
		operator () () const
		{
	    	__shared__ float shm[dx];
	    	__shared__ unsigned int shm_length;


	    	unsigned int tid = threadIdx.x;
	    	unsigned int i = tid;

	    	if(tid==0)
	    	{
	    		shm_length = listMetaData[0];
//	    		printf("sumKernel:length: %d \n",shm_length);
	    	}

	    	__syncthreads();
	    	if(blockIdx.x>shm_length)
	    	{
	    		output_errorList[(blockIdx.y*n_rsac+blockIdx.x)] = numeric_limits<float>::max();
				return;
	    	}

	    	float sum = 0.f;
	    	while(i<gxy)
	    	{
	    		sum += input_errorTable[(blockIdx.y*n_rsac+blockIdx.x)*gxy+i];
	    		if( (i+dx) < gxy )
	    			sum += input_errorTable[(blockIdx.y*n_rsac+blockIdx.x)*gxy+i+dx];

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
	    			output_errorList[(blockIdx.y*n_rsac)+blockIdx.x] = (sum>0.f)? sum : numeric_limits<float>::max();
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
		unsigned int *errorListIdx;
		float *transformationMatrices;

		float *output_minimumErrorTransformationMatrices;

//		unsigned int gxy;
		unsigned int n_rsac;
		unsigned int n_views;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_error[dx];
			__shared__ unsigned int shm_idx[dx];

	    	unsigned int tid = threadIdx.x;
	    	unsigned int i = tid;

	    	float min = numeric_limits<float>::max()/2;
	    	unsigned int minIdx = 0;

//	    	if(tid==0)
//	    	{
//	    		printf("max float: %f \n",min);
//	    		for(int i=0;i<n_rsac;i++)
//	    		{
//					float tmp = errorList[blockIdx.x*n_rsac+i];
//					printf(" (%d/%f) ",i,tmp);
//	    		}
//	    		printf("\n");
//	    	}



	    	while(i<n_rsac)
	    	{
	    		float tmp = errorList[blockIdx.x*n_rsac+i];
	    		if(tmp==0.f) printf("%d \n",(blockIdx.x*n_rsac+i));
	    		if(tmp<min)
	    		{
	    			min = tmp;
	    			minIdx = errorListIdx[blockIdx.x*n_rsac+i];
	    		}
	    		if((i+dx) < n_rsac )
	    		{
	    			tmp = errorList[blockIdx.x*n_rsac+i+dx];
	    			if(tmp==0.f) printf("%d \n",(blockIdx.x*n_rsac+i+dx));
	    			if(tmp<min)
	    			{
	    				min = tmp;
	    				minIdx = errorListIdx[blockIdx.x*n_rsac+i+dx];
	    			}
	    		}

	    		i += dx*2;
	    	}

	    	shm_error[tid] = min;
	    	shm_idx[tid] = minIdx;
	    	__syncthreads();

	    	if(tid==0)
	    	{
	    		for(int i=0;i<dx;i++)
	    			if(shm_error[i]==0.f) printf("(%d/%f) ",i,shm_error[i]);
	    		printf("\n");
	    	}

	    	if(dx>=1024){ if(tid<512) 	{ float tmp = shm_error[tid + 512]; if(tmp<min) { shm_error[tid] = min = tmp; shm_idx[tid] = minIdx = shm_idx[tid + 512]; } } __syncthreads(); }
	    	if(dx>=512)	{ if(tid<256) 	{ float tmp = shm_error[tid + 256]; if(tmp<min) { shm_error[tid] = min = tmp; shm_idx[tid] = minIdx = shm_idx[tid + 256]; } } __syncthreads(); }
	    	if(dx>=256)	{ if(tid<128) 	{ float tmp = shm_error[tid + 128]; if(tmp<min) { shm_error[tid] = min = tmp; shm_idx[tid] = minIdx = shm_idx[tid + 128]; } } __syncthreads(); }
	    	if(dx>=128)	{ if(tid<64) 	{ float tmp = shm_error[tid + 64]; 	if(tmp<min) { shm_error[tid] = min = tmp; shm_idx[tid] = minIdx = shm_idx[tid + 64];  } } __syncthreads(); }


	    	if(tid<32)
	    	{
	    		volatile float *smem = shm_error;
	    		volatile unsigned int *smemIdx = shm_idx;

	    		if(dx>=64)	{ float tmp = smem[tid + 32]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 32]; } }
	    		if(dx>=32)	{ float tmp = smem[tid + 16]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 16]; } }
	    		if(dx>=16)	{ float tmp = smem[tid + 8]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 8]; 	} }
	    		if(dx>=8)	{ float tmp = smem[tid + 4]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 4]; 	} }
	    		if(dx>=4)	{ float tmp = smem[tid + 2]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 2]; 	} }
	    		if(dx>=2)	{ float tmp = smem[tid + 1]; 	if(tmp<min) { smem[tid] = min = tmp; smemIdx[tid] = minIdx = smemIdx[tid + 1]; 	} }

	    		if(tid==0)
	    		{
	    			printf("min_kernel: %f \n",min);
	    			if(min==0.f) printf("!!!\n");
	    		}

	    	}
	    	__syncthreads();
	    	if(tid<12)
	    	{
	    		output_minimumErrorTransformationMatrices[(n_views*(n_views-1)/2)*tid + blockIdx.x] = transformationMatrices[(blockIdx.x*12+tid)*n_rsac+shm_idx[0]];
	    	}
//
    		if(tid==0)
    		{
//	    			printf("min_kernel: %f \n",min);
    			output_minimumErrorTransformationMatrices[(n_views*(n_views-1)/2)*12 + blockIdx.x] = min;
    			errorListIdx[0] = minIdx;
    		}


		}
	};
	__global__ void estimateMinimumErrorTransformation(ErrorListMinimumPicker<256> elmp){ elmp(); }
}

device::TransforamtionErrorEstimator transformationErrorestimator;
device::ErrorListSumCalculator<256> errorListSumcalculator256;
device::ErrorListMinimumPicker<256> errorListMinimumPicker256;

void
TranformationValidator::init()
{
	transformationErrorestimator.coords = (float4 *)getInputDataPointer(WorldCoords);
	transformationErrorestimator.normals = (float4 *)getInputDataPointer(Normals);

	//TODO: set to constant memory??
	transformationErrorestimator.sinfo = (SensorInfo *)getInputDataPointer(SensorInfoList);

	transformationErrorestimator.transformation_matrices = (float *)getInputDataPointer(TransforamtionMatrices);
	transformationErrorestimator.transformationMetaData = (int *)getInputDataPointer(TransformationInfoList);

	transformationErrorestimator.output_errorTable = (float *)getTargetDataPointer(ErrorTable);
	transformationErrorestimator.output_errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);
	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 300;
	transformationErrorestimator.angle_threshold = 0.5f;


	errorListSumcalculator256.input_errorTable = (float *)getTargetDataPointer(ErrorTable);
	errorListSumcalculator256.output_errorList = (float *)getTargetDataPointer(ErrorList);
	errorListSumcalculator256.listMetaData = (int *)getInputDataPointer(TransformationInfoList);
	errorListSumcalculator256.gxy = grid.x * grid.y;
	errorListSumcalculator256.n_rsac = n_rsac;


	errorListMinimumPicker256.errorList = (float *)getTargetDataPointer(ErrorList);
	errorListMinimumPicker256.errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);

	errorListMinimumPicker256.transformationMatrices = (float *)getInputDataPointer(TransforamtionMatrices);
	errorListMinimumPicker256.output_minimumErrorTransformationMatrices = (float *)getTargetDataPointer(MinimumErrorTransformationMatrices);
//	errorListMinimumPicker256.gxy = grid.x * grid.y;
	errorListMinimumPicker256.n_rsac = n_rsac;
	errorListMinimumPicker256.n_views = n_views;


}

void TranformationValidator::execute()
{

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



//	int load = 10;
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

//	int load = 10;
//	float *h_tmp2 = (float *)malloc(load*sizeof(float));
//	checkCudaErrors( cudaMemcpy(h_tmp2,transformationErrorestimator.output_errorList,load*sizeof(float),cudaMemcpyDeviceToHost));
//
//	for(int j=0;j<load;j++)
//	{
//		printf("(%d/%f) | ",j,h_tmp2[j]);
//	}
//	printf("\n");


//	int load3 = 5;
//	float *h_tmp3 = (float *)malloc(load3*sizeof(float));
//
//	h_tmp3[0] = 5.f;
//	h_tmp3[1] = 4.f;
//	h_tmp3[2] = 3.f;
//	h_tmp3[3] = 2.6f;
//	h_tmp3[4] = 2.3f;
//	checkCudaErrors(cudaMemcpy(errorListMinimumPicker256.errorList,h_tmp3,load3*sizeof(float),cudaMemcpyHostToDevice));



	device::estimateMinimumErrorTransformation<<<1,errorListMinimumPicker256.dx>>>(errorListMinimumPicker256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int load4 = 13;
	float *h_tmp4 = (float *)malloc(load4*sizeof(float));
	checkCudaErrors( cudaMemcpy(h_tmp4,errorListMinimumPicker256.output_minimumErrorTransformationMatrices,load4*sizeof(float),cudaMemcpyDeviceToHost));

	printf("final transformation: \n");
	for(int i=0;i<12;i++)
	{
		printf("%f | ",h_tmp4[i]);
	}
	printf("\n");
	printf("min_error: %f \n",h_tmp4[12]);





}

TranformationValidator::TranformationValidator(unsigned int n_views,unsigned int n_rsac)
: n_views(n_views), n_rsac(n_rsac)
{

	block = dim3(transformationErrorestimator.dx,transformationErrorestimator.dy);
	grid = dim3(640/block.x,480/block.y);

	DeviceDataParams errorTableParams;
	errorTableParams.elements = ((n_views-1)*n_views)/2 * n_rsac * grid.x * grid.y;
	errorTableParams.element_size = sizeof(float);
	errorTableParams.elementType = FLOAT1;
	errorTableParams.dataType = Point;
	addTargetData(addDeviceDataRequest(errorTableParams),ErrorTable);

//	printf("elements: %d \gxy",errorListParams.elements);

	DeviceDataParams errorListParams;
	errorListParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
	errorListParams.element_size = sizeof(float);
	errorListParams.elementType = FLOAT1;
	errorListParams.dataType = Point;
	addTargetData(addDeviceDataRequest(errorListParams),ErrorList);


	DeviceDataParams errorListIdxParams;
	errorListIdxParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
	errorListIdxParams.element_size = sizeof(unsigned int);
	errorListIdxParams.elementType = UINT1;
	errorListIdxParams.dataType = Indice;
	addTargetData(addDeviceDataRequest(errorListIdxParams),ErrorListIndices);

	unsigned int length = (MAXVIEWS*(MAXVIEWS-1))/2;
	unsigned int h_idx2d[length];
	unsigned int h_idx2m[length];

	{
		unsigned i = 0;
		for(int iy=0;iy<n_views;iy++)
		{
			for(int ix = iy+1; ix<n_views;ix++)
			{
				h_idx2d[i]=ix;
				h_idx2m[i]=iy;
				i++;
			}
		}
	}

	cudaMemcpyToSymbol(device::constant::idx2d,&h_idx2d,length*sizeof(unsigned int));
	cudaMemcpyToSymbol(device::constant::idx2m,&h_idx2m,length*sizeof(unsigned int));

	DeviceDataParams minErrorTransformationMatricesParams;
	minErrorTransformationMatricesParams.elements = ((n_views-1)*n_views)/2;
	minErrorTransformationMatricesParams.element_size = 13 * sizeof(float);
	minErrorTransformationMatricesParams.elementType = FLOAT1;
	minErrorTransformationMatricesParams.dataType = Matrix;
	addTargetData(addDeviceDataRequest(minErrorTransformationMatricesParams),MinimumErrorTransformationMatrices);



//	for(int i=0;i<(n_views*(n_views-1))/2;i++)
//		printf("(%d/%d) | ",h_idx2y[i],h_idx2x[i]);
//
//	printf("\n");
}

void
TranformationValidator::TestMinimumPicker()
{
	printf("TestMinimumPicker! \n");
//	checkCudaErrors(cudaMalloc((void **)&errorListMinimumPicker256.errorList,n_rsac*sizeof(float)));

	thrust::default_random_engine rng(1234L);
	thrust::uniform_real_distribution<float> dist(0, 64);

	thrust::host_vector<float> h_data(n_rsac);
	thrust::host_vector<unsigned int> h_idx(n_rsac);
	for(size_t i = 0; i < h_data.size(); i++)
	{
		h_data[i] = dist(rng);
		h_idx[i] = i;
		printf("(%d/%f) | ",h_idx[i],h_data[i]);
	}
	printf("\n");

	thrust::device_vector<float> d_data = h_data;
	thrust::device_vector<unsigned int> d_idx = h_idx;
	errorListMinimumPicker256.errorList = thrust::raw_pointer_cast(d_data.data());
	errorListMinimumPicker256.errorListIdx = thrust::raw_pointer_cast(d_idx.data());

	errorListMinimumPicker256.n_rsac = n_rsac;
	errorListMinimumPicker256.n_views = n_views;

	thrust::device_vector<float> d_output(13);
	errorListMinimumPicker256.output_minimumErrorTransformationMatrices = thrust::raw_pointer_cast(d_output.data());

	device::estimateMinimumErrorTransformation<<<1,errorListMinimumPicker256.dx>>>(errorListMinimumPicker256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int load4 = 13;
	float *h_tmp4 = (float *)malloc(load4*sizeof(float));
	checkCudaErrors( cudaMemcpy(h_tmp4,errorListMinimumPicker256.output_minimumErrorTransformationMatrices,load4*sizeof(float),cudaMemcpyDeviceToHost));

	printf("min_error: %f \n",h_tmp4[12]);

	h_idx = d_idx;
	printf("minIdx: %d \n",h_idx[0]);

}

void
TranformationValidator::TestSumCalculator()
{
	printf("ErrorSumCalculator! \n");
	printf("gxy: %d \n",grid.x*grid.y);
	thrust::host_vector<float> h_errorTable(n_rsac*grid.x*grid.y);

	for(size_t i = 0; i < n_rsac; i++)
	{
		for(size_t j = 0; j < grid.x*grid.y; j++)
		{
			h_errorTable[i*grid.x*grid.y+j] = ((float)i*2)/((float)(grid.x*grid.y));
		}
	}
	printf("%f \n",h_errorTable[5*n_rsac+5]);

	thrust::host_vector<int> h_metaData(1);
	h_metaData[0] = 80;

	thrust::device_vector<float> d_errorTable = h_errorTable;
	thrust::device_vector<float> d_errorList(n_rsac);
	thrust::device_vector<int> d_metaData = h_metaData;

	errorListSumcalculator256.input_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	errorListSumcalculator256.output_errorList = thrust::raw_pointer_cast(d_errorList.data());
	errorListSumcalculator256.listMetaData = thrust::raw_pointer_cast(d_metaData.data());

	errorListSumcalculator256.gxy = grid.x * grid.y;
	errorListSumcalculator256.n_rsac = n_rsac;


	device::computeErrorListSum<<<n_rsac,errorListSumcalculator256.dx>>>(errorListSumcalculator256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float> h_errorList = d_errorList;
	for(size_t i = 0; i < h_errorList.size(); i++)
	{
		printf("(%d/%f) | ",i,h_errorList[i]);
	}
}

void
TranformationValidator::TestTransform()
{
	thrust::device_vector<float4> d_coordStub(640*480*n_views);
	thrust::device_vector<float4> d_normalStub(640*480*n_views);

	transformationErrorestimator.coords = thrust::raw_pointer_cast(d_coordStub.data());
	transformationErrorestimator.normals = thrust::raw_pointer_cast(d_normalStub.data());

	printf("n_rsac: %d \n",n_rsac);

	thrust::device_vector<float> d_transformation_matricesStubs(n_rsac*12);
	thrust::device_vector<int> d_metaData(n_rsac+1);
	d_metaData[0]=n_rsac;

	transformationErrorestimator.transformation_matrices = thrust::raw_pointer_cast(d_transformation_matricesStubs.data());
	transformationErrorestimator.transformationMetaData = thrust::raw_pointer_cast(d_metaData.data());


	thrust::device_vector<float> d_errorTable(n_rsac*grid.x*grid.y);
	thrust::device_vector<unsigned int> d_errorListIdx(n_rsac);
	transformationErrorestimator.output_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	transformationErrorestimator.output_errorListIdx = thrust::raw_pointer_cast(d_errorListIdx.data());

	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 300;
	transformationErrorestimator.dist_threshold = 1.5f;


	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float> h_errorTable = d_errorTable;
	for(size_t i = 0; i < n_rsac; i++)
	{
		printf("(%d/%f) | ",(int)i,h_errorTable[i*grid.x*grid.y+17]);
	}


/*
	//TODO: set to constant memory??
	transformationErrorestimator.sinfo = (SensorInfo *)getInputDataPointer(SensorInfoList);

	transformationErrorestimator.transformation_matrices = (float *)getInputDataPointer(TransforamtionMatrices);
	transformationErrorestimator.transformationMetaData = (int *)getInputDataPointer(TransformationInfoList);

	transformationErrorestimator.output_errorTable = (float *)getTargetDataPointer(ErrorTable);
	transformationErrorestimator.output_errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);

	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 300;
	transformationErrorestimator.dist_threshold = 0.5f;
	*/
}

void
TranformationValidator::TestTransformError(thrust::host_vector<float4> v0,thrust::host_vector<float4> v1, thrust::host_vector<float4> n0, thrust::host_vector<float4> n1, thrust::host_vector<float> transform)
{
	thrust::host_vector<float4> h_views(v0.size()*2);
	thrust::host_vector<float4> h_normals(v0.size()*2);
	for(int i=0;i<v0.size();i++)
	{
		h_views[i] = v1[i];
		h_views[v0.size()+i] = v0[i];

		h_normals[i] = n1[i];
		h_normals[v0.size()+i] = n0[i];
	}

	thrust::device_vector<float4> d_coords = h_views;
	thrust::device_vector<float4> d_normals = h_normals;

	transformationErrorestimator.coords = thrust::raw_pointer_cast(d_coords.data());
	transformationErrorestimator.normals = thrust::raw_pointer_cast(d_normals.data());

	printf("n_rsac: %d \n",n_rsac);

	thrust::host_vector<int> h_metaData(n_rsac+1);
	h_metaData[0]=n_rsac;
	for(int i=1;i<n_rsac+1;i++)
	{
		h_metaData[i]=0;
	}

	thrust::host_vector<float> h_transformation_matrices(n_rsac*12);
	for(int i=0;i<12;i++)
		h_transformation_matrices[i*n_rsac] = transform[i];

	thrust::device_vector<float> d_transformation_matrices = h_transformation_matrices;

	thrust::device_vector<int> d_metaData = h_metaData;


	transformationErrorestimator.transformation_matrices = thrust::raw_pointer_cast(d_transformation_matrices.data());
	transformationErrorestimator.transformationMetaData = thrust::raw_pointer_cast(d_metaData.data());

	thrust::device_vector<float> d_errorTable(n_rsac*grid.x*grid.y);
	thrust::device_vector<unsigned int> d_errorListIdx(n_rsac);
	transformationErrorestimator.output_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	transformationErrorestimator.output_errorListIdx = thrust::raw_pointer_cast(d_errorListIdx.data());

	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 50;
	transformationErrorestimator.angle_threshold = 0.5f;


	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::host_vector<float> h_errorTable = d_errorTable;
	printf("errortable: ");
	for(int i=0;i<grid.x*grid.y;i++)
	{
		printf("(%d/%f) | ",i,h_errorTable[i]);
	}
	printf(" \n");

//	thrust::device_vector<float> d_errorTable = h_errorTable;
	thrust::device_vector<float> d_errorList(n_rsac);
//	thrust::device_vector<int> d_metaData = h_metaData;


	errorListSumcalculator256.input_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	errorListSumcalculator256.output_errorList = thrust::raw_pointer_cast(d_errorList.data());
	errorListSumcalculator256.listMetaData = thrust::raw_pointer_cast(d_metaData.data());

	errorListSumcalculator256.gxy = grid.x * grid.y;
	errorListSumcalculator256.n_rsac = n_rsac;


	device::computeErrorListSum<<<n_rsac,errorListSumcalculator256.dx>>>(errorListSumcalculator256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float> h_errorList = d_errorList;
	for(int i=0;i<h_errorList.size();i++)
	{
		printf("%f | ",h_errorList[i]);
	}


	/*

	transformationErrorestimator.output_errorTable = (float *)getTargetDataPointer(ErrorTable);
	transformationErrorestimator.output_errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);
	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 300;
	transformationErrorestimator.dist_threshold = 0.5f;
	*/
}

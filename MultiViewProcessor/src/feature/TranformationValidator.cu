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
//#include <thrust/sort.h>
#include <thrust/random.h>

#include "point_info.hpp"
#include "utils.hpp"
#include "../debug/EigenCheckClass.h"
#include "../include/processor.h"

#include "TranformationValidator.h"
#include "../sink/pcd_io.h"

#include <time.h>

#ifndef MAXVIEWS
#define MAXVIEWS 8
#endif

namespace device
{
//	namespace constant
//	{
//		__constant__ unsigned int idx2d[(MAXVIEWS*MAXVIEWS-1)/2];
//		__constant__ unsigned int idx2m[(MAXVIEWS*MAXVIEWS-1)/2];
//	}

	struct TransformationBaseKernel : public FeatureBaseKernel
	{
		enum
		{

		};
	};

	struct TransforamtionErrorEstimator : public TransformationBaseKernel
	{
		enum
		{
			dx = 32,
			dy = 32,

			dbx = 10,
			dby = 10,
			dbtid = 0,

			tx = 32,

			FloatingPointAdjustment = 1,

			ForgroundOnly = 1,
			BlaisLevine = 0,
		};

		float4 *coords;
		float4 *normals;
//		SensorInfo *sinfo;
		SensorInfoV2 *sinfo;

		unsigned int n_rsac;
		unsigned int n_view;

		float *transformation_matrices;
//		int *transformationMetaData;

		float *output_errorTable;
		unsigned int *output_errorListIdx;
		unsigned int *output_validPointsTable;

		float dist_threshold;
		float angle_threshold;


		__device__ __forceinline__ void
		operator () () const
		{
//			__shared__ int shm_transform_idx[WARP_SIZE];
			__shared__ float shm_transformMatrix[WARP_SIZE*TMatrixDim];

			__shared__ float shm_error_buffer[dx*dy];
			__shared__ unsigned int shm_valid_points_buffer[dx*dy];

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			unsigned int tid = threadIdx.y*blockDim.x+threadIdx.x;

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

			unsigned int sx = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int sy = blockIdx.y*blockDim.y+threadIdx.y;

			__syncthreads();
			float4 pos = coords[view_src*640*480+sy*640+sx];
			float4 normal = normals[view_src*640*480+sy*640+sx];


			for(int soff=0;soff<n_rsac;soff+=32)
			{

//				if(tid<32 && (tid+soff<n_rsac) )
//				{
//					shm_transform_idx[tid] = transformationMetaData[blockIdx.z*n_rsac+tid+soff];
//				}
//				__syncthreads();

				if(blockIdx.x==0 && blockIdx.y==0 && tid < 32)
				{
					output_errorListIdx[blockIdx.z*n_rsac+soff+tid] = soff+tid;
				}
				__syncthreads();

				for(int t = tid;t<tx*TMatrixDim;t+=blockDim.x*blockDim.y)
				{
					unsigned int trans_id = t/tx;
					unsigned int trans_tid = t - trans_id*tx;

					shm_transformMatrix[trans_id*tx+trans_tid] = transformation_matrices[blockIdx.z*TMatrixDim*n_rsac+trans_id*n_rsac+soff+trans_tid];//transformation_matrices[(blockIdx.z*12+wid)*n_rsac+idx];

				}
				__syncthreads();

//				if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && tid==0 && soff==0)
//				{
//					for(int i=0;i<WARP_SIZE;i++)
//					{
//						printf("IK: %d -> ",soff+i);
//						for(int d=0;d<TMatrixDim;d++)
//						{
//							printf("%f | ",shm_transformMatrix[d*WARP_SIZE+i]);
//						}
//						printf("\n");
//					}
//				}



				for(int i=0; (i<tx) && (soff+i < n_rsac) ;i++)
				{
//					__syncthreads();

					double error = 0.f;
					bool validPoint = false;


					float3 pos_m,n_m;
					pos_m.x = shm_transformMatrix[0*32+i]*pos.x + shm_transformMatrix[1*32+i]*pos.y + shm_transformMatrix[2*32+i]*pos.z + shm_transformMatrix[9*32+i];
					pos_m.y = shm_transformMatrix[3*32+i]*pos.x + shm_transformMatrix[4*32+i]*pos.y + shm_transformMatrix[5*32+i]*pos.z + shm_transformMatrix[10*32+i];
					pos_m.z = shm_transformMatrix[6*32+i]*pos.x + shm_transformMatrix[7*32+i]*pos.y + shm_transformMatrix[8*32+i]*pos.z + shm_transformMatrix[11*32+i];

					if(pos.z != 0 && pos_m.z != 0 && !isReconstructed(pos.w) && (!ForgroundOnly || isForeground(pos.w)) )
					{

						n_m.x = shm_transformMatrix[0*32+i]*normal.x + shm_transformMatrix[1*32+i]*normal.y+ shm_transformMatrix[2*32+i]*normal.z;
						n_m.y = shm_transformMatrix[3*32+i]*normal.x + shm_transformMatrix[4*32+i]*normal.y+ shm_transformMatrix[5*32+i]*normal.z;
						n_m.z = shm_transformMatrix[6*32+i]*normal.x + shm_transformMatrix[7*32+i]*normal.y+ shm_transformMatrix[8*32+i]*normal.z;

//
//						int ix = (int) (((pos_m.x*120.f)/(0.2084f*pos_m.z))+320);
//						int iy = (int) (((pos_m.y*120.f)/(0.2084f*pos_m.z))+240);


						int ix = (int) ((pos_m.x*sinfo[0].fx)/pos_m.z + sinfo[0].cx);
						int iy = (int) ((pos_m.y*sinfo[0].fy)/pos_m.z + sinfo[0].cy);

	//						if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
	//							printf("%d %d -> %d %d | %f \n",cx,cy,cx,cy);
	//

//						if(blockIdx.z==2 && blockIdx.x==dbx && blockIdx.y==dby)
//							printf("(%d) -> %f %f %f -> %d / %d \n",view_src*640*480+iy*640+ix,pos_m.x,pos_m.y,pos_m.z,ix,iy);

						if(ix>=0 && ix<640 && iy>=0 && iy<480)
						{

							float4 pos_d = coords[view_target*640*480+iy*640+ix];
							float4 n_d = normals[view_target*640*480+iy*640+ix];
//
//							if(blockIdx.z==2 && blockIdx.x==dbx && blockIdx.y==dby)
//								printf("%d:(%d) -> %f %f %f -> %d / %d \n",view_target,view_target*640*480+iy*640+ix,pos_d.x,pos_d.y,pos_d.z,ix,iy);

							if( pos_d.z != 0 && !isReconstructed(pos_d.w) && (!ForgroundOnly || isForeground(pos_d.w)) )
							{
//								if(blockIdx.z==2 && blockIdx.x==dbx && blockIdx.y==dby)
//									printf("in!! \n");

	//								continue;

								float3 dv = make_float3(pos_m.x-pos_d.x,pos_m.y-pos_d.y,pos_m.z-pos_d.z);
								float angle = n_m.x * n_d.x + n_m.y * n_d.y + n_m.z * n_d.z;

	//							if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
	//							{
	//								printf("pos_m: %f %f %f (%d/%d)\n",pos_m.x,pos_m.y,pos_m.z,sx,sy);
	//								printf("pos_d: %f %f %f (%d/%d)\n",pos_d.x,pos_d.y,pos_d.z,cx,cy);
	//								printf("dv_length: %f dist_tresh: %f \n",norm(dv),dist_threshold);
	//							}

	//							shm_error_buffer[tid] = dist_threshold;
	//							shm_valid_points_buffer[tid] = 1;

								if(BlaisLevine)
									validPoint = true;

								if( norm(dv) < dist_threshold && ( angle >= angle_threshold) )
								{
									validPoint = true;

									error = dv.x*n_d.x + dv.y*n_d.y + dv.z*n_d.z;
									error *= error;

									if(BlaisLevine)
									{
										if(error>dist_threshold)
											error = dist_threshold;

									}
//
//									if(error<dist_threshold)
	//									shm_error_buffer[tid] = error;

	//								if(blockIdx.x==dbx && blockIdx.y==dby && tid == dbtid && i==0)
	//								{
	//									printf("dv: %f %f %f \n",dv.x,dv.y,dv.z);
	//									printf("nd: %f %f %f \n",n_d.x,n_d.y,n_d.z);
	//									printf("error: %f error^2: %f \n",error,error*error);
	//								}
	//								shm_error_buffer[tid] = 0;// * error;
								}
							}
						}

					}
					__syncthreads();

					shm_error_buffer[tid] = error / ((float)FloatingPointAdjustment);
					shm_valid_points_buffer[tid] = validPoint;
					__syncthreads();

//					if(blockIdx.x==dbx && blockIdx.y==dby && i==0 && tid==dbtid)
//					{
//						printf("errorList inBlock: ");
//						for(int io=0;io<dx*dy;io++)
//						{
//							printf("(%d/%f) | ",io,shm_error_buffer[io]);
//						}
//						printf("\n");
//					}

//					if(blockIdx.x==dbx && blockIdx.y==dby && i==0 && tid==dbtid)
//					{
//						printf("validPoints inBlock: ");
//						for(int io=0;io<dx*dy;io++)
//						{
//							printf("(%d/%d) | ",io,shm_valid_points_buffer[io]);
//						}
//						printf("\n");
//					}

					__syncthreads();
//					float sum = 0.f;
//					unsigned int sum_points = 0;
					if(dx*dy>=1024) { if(tid < 512) { 	shm_error_buffer[tid] += shm_error_buffer[tid+512];
														shm_valid_points_buffer[tid] += shm_valid_points_buffer[tid + 512];
													} 	__syncthreads(); }

					if(dx*dy>=512) 	{ if(tid < 256) { 	shm_error_buffer[tid] += shm_error_buffer[tid+256];
														shm_valid_points_buffer[tid] += shm_valid_points_buffer[tid + 256];
													} 	__syncthreads(); }

					if(dx*dy>=256) 	{ if(tid < 128) { 	shm_error_buffer[tid] += shm_error_buffer[tid+128];
														shm_valid_points_buffer[tid] += shm_valid_points_buffer[tid + 128];
													} 	__syncthreads(); }

					if(dx*dy>=128) 	{ if(tid < 64) 	{ 	shm_error_buffer[tid] += shm_error_buffer[tid+64];
														shm_valid_points_buffer[tid] += shm_valid_points_buffer[tid + 64];
													} 	__syncthreads(); }


					if(tid < 32)
					{
						volatile float *smem = shm_error_buffer;
						volatile unsigned int *smem_points = shm_valid_points_buffer;
						if(dx*dy>=64) 	{ 	smem[tid] += smem[tid+32];
											smem_points[tid] +=smem_points[tid+32];
						}
						if(dx*dy>=32) 	{ 	smem[tid] += smem[tid+16];
											smem_points[tid] += smem_points[tid+16];
						}
						if(dx*dy>=16) 	{ 	smem[tid] += smem[tid+8];
											smem_points[tid] += smem_points[tid+8];
						}
						if(dx*dy>=8) 	{ 	smem[tid] += smem[tid+4];
											smem_points[tid] +=smem_points[tid+4];
						}
						if(dx*dy>=4) 	{ 	smem[tid] += smem[tid+2];
											smem_points[tid] += smem_points[tid+2];
						}
						if(dx*dy>=2) 	{ 	smem[tid] += smem[tid+1];
											smem_points[tid] += smem_points[tid+1];
						}

						if(tid==0)
						{
							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = shm_error_buffer[0];
							output_validPointsTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = shm_valid_points_buffer[0];

//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = blockIdx.y*gridDim.x+blockIdx.x;//shm_error_buffer[0];
//							output_validPointsTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = soff+i;//shm_valid_points_buffer[0];



//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = 1;//shm_error_buffer[0];
//							output_validPointsTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = 1;//shm_valid_points_buffer[0];

//							output_errorList[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = blockIdx.y*gridDim.x+blockIdx.x;
//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = i/(gridDim.x*gridDim.y);
//							output_errorList[(i)*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x] = smem[0];
//							output_errorTable[(blockIdx.z*n_rsac+soff+i)*gridDim.y*gridDim.x + blockIdx.y*gridDim.x+blockIdx.x] = soff + i;
						}

//						if(blockIdx.z==2 && blockIdx.x==dbx && blockIdx.y==dby && i==0 && tid==dbtid && threadIdx.x==0 && threadIdx.y==0)
//							printf("(%d) block_sum: %f block_point_sum: %d \n",i,shm_error_buffer[0],shm_valid_points_buffer[0]);

					}
					__syncthreads();

				}

				__syncthreads();

			}

		}

	};
	__global__ void computeTransformationError(TransforamtionErrorEstimator tee) { tee(); }


	template<unsigned int blockSize>
	struct ErrorListSumCalculator
	{
		enum
		{
			dx = blockSize,
			Relative = 1,
		};

		float *input_errorTable;
		unsigned int *input_validPoints;

		float *output_errorList;
		int *listMetaData;

		unsigned int min_overlap;

		unsigned int gxy;
		unsigned int n_rsac;

	    __device__ __forceinline__ void
		operator () () const
		{
	    	__shared__ float shm_error[dx];
	    	__shared__ unsigned int shm_points[dx];
//	    	__shared__ unsigned int shm_length;


	    	unsigned int tid = threadIdx.x;
	    	unsigned int i = tid;

//	    	if(tid==0)
//	    	{
//	    		shm_length = listMetaData[0];
////	    		printf("sumKernel:length: %d \n",shm_length);
//	    	}

//	    	__syncthreads();
//	    	if(blockIdx.x>n_rsac)
//	    	{
//	    		output_errorList[(blockIdx.y*n_rsac+blockIdx.x)] = numeric_limits<float>::max();
//				return;
//	    	}

	    	float sum = 0.f;
	    	unsigned int sum_points = 0;
	    	while(i<gxy)
	    	{
	    		sum += input_errorTable[(blockIdx.z*n_rsac+blockIdx.x)*gxy+i];
	    		sum_points += input_validPoints[(blockIdx.z*n_rsac+blockIdx.x)*gxy+i];

	    		if( (i+dx) < gxy )
	    		{
	    			sum += input_errorTable[(blockIdx.z*n_rsac+blockIdx.x)*gxy+i+dx];
	    			sum_points += input_validPoints[(blockIdx.z*n_rsac+blockIdx.x)*gxy+i+dx];
	    		}

	    		i += dx*2;
	    	}

	    	shm_error[tid] = sum;
	    	shm_points[tid] = sum_points;
	    	__syncthreads();


	    	if(dx>=1024){ if(tid<512) 	{	shm_error[tid] = sum = sum + shm_error[tid + 512];
	    									shm_points[tid] = sum_points = sum_points + shm_points[tid + 512];
										} 	__syncthreads(); }

	    	if(dx>=512)	{ if(tid<256) 	{	shm_error[tid] = sum = sum + shm_error[tid + 256];
											shm_points[tid] = sum_points = sum_points + shm_points[tid + 256];
										} 	__syncthreads(); }

	    	if(dx>=256)	{ if(tid<128) 	{	shm_error[tid] = sum = sum + shm_error[tid + 128];
											shm_points[tid] = sum_points = sum_points + shm_points[tid + 128];
										} 	__syncthreads(); }

	    	if(dx>=128)	{ if(tid<64) 	{	shm_error[tid] = sum = sum + shm_error[tid + 64];
											shm_points[tid] = sum_points = sum_points + shm_points[tid + 64];
										} 	__syncthreads(); }

	    	if(tid<32)
	    	{
	    		volatile float *smem = shm_error;
	    		volatile unsigned int *smem_points = shm_points;

	    		if(dx>=64) 	{ 	smem[tid] = sum = sum + smem[tid + 32];
	    						smem_points[tid] = sum_points = sum_points + smem_points[tid + 32]; }

	    		if(dx>=32) 	{ 	smem[tid] = sum = sum + smem[tid + 16];
								smem_points[tid] = sum_points = sum_points + smem_points[tid + 16]; }

	    		if(dx>=16) 	{ 	smem[tid] = sum = sum + smem[tid + 8];
								smem_points[tid] = sum_points = sum_points + smem_points[tid + 8]; }

	    		if(dx>=8) 	{ 	smem[tid] = sum = sum + smem[tid + 4];
								smem_points[tid] = sum_points = sum_points + smem_points[tid + 4]; }

	    		if(dx>=4)	{ 	smem[tid] = sum = sum + smem[tid + 2];
								smem_points[tid] = sum_points = sum_points + smem_points[tid + 2]; }

	    		if(dx>=2) 	{ 	smem[tid] = sum = sum + smem[tid + 1];
								smem_points[tid] = sum_points = sum_points + smem_points[tid + 1]; }

	    		if(tid==0)
	    		{
	    			output_errorList[(blockIdx.z*n_rsac)+blockIdx.x] = (sum>0.f && sum_points >= min_overlap)? sum/((Relative)?sum_points:1.f) : numeric_limits<float>::max();
	    		}

//	    		if(blockIdx.x == 2 && tid == 0)
//	    			printf("sum_error: %f | sum_validPoints: %d \n",sum,sum_points);
	    	}


		}
	};
	__global__ void computeErrorListSum(ErrorListSumCalculator<256> elsc) { elsc(); }


	template<unsigned int bx>
	struct ErrorListMinimumPicker : public TransformationBaseKernel
	{
		enum
		{
			dx = bx,
		};

		float *input_errorList;
		unsigned int *input_errorListIdx;
		float *input_transformationMatrices;

		float *output_minimumErrorTransformationMatrices;
		float *output_minimumError;

//		unsigned int gxy;
		unsigned int n_rsac;
//		unsigned int n_views;

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
	    		float tmp = input_errorList[blockIdx.x*n_rsac+i];
//	    		if(tmp==0.f) printf("%d \n",(blockIdx.x*n_rsac+i));
	    		if(tmp<min)
	    		{
	    			min = tmp;
	    			minIdx = input_errorListIdx[blockIdx.x*n_rsac+i];
	    		}
	    		if((i+dx) < n_rsac )
	    		{
	    			tmp = input_errorList[blockIdx.x*n_rsac+i+dx];
//	    			if(tmp==0.f) printf("%d \n",(blockIdx.x*n_rsac+i+dx));
	    			if(tmp<min)
	    			{
	    				min = tmp;
	    				minIdx = input_errorListIdx[blockIdx.x*n_rsac+i+dx];
	    			}
	    		}

	    		i += dx*2;
	    	}

	    	shm_error[tid] = min;
	    	shm_idx[tid] = minIdx;
	    	__syncthreads();

//	    	if(tid==0)
//	    	{
//	    		for(int i=0;i<dx;i++)
//	    			if(shm_error[i]==0.f) printf("(%d/%f) ",i,shm_error[i]);
//	    		printf("\n");
//	    	}

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

//	    		if(tid==0)
//	    		{
//	    			printf("min_kernel: %f \n",min);
//	    			if(min==0.f) printf("!!!\n");
//	    		}

	    	}
	    	__syncthreads();
	    	if(tid<12)
	    	{
	    		output_minimumErrorTransformationMatrices[blockIdx.x*TMatrixDim+tid] = input_transformationMatrices[blockIdx.x*TMatrixDim*n_rsac + tid*n_rsac + shm_idx[0]];
	    	}
//
    		if(tid==0)
    		{
//	    		printf("min_kernel (%d): idx %d -> %f \n",blockIdx.x,minIdx,min);
	    		output_minimumError[blockIdx.x] = min;
//	    		for(int k=0;k<12;k++)
//	    			printf("%f | ",input_transformationMatrices[blockIdx.x*TMatrixDim*n_rsac + k*n_rsac + shm_idx[0]]);
//    			printf("\n");
//	    		output_minimumErrorTransformationMatrices[gridDim.x*12 + blockIdx.x] = min;
//    			input_errorListIdx[0] = minIdx;
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
	transformationErrorestimator.sinfo = (SensorInfoV2 *)getInputDataPointer(SensorInfoListV2);
	transformationErrorestimator.transformation_matrices = (float *)getInputDataPointer(TransforamtionMatrices);

//	transformationErrorestimator.output_errorTable = (float *)getTargetDataPointer(ErrorTable);
//	transformationErrorestimator.output_validPointsTable = (unsigned int *)getTargetDataPointer(ValidPointsTable);
//	transformationErrorestimator.output_errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);



//	errorListSumcalculator256.input_errorTable = (float *)getTargetDataPointer(ErrorTable);
//	errorListSumcalculator256.input_validPoints = (unsigned int *)getTargetDataPointer(ValidPointsTable);
//	errorListSumcalculator256.output_errorList = (float *)getTargetDataPointer(ErrorList);
//	errorListSumcalculator256.listMetaData = (int *)getInputDataPointer(TransformationInfoList);
//	errorListSumcalculator256.gxy = grid.x * grid.y;
//	errorListSumcalculator256.n_rsac = n_rsac;
//	errorListSumcalculator256.min_overlap = (640*480)/20;
//
//
//	errorListMinimumPicker256.errorList = (float *)getTargetDataPointer(ErrorList);
//	errorListMinimumPicker256.errorListIdx = (unsigned int *)getTargetDataPointer(ErrorListIndices);
//
	errorListMinimumPicker256.input_transformationMatrices = (float *)getInputDataPointer(TransforamtionMatrices);
	errorListMinimumPicker256.output_minimumErrorTransformationMatrices = (float *)getTargetDataPointer(MinimumErrorTransformationMatrices);
////	errorListMinimumPicker256.gxy = grid.x * grid.y;
//	errorListMinimumPicker256.n_rsac = n_rsac;
//	errorListMinimumPicker256.n_views = n_views;


}

void TranformationValidator::execute()
{
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int view_combinations = (n_views*(n_views-1))/2;
	if(outputlevel>2)
		printf("TranformationValidatorInfo: n_rsac: %d | view_combinstaions: %d \n",n_rsac,view_combinations);

	dim3 transformErrorBlock(transformationErrorestimator.dx,transformationErrorestimator.dy);
	dim3 transformErrorGrid(640/transformErrorBlock.x,480/transformErrorBlock.y,view_combinations);

	thrust::device_vector<float> d_errorTable( view_combinations * n_rsac * transformErrorGrid.x * transformErrorGrid.y);
	thrust::device_vector<unsigned int> d_validPointsTable(view_combinations * n_rsac * transformErrorGrid.x * transformErrorGrid.y);
	thrust::device_vector<unsigned int> d_errorListIdx( view_combinations * n_rsac);


	transformationErrorestimator.output_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	transformationErrorestimator.output_validPointsTable = thrust::raw_pointer_cast(d_validPointsTable.data());
	transformationErrorestimator.output_errorListIdx = thrust::raw_pointer_cast(d_errorListIdx.data());


	transformationErrorestimator.n_view = n_views;
	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 50.f;
	transformationErrorestimator.angle_threshold = cos(M_PI/4);
	if(outputlevel>2)
		printf("transformErrorBlock:(%d/%d) | transformErrorGrid:(%d/%d/%d) \n",transformErrorBlock.x,transformErrorBlock.y,transformErrorGrid.x,transformErrorGrid.y,transformErrorGrid.z);
	device::computeTransformationError<<<transformErrorGrid,transformErrorBlock>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



//	thrust::host_vector<float> h_errorTable = d_errorTable;
//	thrust::host_vector<unsigned int> h_validPointsTable = d_validPointsTable;
//	for(int r=0;r<5;r++)
//	{
//		for(int j=0;j<transformErrorGrid.x*transformErrorGrid.y;j++)
//		{
//			printf("(%f/%d) | ",h_errorTable[r*transformErrorGrid.x*transformErrorGrid.y+j],h_validPointsTable[r*transformErrorGrid.x*transformErrorGrid.y+j]);
////			printf("(%f) | ",h_errorTable[r*transformErrorGrid.x*transformErrorGrid.y+j]);
//		}
//		printf("\n");
//	}


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


	thrust::device_vector<float> d_errorList(view_combinations*n_rsac);
	errorListSumcalculator256.output_errorList = thrust::raw_pointer_cast(d_errorList.data());

	errorListSumcalculator256.input_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	errorListSumcalculator256.input_validPoints = thrust::raw_pointer_cast(d_validPointsTable.data());
	errorListSumcalculator256.gxy = transformErrorGrid.x*transformErrorGrid.y;
	errorListSumcalculator256.n_rsac = n_rsac;
	errorListSumcalculator256.min_overlap = (unsigned int)(640*480*0.1f);

	dim3 sumErrorListGrid(n_rsac,1,view_combinations);

	device::computeErrorListSum<<<sumErrorListGrid,errorListSumcalculator256.dx>>>(errorListSumcalculator256);
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


	thrust::device_vector<float> d_minTransformError(view_combinations);
	errorListMinimumPicker256.output_minimumError = thrust::raw_pointer_cast(d_minTransformError.data());

	errorListMinimumPicker256.input_errorList = thrust::raw_pointer_cast(d_errorList.data());
	errorListMinimumPicker256.input_errorListIdx = thrust::raw_pointer_cast(d_errorListIdx.data());
	errorListMinimumPicker256.n_rsac = n_rsac;

//
//	dim3 minErrorListPickerGrid(n_rsac,1,view_combinations);

	device::estimateMinimumErrorTransformation<<<view_combinations,errorListMinimumPicker256.dx>>>(errorListMinimumPicker256);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if(outputlevel>1)
	{
		int load4 = 12*view_combinations;
		float *h_tmp4 = (float *)malloc(load4*sizeof(float));
		checkCudaErrors( cudaMemcpy(h_tmp4,errorListMinimumPicker256.output_minimumErrorTransformationMatrices,load4*sizeof(float),cudaMemcpyDeviceToHost));

		thrust::host_vector<float> h_minTransformError = d_minTransformError;


		float maxErr = -1.f;
		int maxV = -1;
		int inValidTransCount = 0;
		float maxSimError = 500.f;

		printf("final transformation: \n");
		for(int v=0;v<view_combinations;v++)
		{
			for(int i=0;i<12;i++)
			{
				printf("%f | ",h_tmp4[v*12+i]);
			}
			printf("minTransformError: %f \n",h_minTransformError[v]);

			if(h_minTransformError[v]>maxErr)
			{
				maxErr = h_minTransformError[v];
				maxV = v;
			}

			if(h_minTransformError[v]>maxSimError)
			{
				inValidTransCount++;
			}
		}
		printf("maxTransV: %d \n",maxV);
		printf("invalidCount: %d \n",inValidTransCount);

		if(n_views == 3 && inValidTransCount<2)
		{
			thrust::device_ptr<float4> pos_ptr = thrust::device_pointer_cast(transformationErrorestimator.coords);
			thrust::device_vector<float4> d_pos(pos_ptr,pos_ptr+n_views*640*480);
			thrust::host_vector<float4> h_pos = d_pos;
			thrust::host_vector<float4> h_pos_tmp = d_pos;
			thrust::host_vector<float> h_tmpTrans(12);

			int i = 0;
			int t = 1;
			switch (maxV) {
				case 0:
					printf("case %d \n",1);
					i = 0;
					t = 1;

					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmp4[t*12+0]*pt.x + h_tmp4[t*12+1]*pt.y + h_tmp4[t*12+2]*pt.z + h_tmp4[t*12+9];
							o.y = h_tmp4[t*12+3]*pt.x + h_tmp4[t*12+4]*pt.y + h_tmp4[t*12+5]*pt.z + h_tmp4[t*12+10];
							o.z = h_tmp4[t*12+6]*pt.x + h_tmp4[t*12+7]*pt.y + h_tmp4[t*12+8]*pt.z + h_tmp4[t*12+11];
						}
						h_pos[i*640*480+p] = o;
					}

					i = 1;
					t = 2;
					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmp4[t*12+0]*pt.x + h_tmp4[t*12+1]*pt.y + h_tmp4[t*12+2]*pt.z + h_tmp4[t*12+9];
							o.y = h_tmp4[t*12+3]*pt.x + h_tmp4[t*12+4]*pt.y + h_tmp4[t*12+5]*pt.z + h_tmp4[t*12+10];
							o.z = h_tmp4[t*12+6]*pt.x + h_tmp4[t*12+7]*pt.y + h_tmp4[t*12+8]*pt.z + h_tmp4[t*12+11];
						}
						h_pos[i*640*480+p] = o;
					}

					i = 2;
					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
							h_pos[i*640*480+p] = pt;
						else
							h_pos[i*640*480+p] = make_float4(0,0,0,0);
					}
					break;

				case 1:
					printf("case %d \n",1);
					i = 0;
					t = 0;

					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmp4[t*12+0]*pt.x + h_tmp4[t*12+1]*pt.y + h_tmp4[t*12+2]*pt.z + h_tmp4[t*12+9];
							o.y = h_tmp4[t*12+3]*pt.x + h_tmp4[t*12+4]*pt.y + h_tmp4[t*12+5]*pt.z + h_tmp4[t*12+10];
							o.z = h_tmp4[t*12+6]*pt.x + h_tmp4[t*12+7]*pt.y + h_tmp4[t*12+8]*pt.z + h_tmp4[t*12+11];
						}
						h_pos[i*640*480+p] = o;
					}



					i = 2;
					t = 2;

					h_tmpTrans[0] = h_tmp4[t*12+0];
					h_tmpTrans[1] = h_tmp4[t*12+3];
					h_tmpTrans[2] = h_tmp4[t*12+6];

					h_tmpTrans[3] = h_tmp4[t*12+1];
					h_tmpTrans[4] = h_tmp4[t*12+4];
					h_tmpTrans[5] = h_tmp4[t*12+7];

					h_tmpTrans[6] = h_tmp4[t*12+2];
					h_tmpTrans[7] = h_tmp4[t*12+5];
					h_tmpTrans[8] = h_tmp4[t*12+8];

					h_tmpTrans[9] = -1.f * (h_tmpTrans[0] * h_tmp4[t*12+9] + h_tmpTrans[1] * h_tmp4[t*12+10] + h_tmpTrans[2] * h_tmp4[t*12+11]);
					h_tmpTrans[10] = -1.f * (h_tmpTrans[3] * h_tmp4[t*12+9] + h_tmpTrans[4] * h_tmp4[t*12+10] + h_tmpTrans[5] * h_tmp4[t*12+11]);
					h_tmpTrans[11] = -1.f * (h_tmpTrans[6] * h_tmp4[t*12+9] + h_tmpTrans[7] * h_tmp4[t*12+10] + h_tmpTrans[8] * h_tmp4[t*12+11]);

					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmpTrans[0]*pt.x + h_tmpTrans[1]*pt.y + h_tmpTrans[2]*pt.z + h_tmpTrans[9];
							o.y = h_tmpTrans[3]*pt.x + h_tmpTrans[4]*pt.y + h_tmpTrans[5]*pt.z + h_tmpTrans[10];
							o.z = h_tmpTrans[6]*pt.x + h_tmpTrans[7]*pt.y + h_tmpTrans[8]*pt.z + h_tmpTrans[11];
						}
						h_pos[i*640*480+p] = o;
					}

					i = 1;
					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
							h_pos[i*640*480+p] = pt;
						else
							h_pos[i*640*480+p] = make_float4(0,0,0,0);
					}

					break;


				case 2:
					printf("case %d \n",1);

					i = 1;
					t = 0;

					h_tmpTrans[0] = h_tmp4[t*12+0];
					h_tmpTrans[1] = h_tmp4[t*12+3];
					h_tmpTrans[2] = h_tmp4[t*12+6];

					h_tmpTrans[3] = h_tmp4[t*12+1];
					h_tmpTrans[4] = h_tmp4[t*12+4];
					h_tmpTrans[5] = h_tmp4[t*12+7];

					h_tmpTrans[6] = h_tmp4[t*12+2];
					h_tmpTrans[7] = h_tmp4[t*12+5];
					h_tmpTrans[8] = h_tmp4[t*12+8];

					h_tmpTrans[9] = -1.f * (h_tmpTrans[0] * h_tmp4[t*12+9] + h_tmpTrans[1] * h_tmp4[t*12+10] + h_tmpTrans[2] * h_tmp4[t*12+11]);
					h_tmpTrans[10] = -1.f * (h_tmpTrans[3] * h_tmp4[t*12+9] + h_tmpTrans[4] * h_tmp4[t*12+10] + h_tmpTrans[5] * h_tmp4[t*12+11]);
					h_tmpTrans[11] = -1.f * (h_tmpTrans[6] * h_tmp4[t*12+9] + h_tmpTrans[7] * h_tmp4[t*12+10] + h_tmpTrans[8] * h_tmp4[t*12+11]);

					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmpTrans[0]*pt.x + h_tmpTrans[1]*pt.y + h_tmpTrans[2]*pt.z + h_tmpTrans[9];
							o.y = h_tmpTrans[3]*pt.x + h_tmpTrans[4]*pt.y + h_tmpTrans[5]*pt.z + h_tmpTrans[10];
							o.z = h_tmpTrans[6]*pt.x + h_tmpTrans[7]*pt.y + h_tmpTrans[8]*pt.z + h_tmpTrans[11];
						}
						h_pos[i*640*480+p] = o;
					}

					i = 2;
					t = 1;

					h_tmpTrans[0] = h_tmp4[t*12+0];
					h_tmpTrans[1] = h_tmp4[t*12+3];
					h_tmpTrans[2] = h_tmp4[t*12+6];

					h_tmpTrans[3] = h_tmp4[t*12+1];
					h_tmpTrans[4] = h_tmp4[t*12+4];
					h_tmpTrans[5] = h_tmp4[t*12+7];

					h_tmpTrans[6] = h_tmp4[t*12+2];
					h_tmpTrans[7] = h_tmp4[t*12+5];
					h_tmpTrans[8] = h_tmp4[t*12+8];

					h_tmpTrans[9] = -1.f * (h_tmpTrans[0] * h_tmp4[t*12+9] + h_tmpTrans[1] * h_tmp4[t*12+10] + h_tmpTrans[2] * h_tmp4[t*12+11]);
					h_tmpTrans[10] = -1.f * (h_tmpTrans[3] * h_tmp4[t*12+9] + h_tmpTrans[4] * h_tmp4[t*12+10] + h_tmpTrans[5] * h_tmp4[t*12+11]);
					h_tmpTrans[11] = -1.f * (h_tmpTrans[6] * h_tmp4[t*12+9] + h_tmpTrans[7] * h_tmp4[t*12+10] + h_tmpTrans[8] * h_tmp4[t*12+11]);

					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						float4 o = make_float4(0,0,0,0);
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
						{
							o.x = h_tmpTrans[0]*pt.x + h_tmpTrans[1]*pt.y + h_tmpTrans[2]*pt.z + h_tmpTrans[9];
							o.y = h_tmpTrans[3]*pt.x + h_tmpTrans[4]*pt.y + h_tmpTrans[5]*pt.z + h_tmpTrans[10];
							o.z = h_tmpTrans[6]*pt.x + h_tmpTrans[7]*pt.y + h_tmpTrans[8]*pt.z + h_tmpTrans[11];
						}
						h_pos[i*640*480+p] = o;
					}

					i = 0;
					for(int p=0;p<640*480;p++)
					{
						float4 pt = h_pos[i*640*480+p];
						if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
							h_pos[i*640*480+p] = pt;
						else
							h_pos[i*640*480+p] = make_float4(0,0,0,0);
					}

					break;

				default:
					printf("Dam shit! \n");
					break;
			}

			char path[100];
			for(int i=0;i<3;i++)
			{
				sprintf(path,"/home/avo/pcds/mt/mintransformed_%d.pcd",i);
				host::io::PCDIOController pcdIOCtrl;
				float4 *f4p = h_pos.data();
				pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
			}


//			for(int i=0;i<3;i++)
//			{
//				for(int p=0;p<640*480;p++)
//				{
//					float4 pt = h_pos_tmp[i*640*480+p];
//					if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
//						h_pos_tmp[i*640*480+p] = pt;
//					else
//						h_pos_tmp[i*640*480+p] = make_float4(0,0,0,0);
//				}
//
//				sprintf(path,"/home/avo/pcds/mt/untransformed_%d.pcd",i);
//				host::io::PCDIOController pcdIOCtrl;
//				float4 *f4p = h_pos_tmp.data();
//				pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
//			}

			Processor::breakRun();
			printf("breakRun! \n");
		}


		/*
		float sum = 0.f;
		int ca = 2;
		if(ca==2 && n_views==3)
		{
			thrust::device_ptr<float4> pos_ptr = thrust::device_pointer_cast(transformationErrorestimator.coords);
			thrust::device_vector<float4> d_pos(pos_ptr,pos_ptr+n_views*640*480);
			thrust::host_vector<float4> h_pos = d_pos;

			int i = 0;
			int t = 1;
			sum += h_minTransformError[1]/100000.f;
			for(int p=0;p<640*480;p++)
			{
				float4 pt = h_pos[i*640*480+p];
				float4 o = make_float4(0,0,0,0);
				if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
				{
					o.x = h_tmp4[t*12+0]*pt.x + h_tmp4[t*12+1]*pt.y + h_tmp4[t*12+2]*pt.z + h_tmp4[t*12+9];
					o.y = h_tmp4[t*12+3]*pt.x + h_tmp4[t*12+4]*pt.y + h_tmp4[t*12+5]*pt.z + h_tmp4[t*12+10];
					o.z = h_tmp4[t*12+6]*pt.x + h_tmp4[t*12+7]*pt.y + h_tmp4[t*12+8]*pt.z + h_tmp4[t*12+11];
				}
				h_pos[i*640*480+p] = o;
			}

			i = 1;
			t = 2;
			sum += h_minTransformError[2]/100000.f;
			for(int p=0;p<640*480;p++)
			{
				float4 pt = h_pos[i*640*480+p];
				float4 o = make_float4(0,0,0,0);
				if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
				{
					o.x = h_tmp4[t*12+0]*pt.x + h_tmp4[t*12+1]*pt.y + h_tmp4[t*12+2]*pt.z + h_tmp4[t*12+9];
					o.y = h_tmp4[t*12+3]*pt.x + h_tmp4[t*12+4]*pt.y + h_tmp4[t*12+5]*pt.z + h_tmp4[t*12+10];
					o.z = h_tmp4[t*12+6]*pt.x + h_tmp4[t*12+7]*pt.y + h_tmp4[t*12+8]*pt.z + h_tmp4[t*12+11];
				}
				h_pos[i*640*480+p] = o;
			}

			i = 2;
			for(int p=0;p<640*480;p++)
			{
				float4 pt = h_pos[i*640*480+p];
				if(device::isForeground(pt.w) && !device::isReconstructed(pt.w))
					h_pos[i*640*480+p] = pt;
				else
					h_pos[i*640*480+p] = make_float4(0,0,0,0);
			}


			char path[100];
			for(int i=0;i<3;i++)
			{
				sprintf(path,"/home/avo/pcds/mt/mintransformed_%d.pcd",i);
				host::io::PCDIOController pcdIOCtrl;
				float4 *f4p = h_pos.data();
				pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
			}

		}
*/



	//	printf("min_error: %f \n",h_tmp4[12]);

//		thrust::device_ptr<float4> pos_ptr = thrust::device_pointer_cast(transformationErrorestimator.coords);
//		thrust::device_vector<float4> d_pos(pos_ptr,pos_ptr+n_views*640*480);
//		thrust::host_vector<float4> h_pos = d_pos;
//
//		int o = 2;
//
//		for(int i=0;i<o;i++)
//		{
//			for(int p=0;p<i*640*480;p++)
//			{
//				float4 t = h_pos[p];
//				float4 o = make_float4(0,0,0,0);
//				if(device::isForeground(t.w) && !device::isReconstructed(t.w))
//				{
//					o.x = h_tmp4[i*12+0]*t.x + h_tmp4[i*12+1]*t.y + h_tmp4[i*12+2]*t.z + h_tmp4[i*12+9];
//					o.y = h_tmp4[i*12+3]*t.x + h_tmp4[i*12+4]*t.y + h_tmp4[i*12+5]*t.z + h_tmp4[i*12+10];
//					o.z = h_tmp4[i*12+6]*t.x + h_tmp4[i*12+7]*t.y + h_tmp4[i*12+8]*t.z + h_tmp4[i*12+11];
//				}
//				h_pos[p] = o;
//			}
//		}
//
//		for(int p=0;p<640*480;p++)
//		{
//			float4 t = h_pos[o*640*480+p];
//			if(device::isForeground(t.w) && !device::isReconstructed(t.w))
//				h_pos[o*640*480+p] = t;
//			else
//				h_pos[o*640*480+p] = make_float4(0,0,0,0);
//		}
//
//
//		time_t rawtime;
//		time ( &rawtime );
//
//		char path[50];
//		for(int i=0;i<o+1;i++)
//		{
//			sprintf(path,"/home/avo/pcds/mintransformed_%d.pcd",i);
//			host::io::PCDIOController pcdIOCtrl;
//			float4 *f4p = h_pos.data();
//			pcdIOCtrl.writeASCIIPCD(path,(float *)(f4p + i*640*480),640*480);
//		}
	}


//	EigenCheckClass::setMinTransformError(minTransformError);
//
//	thrust::device_ptr<float> dptr_finalTransform = thrust::device_pointer_cast(errorListMinimumPicker256.output_minimumErrorTransformationMatrices);
//	thrust::device_vector<float> d_finalTransformations(dptr_finalTransform,dptr_finalTransform+12*view_combinations);
//	thrust::device_vector<float> h_finalTransformations = d_finalTransformations;
//
//	EigenCheckClass::checkTransformationQuality(h_finalTransformations);
//
////	float *data = h_finalTransformations.data();
//	if(outputlevel>1)
//	{
//		printf("minTransformError: %f \n",minTransformError);
//		printf("size: %d \n",h_finalTransformations.size());
//		for(int v=0;v<view_combinations;v++)
//		{
//			printf("v: %d --> ",v);
//			for(int i=0;i<12;i++)
//			{
//	//			printf("%f | ",data[v*12+i]);
//				printf("%f | ",(float)h_finalTransformations[v*12+i]);
//			}
//			printf("\n");
//		}
//	}
//	printf("transformationValidation done! \n");

//	Processor::breakRun();
}

TranformationValidator::TranformationValidator(unsigned int n_views,unsigned int n_rsac,unsigned int outputlevel)
: n_views(n_views), n_rsac(n_rsac), outputlevel(outputlevel)
{

//	block = dim3(transformationErrorestimator.dx,transformationErrorestimator.dy);
//	grid = dim3(640/block.x,480/block.y);

//	DeviceDataParams errorTableParams;
//	errorTableParams.elements = ((n_views-1)*n_views)/2 * n_rsac * grid.x * grid.y;
//	errorTableParams.element_size = sizeof(float);
//	errorTableParams.elementType = FLOAT1;
//	errorTableParams.dataType = Point;
//	addTargetData(addDeviceDataRequest(errorTableParams),ErrorTable);
//
//	DeviceDataParams validPointsTableParams;
//	validPointsTableParams.elements = ((n_views-1)*n_views)/2 * n_rsac * grid.x * grid.y;
//	validPointsTableParams.element_size = sizeof(unsigned int);
//	validPointsTableParams.elementType = UINT1;
//	validPointsTableParams.dataType = Point;
//	addTargetData(addDeviceDataRequest(validPointsTableParams),ValidPointsTable);

//	printf("elements: %d \gxy",errorListParams.elements);

//	DeviceDataParams errorListParams;
//	errorListParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
//	errorListParams.element_size = sizeof(float);
//	errorListParams.elementType = FLOAT1;
//	errorListParams.dataType = Point;
//	addTargetData(addDeviceDataRequest(errorListParams),ErrorList);
//
//
//	DeviceDataParams errorListIdxParams;
//	errorListIdxParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
//	errorListIdxParams.element_size = sizeof(unsigned int);
//	errorListIdxParams.elementType = UINT1;
//	errorListIdxParams.dataType = Indice;
//	addTargetData(addDeviceDataRequest(errorListIdxParams),ErrorListIndices);

//	unsigned int length = (MAXVIEWS*(MAXVIEWS-1))/2;
//	unsigned int h_idx2d[length];
//	unsigned int h_idx2m[length];
//
//	{
//		unsigned i = 0;
//		for(int iy=0;iy<n_views;iy++)
//		{
//			for(int ix = iy+1; ix<n_views;ix++)
//			{
//				h_idx2d[i]=ix;
//				h_idx2m[i]=iy;
//				i++;
//			}
//		}
//	}
//
//	cudaMemcpyToSymbol(device::constant::idx2d,&h_idx2d,length*sizeof(unsigned int));
//	cudaMemcpyToSymbol(device::constant::idx2m,&h_idx2m,length*sizeof(unsigned int));

	DeviceDataParams minErrorTransformationMatricesParams;
	minErrorTransformationMatricesParams.elements = ((n_views-1)*n_views)/2;
	minErrorTransformationMatricesParams.element_size = 12 * sizeof(float);
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
//	checkCudaErrors(cudaMalloc((void **)&errorListMinimumPicker256.input_errorList,n_rsac*sizeof(float)));

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
	errorListMinimumPicker256.input_errorList = thrust::raw_pointer_cast(d_data.data());
	errorListMinimumPicker256.input_errorListIdx = thrust::raw_pointer_cast(d_idx.data());

	errorListMinimumPicker256.n_rsac = n_rsac;
//	errorListMinimumPicker256.n_views = n_views;

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
//	transformationErrorestimator.transformationMetaData = thrust::raw_pointer_cast(d_metaData.data());


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
//	transformationErrorestimator.transformationMetaData = thrust::raw_pointer_cast(d_metaData.data());

	thrust::device_vector<float> d_errorTable(n_rsac*grid.x*grid.y);
	thrust::device_vector<unsigned int> d_validPointsTable(n_rsac*grid.x*grid.y);
	thrust::device_vector<unsigned int> d_errorListIdx(n_rsac);
	transformationErrorestimator.output_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	transformationErrorestimator.output_validPointsTable = thrust::raw_pointer_cast(d_validPointsTable.data());
	transformationErrorestimator.output_errorListIdx = thrust::raw_pointer_cast(d_errorListIdx.data());


	transformationErrorestimator.n_rsac = n_rsac;
	transformationErrorestimator.dist_threshold = 50;
	transformationErrorestimator.angle_threshold = 0.0f;


	device::computeTransformationError<<<grid,block>>>(transformationErrorestimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::host_vector<float> h_errorTable = d_errorTable;
	thrust::host_vector<unsigned int> h_validPoints = d_validPointsTable;
	printf("errortable: ");
	unsigned int validPointsSum = 0;
	float errorSum = 0.f;
	for(int i=0;i<grid.x*grid.y;i++)
	{
		printf("(%d/%f/%d) | ",i,h_errorTable[i],h_validPoints[i]);
		validPointsSum += h_validPoints[i];
		errorSum += h_errorTable[i];
	}
	printf(" \n");
	printf("errorSum: %f | validPointSum: %d | normailzed: %f \n",errorSum,validPointsSum,errorSum/validPointsSum);

//	thrust::device_vector<float> d_errorTable = h_errorTable;
	thrust::device_vector<float> d_errorList(n_rsac);
//	thrust::device_vector<int> d_metaData = h_metaData;


	errorListSumcalculator256.input_errorTable = thrust::raw_pointer_cast(d_errorTable.data());
	errorListSumcalculator256.input_validPoints = thrust::raw_pointer_cast(d_validPointsTable.data());
	errorListSumcalculator256.output_errorList = thrust::raw_pointer_cast(d_errorList.data());
	errorListSumcalculator256.listMetaData = thrust::raw_pointer_cast(d_metaData.data());

	errorListSumcalculator256.min_overlap = (640*480)/10;
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

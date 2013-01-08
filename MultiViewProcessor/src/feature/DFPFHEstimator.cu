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
#include <thrust/copy.h>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "point_info.hpp"


#include "../sink/pcd_io.h"
#include "utils.hpp"
#include "device_utils.hpp"

namespace device
{

	struct DFPFHBaseKernel : public FeatureBaseKernel
	{
		enum
		{
			rx = 5,
			ry = rx,
			lx = 2*rx+1,
			ly = 2*ry+1,
		};

	};

	struct SDPFHEstimator : public DFPFHBaseKernel
	{
		enum
		{
			points_per_block = 32,
			dx = points_per_block * WARP_SIZE,
			dy = 1,

			rx = 10,
			ry = rx,
			lx = 2*rx+1,
			ly = 2*ry+1,

		};

		float4 *input_pos;
		float4 *input_normals;

		float *output_bins;

		unsigned int view;
		float radius;
		float invalidToNormalPointRatio;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float4 shm_pos[points_per_block];
			__shared__ float4 shm_normal[points_per_block];
			__shared__ float shm_histo[points_per_block*bins];

			__shared__ unsigned int shm_off[points_per_block*2];
			__shared__ unsigned int shm_buffer[dx*dy];

			unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
			unsigned int wid = tid / WARP_SIZE;
			unsigned int wtid = tid - wid * WARP_SIZE;

//			unsigned int oy = (blockIdx.x * points_per_block + wid)/640;
//			unsigned int ox = (blockIdx.x * points_per_block + wid) - oy*640;

			int ox,oy;
			if(tid<points_per_block)
			{
				shm_off[tid] 					= oy = (blockIdx.x * points_per_block + tid)/640;
				shm_off[points_per_block+tid] 	= ox = (blockIdx.x * points_per_block + tid) - oy*640;

				shm_pos[tid] = input_pos[blockIdx.z*640*480+oy*640+ox];
				shm_normal[tid] = input_normals[blockIdx.z*640*480+oy*640+ox];
			}

			for(int i=tid;i<bins*points_per_block;i+=dx)
			{
				shm_histo[i] = 0.f;
			}
			__syncthreads();

			if(shm_pos[wid].z==0 || isBackground(shm_pos[wid].w))
			{
				for(int i=wtid;i<bins;i+=WARP_SIZE)
				{
//					output_bins[(blockIdx.z*640*480+blockIdx.x*points_per_block+wid)*bins_n_meta+i] = -1;
					output_bins[640*480*blockIdx.z*bins_n_meta + i*640*480 + blockIdx.x*points_per_block + wid] = 0;
				}
				if(wtid==0)
				{
//					output_bins[(blockIdx.z*640*480+blockIdx.x*points_per_block+wid)*bins_n_meta+bins] = 0;
					output_bins[640*480*blockIdx.z*bins_n_meta + 640*480*bins + blockIdx.x*points_per_block + wid] = 0;
				}
				return;
			}

//			for(int i=wtid;i<bins;i+=WARP_SIZE)
//			{
//				shm_histo[]
//			}

//			ox = shm_off[wid];
//			oy = shm_off[points_per_block+wid];

			__syncthreads();

			unsigned int point_count = 0;
			unsigned int invalid_points = 0;
			for(int j=wtid;j<lx*ly;j+=WARP_SIZE)
			{
				__syncthreads();
				oy = j/ly;
				ox = j - oy*ly;

				oy = shm_off[wid] - ry + oy;
				ox = shm_off[points_per_block+wid] - rx + ox;

				if(oy<0 || oy>=480 || ox <0 || ox>=640 || j==(lx*ly)/2)
					continue;

				float4 p = input_pos[blockIdx.z*640*480+oy*640+ox];
				float4 n = input_normals[blockIdx.z*640*480+oy*640+ox];

				if(!isValid(p.w))
				{
					invalid_points++;
					continue;
				}

				if(lengthf43(minusf43(shm_pos[wid],p))>radius || n.w<0)
				{
					continue;
				}

				float3 ps,pt,ns,nt;
				if( dotf43(shm_normal[wid], minusf43(p,shm_pos[wid]) ) <= dotf43(n, minusf43(shm_pos[wid],p) ) )
				{
					ps = fetch(shm_pos[wid]);
					ns = fetch(shm_normal[wid]);

					pt = fetch(p);
					nt = fetch(n);
				}
				else
				{
					ps = fetch(p);
					ns = fetch(n);

					pt = fetch(shm_pos[wid]);
					nt = fetch(shm_normal[wid]);
				}

//				float3 u = ns;
				float3 d = pt-ps;
				float3 v = cross(d,ns);
//				float3 w = cross(ns,v);

				float vn = norm(v);
				if(vn==0.f || norm(d)==0.f)
				{
					invalid_points++;
					continue;
				}

				point_count++;

				vn = 1.f/vn;
				v = v * vn;

				unsigned int idx = 0;
				float f = dot(v,nt);
				idx = bins_per_feature * ((f + 1.0f) * 0.5f);
				idx = min(idx,bins_per_feature-1) +  0*bins_per_feature + wid*bins;
				atomicAdd(&shm_histo[idx], 1.f);

				f = dot(ns,d)/norm(d);
				idx = bins_per_feature * ((f + 1.0f) * 0.5f);
				idx = min(idx,bins_per_feature-1) +  1*bins_per_feature + wid*bins;
				atomicAdd(&shm_histo[idx], 1.f);

				// v <- w
				v = cross(ns,v);
				f = atan2f(dot(v,nt),dot(ns,nt));
				idx = bins_per_feature * ((f + M_PI) * (1.0f / (2.0f * M_PI)));
				idx = min(idx,bins_per_feature-1) +  2*bins_per_feature + wid*bins;
				atomicAdd(&shm_histo[idx], 1.f);
			}
			__syncthreads();

//			shm_buffer[wid*WARP_SIZE+wtid] = point_count;

			volatile unsigned int *sbuf = &shm_buffer[wid*WARP_SIZE];
			sbuf[wtid] = point_count;
			if(wtid < 16)
			{
				sbuf[wtid] += sbuf[wtid + 16];
				sbuf[wtid] += sbuf[wtid + 8];
				sbuf[wtid] += sbuf[wtid + 4];
				sbuf[wtid] += sbuf[wtid + 2];
				sbuf[wtid] += sbuf[wtid + 1];
			}
			__syncthreads();

			point_count = shm_buffer[wid*WARP_SIZE];

			sbuf[wtid] = invalid_points;
			if(wtid < 16)
			{
				sbuf[wtid] += sbuf[wtid + 16];
				sbuf[wtid] += sbuf[wtid + 8];
				sbuf[wtid] += sbuf[wtid + 4];
				sbuf[wtid] += sbuf[wtid + 2];
				sbuf[wtid] += sbuf[wtid + 1];
			}
			__syncthreads();

			invalid_points = shm_buffer[wid*WARP_SIZE];
//			if((((float)invalid_points)/((float)point_count)) > 0.5f)
//			{
//				printf("(%d/%d) %d / %d = %f \n",wid,wtid,point_count,invalid_points,(((float)invalid_points)/((float)point_count)));
//			}
			__syncthreads();


			bool vali = ( point_count > 0 && (((float)invalid_points)/((float)point_count)) < invalidToNormalPointRatio);
			for(int i=wtid;i<bins;i+=WARP_SIZE)
			{
				float re = (vali)?shm_histo[wid*bins+i]/point_count:0;
//				output_bins[(view*640*480+blockIdx.x*points_per_block+wid)*bins+i] = re;
//				output_bins[640*480*view*bins+i*640*480+blockDim.x*points_per_block + blockIdx.x*points_per_block + wid] = re;

				output_bins[640*480*blockIdx.z*bins_n_meta + i*640*480 + blockIdx.x*points_per_block + wid] = re;
//				output_bins[(blockIdx.z*640*480+blockIdx.x*points_per_block+wid)*bins_n_meta+i] = re;
			}
			if(wtid==0)
			{
				output_bins[640*480*blockIdx.z*bins_n_meta + bins*640*480 + blockIdx.x*points_per_block + wid] = vali;
//				output_bins[(blockIdx.z*640*480+blockIdx.x*points_per_block+wid)*bins_n_meta+bins] = vali;
			}

		}
	};
	__global__ void computeSDPFH(const SDPFHEstimator sdpfhe){ sdpfhe(); }


	struct DFPFHEstimator : public DFPFHBaseKernel
	{
		enum
		{
			points_per_block = 32,
			dx = points_per_block * WARP_SIZE,
			dy = 1,
		};

		float4 *input_pos;
//		float4 *input_normals;
		float *input_bins_sdfpfh;

		float *output_bins;

		float radius;
		int maxReconstructuionLevel;

		template<typename T>
		__device__ __forceinline__ void sum(volatile T* smem_buffer,int wtid) const
		{
//			T reg = smem_buffer[wtid];

			if(wtid<16)
			{
				smem_buffer[wtid] += smem_buffer[wtid + 16];
				smem_buffer[wtid] += smem_buffer[wtid + 8];
				smem_buffer[wtid] += smem_buffer[wtid + 4];
				smem_buffer[wtid] += smem_buffer[wtid + 2];
				smem_buffer[wtid] += smem_buffer[wtid + 1];
			}
		}

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float4 shm_pos[points_per_block];
//			__shared__ float4 shm_normal[points_per_block];
			__shared__ float shm_histo[points_per_block*bins_n_meta];

			__shared__ unsigned int shm_off[points_per_block*2];
			__shared__ float shm_hist_buffer[dx*dy];

			unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
			unsigned int wid = tid / WARP_SIZE;
			unsigned int wtid = tid - wid * WARP_SIZE;

			int ox,oy;
			if(tid<points_per_block)
			{

				shm_off[tid] 					= oy = (blockIdx.x * points_per_block + tid)/640;
				shm_off[points_per_block+tid] 	= ox = (blockIdx.x * points_per_block + tid) - oy*640;

				shm_pos[tid] = input_pos[blockIdx.z*640*480+oy*640+ox];
			}


			for(int i=tid;i<bins_n_meta*points_per_block;i+=dx)
			{
				shm_histo[i] = 0.f;
			}

			__syncthreads();

//			if(tid==1 && blockIdx.x > 4110 && blockIdx.x < 4115 && blockIdx.z==0)
//			{
//				printf("%d/%d -> %d/%d -> %f %f %f %f -> %d %d (%d>%d)->%d %d \n",blockIdx.x,tid,shm_off[tid],shm_off[tid+points_per_block],shm_pos[tid].x,shm_pos[tid].y,shm_pos[tid].z,shm_pos[tid].w,shm_pos[wid].z==0, input_bins_sdfpfh[640*480*blockIdx.z*bins_n_meta + 640*480*bins + blockIdx.x*points_per_block + wid]==0 , getReconstructionLevel(shm_pos[wid].w),maxReconstructuionLevel , getReconstructionLevel(shm_pos[wid].w)>maxReconstructuionLevel, !isForeground(shm_pos[wid].w) );
//
//			}

			if(!(shm_pos[wid].z==0 || input_bins_sdfpfh[640*480*blockIdx.z*bins_n_meta + 640*480*bins + blockIdx.x*points_per_block + wid]==0 || getReconstructionLevel(shm_pos[wid].w)>maxReconstructuionLevel ||   !isForeground(shm_pos[wid].w)))
			{
//				unsigned int point_count = 0;
//				unsigned int invalid_points = 0;
				unsigned int warpRuns = (lx*ly-1)/WARP_SIZE +1;
//				if(blockIdx.z==0 && blockIdx.x==0 && tid==0)
//					printf("warpRuns: %d | lx*ly: %d \n",warpRuns,lx*ly);
				for(int r=0;r<warpRuns;r++)
				{
					unsigned int j = r*WARP_SIZE+wtid;
	//				__syncthreads();
					oy = j/ly;
					ox = j - oy*ly;

					oy = shm_off[wid] - ry + oy;
					ox = shm_off[points_per_block+wid] - rx + ox;

					bool load = false;
					float weight = 1.f;
					if(!(oy<0 || oy>=480 || ox <0 || ox>=640 || j==(lx*ly)/2 || j>=lx*ly))
					{
						float4 p = input_pos[blockIdx.z*640*480+oy*640+ox];

		//				if(!isValid(p.w))
		//				{
		//					invalid_points++;
		//					continue;
		//				}

						float dis;
						if((dis=lengthf43(minusf43(shm_pos[wid],p)))<=radius)
						{
							load = true;
							weight = 1.f/(dis/1000.f);
						}
					}


					//put in warp buffer and reduce
					for(int b=0;b<bins_n_meta;b++)
					{
//						shm_hist_buffer[wid*WARP_SIZE+j] = (load)?input_bins_sdfpfh[640*480*blockIdx.z*bins_n_meta + b*640*480 + oy*640+ox]:0.f;
						shm_hist_buffer[wid*WARP_SIZE+wtid] = (load)?weight*input_bins_sdfpfh[640*480*blockIdx.z*bins_n_meta + b*640*480 + oy*640+ox]:0.f;
						volatile float *smem_buffer = &shm_hist_buffer[wid*WARP_SIZE];
						sum(smem_buffer,wtid);

						if(wtid==0)
						{
//							if(blockIdx.z==0 && blockIdx.x==0 && tid==0)
//								printf("shm_hist[0]: %f | smem_buffer[0]: %f \n",shm_histo[b*points_per_block+wid],smem_buffer[wtid]);

							shm_histo[b*points_per_block+wid] += smem_buffer[wtid];
						}
					}
				}
			}

			__syncthreads();

			for(int i=tid;i<points_per_block*bins_n_meta;i+=dx*dy)
			{
				unsigned int oid = i/points_per_block;
				unsigned int otid = i - oid*points_per_block;
//				if(oid<bins)
					output_bins[blockIdx.z*640*480*bins_n_meta+oid*640*480+blockIdx.x*points_per_block+otid] = (shm_histo[points_per_block*bins+otid]>0.f)?shm_histo[i]/shm_histo[points_per_block*bins+otid]:0.f;
//				else
//					output_bins[blockIdx.z*640*480*bins_n_meta+oid*640*480+blockIdx.x*points_per_block+otid] = shm_histo[i];
			}

		}


	};
	__global__ void computeDFPFH(const DFPFHEstimator dfpfhe){ dfpfhe(); }

	struct MeanDFPFHBlockEstimator : public DFPFHBaseKernel
	{

		enum
		{
			dx = 1024,
			dy = 1,

			max_shared = 11*1024, // 20kb
			shared_lines = max_shared/dx,
			shared_buffer = shared_lines * dx
		};

		float *input_bins;
		float *output_block_mean;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_count[dx];
			__shared__ float shm_buffer[shared_buffer];

			unsigned int tid = threadIdx.x;
			shm_count[tid] = input_bins[blockIdx.z*640*480*bins_n_meta+bins*640*480+blockIdx.x*dx+tid];
			__syncthreads();
			reduceBlock<dx>(shm_count);
			__syncthreads();

			if(tid==0)
			{
				output_block_mean[blockIdx.z*gridDim.x*bins_n_meta + bins*gridDim.x+blockIdx.x] = shm_count[0];// (shm_count[0]>0)?1:0;
			}

			if(shm_count[0]<=0)
				return;

//			if(blockIdx.x==0 && tid==0)
//			{
//				printf("count %f \n",shm_count[0]);
//				output_block_mean[0] = shm_count[0];
//			}

//			shm_buffer[threadIdx.x] = 1;
//			shm_buffer[dx+threadIdx.x] = 2;

			unsigned int loops = (bins-1)/shared_lines +1;
			for(int l=0;l<loops;l++)
			{
				unsigned int lines = ((l+1)*shared_lines<bins)?shared_lines:bins-l*shared_lines;
				for(int i=0;i<lines;i++)
				{
					shm_buffer[i*dx+tid] = input_bins[blockIdx.z*640*480*bins_n_meta+(l*shared_lines+i)*640*480+blockIdx.x*dx+tid];
				}
				__syncthreads();
				reduceBlockNoReg<dx>(shm_buffer,lines);
				__syncthreads();

				if(tid<lines)
					output_block_mean[blockIdx.z*gridDim.x*bins_n_meta+(l*shared_lines+tid)*gridDim.x+blockIdx.x] = shm_buffer[tid*dx];///shm_count[0];

//				__syncthreads();

			}

//

		}

//	  template<unsigned int blockSize, typename T>
//	  __device__ __forceinline__ void reduceBlock(T* shm) const
//	  {
//
//		  unsigned int tid = threadIdx.x;
//		  float sum = shm[tid];
//		  if (blockSize >= 1024) { if(tid < 512) 	{ shm[tid] = sum = sum + shm[tid + 512]; 	} __syncthreads(); }
//		  if (blockSize >=  512) { if(tid < 256) 	{ shm[tid] = sum = sum + shm[tid + 256]; 	} __syncthreads(); }
//		  if (blockSize >=  256) { if(tid < 128) 	{ shm[tid] = sum = sum + shm[tid + 128]; 	} __syncthreads(); }
//		  if (blockSize >=  128) { if(tid < 64) 	{ shm[tid] = sum = sum + shm[tid + 64]; 	} __syncthreads(); }
//
//		  if(tid < 32)
//		  {
//			  volatile T* smem = shm;
//			  if (blockSize >=  64) { smem[tid] = sum = sum + smem[tid + 32]; 	}
//			  if (blockSize >=  32) { smem[tid] = sum = sum + smem[tid + 16]; 	}
//			  if (blockSize >=  16) { smem[tid] = sum = sum + smem[tid +  8]; 	}
//			  if (blockSize >=   8) { smem[tid] = sum = sum + smem[tid +  4]; 	}
//			  if (blockSize >=   4) { smem[tid] = sum = sum + smem[tid +  2]; 	}
//			  if (blockSize >=   2) { smem[tid] = sum = sum + smem[tid +  1]; 	}
//		  }
//	  }
//
//	  template<unsigned int blockSize, typename T>
//	  __device__ __forceinline__ void reduceBlockNoReg(T* shm) const
//	  {
//
//		  unsigned int tid = threadIdx.x;
//		  if (blockSize >= 1024) { if(tid < 512) 	{ shm[tid] += shm[tid + 512]; 	} __syncthreads(); }
//		  if (blockSize >=  512) { if(tid < 256) 	{ shm[tid] += shm[tid + 256]; 	} __syncthreads(); }
//		  if (blockSize >=  256) { if(tid < 128) 	{ shm[tid] += shm[tid + 128]; 	} __syncthreads(); }
//		  if (blockSize >=  128) { if(tid < 64) 	{ shm[tid] += shm[tid + 64]; 	} __syncthreads(); }
//
//		  if(tid < 32)
//		  {
//			  volatile T* smem = shm;
//			  if (blockSize >=  64) { smem[tid] += smem[tid + 32]; 	}
//			  if (blockSize >=  32) { smem[tid] += smem[tid + 16]; 	}
//			  if (blockSize >=  16) { smem[tid] += smem[tid +  8]; 	}
//			  if (blockSize >=   8) { smem[tid] += smem[tid +  4]; 	}
//			  if (blockSize >=   4) { smem[tid] += smem[tid +  2]; 	}
//			  if (blockSize >=   2) { smem[tid] += smem[tid +  1]; 	}
//		  }
//	  }
//
//	  template<unsigned int blockSize, typename T>
//	  __device__ __forceinline__ void reduceBlockNoReg(T* shm,int lines) const
//	  {
//
//		  unsigned int tid = threadIdx.x;
//		  if (blockSize >= 1024) { if(tid < 512) 	{ for(int i=0;i<lines;i++){ shm[i*dx + tid] += shm[i*dx + tid + 512]; } 	} __syncthreads(); }
//		  if (blockSize >=  512) { if(tid < 256) 	{ for(int i=0;i<lines;i++){ shm[i*dx + tid] += shm[i*dx + tid + 256]; } 	} __syncthreads(); }
//		  if (blockSize >=  256) { if(tid < 128) 	{ for(int i=0;i<lines;i++){ shm[i*dx + tid] += shm[i*dx + tid + 128]; }		} __syncthreads(); }
//		  if (blockSize >=  128) { if(tid < 64) 	{ for(int i=0;i<lines;i++){ shm[i*dx + tid] += shm[i*dx + tid +  64]; } 	} __syncthreads(); }
//
//		  if(tid < 32)
//		  {
//			  volatile T* smem = shm;
//			  if (blockSize >=  64) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid + 32]; } 	}
//			  if (blockSize >=  32) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid + 16]; } 	}
//			  if (blockSize >=  16) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid +  8]; }	}
//			  if (blockSize >=   8) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid +  4]; } 	}
//			  if (blockSize >=   4) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid +  2]; } 	}
//			  if (blockSize >=   2) { for(int i=0;i<lines;i++){ smem[i*dx + tid] += smem[i*dx + tid +  1]; } 	}
//		  }
//	  }

	};
	__global__ void computeBlockMeanDFPFH(const MeanDFPFHBlockEstimator blockMean){ blockMean(); }

	template<unsigned int blockSize>
	struct MeanDFPFHEstimator : public DFPFHBaseKernel
	{
		float *input_bins;
		float *output_block_mean;
		unsigned int length;

		enum
		{
			dx = blockSize,

			max_shared = 11*1024, // 11kb
			shared_lines = max_shared/dx,
			shared_buffer = shared_lines * dx
		};

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_count[dx];
			__shared__ float shm_buffer[shared_buffer];

	        unsigned int tid = threadIdx.x;
	        unsigned int gridSize = dx*2*gridDim.x;

//	        if(tid==0)
//	        	printf("block: %d \n",blockIdx.z);
//
//	        if(blockIdx.z==1 & tid==0)
//	        {
//	        	for(int t=0;t<length;t+=50)
//	        	{
//		    			printf("inKernelCount(%d): %f \n",t,input_bins[blockIdx.z*length*bins_n_meta+bins*length+t]);
//	        	}
//	        }

	    	float sum = 0.f;
	    	unsigned int i = blockIdx.x*dx*2 + tid;
	    	while(i<length)
	    	{
	    		sum += input_bins[blockIdx.z*length*bins_n_meta+bins*length+i];

	            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
	            if ( (i + dx) < length)
	            	sum += input_bins[blockIdx.z*length*bins_n_meta+bins*length+i+dx];

	            i += gridSize;
	    	}

	    	shm_count[tid] = sum;
	        __syncthreads();
	        reduceBlock<dx>(shm_count);
	        __syncthreads();

	        if(tid==0)
			{
	        	output_block_mean[blockIdx.z*bins_n_meta+bins] = shm_count[0];// (shm_count[0]>0)?1.f:0.f;
//	        	printf("count: %f \n",shm_count[0]);
			}

	        if(shm_count[0]<=0)
	        	return;

	        unsigned int loops = (bins-1)/shared_lines +1;

//			if(threadIdx.x==0 && blockIdx.x==0)
//				printf("loops: %d \n",loops);

			for(int l=0;l<loops;l++)
			{
				unsigned int lines = ((l+1)*shared_lines<bins)?shared_lines:bins-l*shared_lines;
				for(int b=0;b<lines;b++)
				{
					sum = 0.f;
					i = blockIdx.x*dx*2 + threadIdx.x;
					while(i<length)
					{
						sum += input_bins[blockIdx.z*length*bins_n_meta+(l*shared_lines+b)*length+i];

						if ( (i + dx) < length)
							sum += input_bins[blockIdx.z*length*bins_n_meta+(l*shared_lines+b)*length+i+dx];

						i += gridSize;
					}

					shm_buffer[b*dx+tid] = sum;
				}
				__syncthreads();
				reduceBlockNoReg<dx>(shm_buffer,lines);
				__syncthreads();
				if(tid<lines)
					output_block_mean[blockIdx.z*bins_n_meta+(l*shared_lines+tid)] = shm_buffer[tid*dx]/shm_count[0];
			}

		}
	};
	__global__ void computeMeanDFPFH(const MeanDFPFHEstimator<1024> meandfpfhe){ meandfpfhe(); }
	__global__ void computeMeanDFPFH(const MeanDFPFHEstimator<512> meandfpfhe){ meandfpfhe(); }

	struct DivDFPFHEstimator: public DFPFHBaseKernel
	{
		enum
		{
			dx = 1024,

		};

		float *input_dfpfh_bins;
		float *input_mean_bins;
		float *output_div;

//		__device__  __forceinline__ float
//		klDivergence(float *feature, float *mean, unsigned int bins_count, unsigned int offset_feature, unsigned int offset_mean) const
//		{
//
//			float div = 0.f;
//			for(int i=0;i<bins_count;i++)
//			{
//				float p = feature[i*offset_feature];
//				float m = mean[i*offset_mean];
//
//				if(p/m>0)
//					div += (p - m) * __logf(p/m);
//			}
//
//			return div;
//		}
//
//		__device__  __forceinline__ float
//		klEuclideanDivergence(float *feature, float *mean, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_feature, unsigned int offset_mean) const
//		{
//			float div = 0.f;
//
//			for(int f=0;f<feature_count;f++)
//			{
//				float tmpDiv = 0.f;
//				for(int i=0;i<bin_count_per_feature;i++)
//				{
//					float p = feature[(f*bin_count_per_feature+i)*offset_feature];
//					float m = mean[(f*bin_count_per_feature+i)*offset_mean];
//
//					if(p/m>0)
//						tmpDiv += (p - m) * __logf(p/m);
//				}
//				div += (tmpDiv * tmpDiv);
//			}
//
//			return sqrtf(div);
//		}

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_mean[bins];

			unsigned int tid = threadIdx.x;
			unsigned int i = blockIdx.x*dx+tid;
//			unsigned int off = gridDim.x;

			if(tid<bins)
				shm_mean[tid] = input_mean_bins[blockIdx.z*bins_n_meta+tid];

			__syncthreads();

//			float div = klDivergence(&input_dfpfh_bins[blockIdx.z*640*480*bins_n_meta+i],shm_mean,bins,640*480,1);

			float div = klEuclideanDivergence(&input_dfpfh_bins[blockIdx.z*640*480*bins_n_meta+i],shm_mean,features,bins_per_feature,640*480,1);
			output_div[blockIdx.z*640*480+i] = div;

		}

	};
	__global__ void computeDivDFPFH(const DivDFPFHEstimator de){ de(); }


	struct SigmaDFPFHBlock : public DFPFHBaseKernel
	{
		enum
		{
			dx = 1024,
		};

		float *input_div;
		float *output_div_block;
		unsigned int length;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_buffer[dx];

			unsigned int tid = threadIdx.x;
			unsigned int i = blockIdx.x*dx*2 + threadIdx.x;
			unsigned int gridSize = dx*2*gridDim.x;

			float sum = 0.f;
//			float count = 0.f;
			while(i<length)
			{
				float tmp =  input_div[blockIdx.z*640*480+i];
				sum += tmp*tmp;

				if ((i+dx)<length)
				{
					tmp = input_div[blockIdx.z*640*480+i+dx];
					sum += tmp * tmp;
				}

				i += gridSize;
			}

			shm_buffer[tid] = sum;
			__syncthreads();
			reduceBlock<dx>(shm_buffer);
			__syncthreads();

			if(tid==0)
				output_div_block[blockIdx.z*gridDim.x+blockIdx.x] = shm_buffer[0];

//			if(tid==0)
//				printf("sigBlock: %d -> %f \n",blockIdx.x,shm_buffer[0]);
		}

	};
	__global__ void computeSigmaDFPFHBlock(const SigmaDFPFHBlock sb){ sb(); }


	template<unsigned int blockSize>
	struct SigmaDFPFH : public DFPFHBaseKernel
	{
		enum
		{
			dx = blockSize,
		};

		float *input_sig_block;
		float *input_mean_data;
		float *output_sigmas;

		unsigned int length;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_buffer[dx];

			unsigned int tid = threadIdx.x;
			unsigned int i = blockIdx.x*dx*2 + threadIdx.x;
			unsigned int gridSize = dx*2*gridDim.x;

			float sum = 0.f;
			while(i<length)
			{
				sum += input_sig_block[blockIdx.z*length+i];

				if ((i+dx)<length)
					sum += input_sig_block[blockIdx.z*length+i+dx];

				 i += gridSize;
			}
			shm_buffer[tid] = sum;
			reduceBlock<dx>(shm_buffer);
			__syncthreads();

			if(tid==0)
			{
				float points = input_mean_data[blockIdx.z*bins_n_meta+bins];
//				printf("(%d) points: %f sigma: %f \n",blockIdx.z,points,sqrtf(shm_buffer[0]/points));
				output_sigmas[blockIdx.z] = (points>0)?sqrtf(shm_buffer[0]/points):-1.f;
			}

		}

	};
	__global__ void computeSigmaDFPFH(const SigmaDFPFH<512> sig){ sig(); }

	struct PersistanceDFPFHEstimator : public DFPFHBaseKernel
	{
		enum
		{
			dx = 1024,

		};

		float beta;
		bool intersection;

		float *input_div1;
		float *input_div2;

		float *input_sigma1;
		float *input_sigma2;

		float *input_bins_n_meta1;
		float *input_bins_n_meta2;

		int *persistence_map;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int tid = threadIdx.x;
			unsigned int vid = blockIdx.y*blockDim.x*blockDim.y+blockIdx.x*blockDim.x+tid;
			unsigned int gid = blockIdx.z*640*480+vid;

			int idx = -1;
			if( input_bins_n_meta1[blockIdx.z*640*480*bins_n_meta+bins*640*480+vid]>0 && input_bins_n_meta2[blockIdx.z*640*480*bins_n_meta+bins*640*480+vid]>0 )
			{
				if( (input_div1[gid] > beta * input_sigma1[blockIdx.z]) && (input_div2[gid] > beta * input_sigma2[blockIdx.z]) )
				{
					if(intersection)
					{
						if(persistence_map[gid] >= 0)
							idx = gid;
					}
					else
						idx = gid;
				}
			}
			persistence_map[gid] = idx;
		}
	};
	__global__ void computePersistanceDFPFHFeatures(const PersistanceDFPFHEstimator pers){ pers(); }

	struct PersistanceIntersectionDFPFHEstimator : public DFPFHBaseKernel
	{
		enum
		{
			dx = 1024,

		};

		float beta;
		unsigned int n_view;

		float **input_div;
		float **input_sigma;

		int output_persistence_map;

		__device__ __forceinline__ void
		operator () () const
		{

		}

	};

	struct PersistanceMapLength
	{
		enum
		{
			dx = 1024,
		};

		int *input_persistanceMap;
		unsigned int *output_persistanceMapLenth;

		unsigned int length;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

			if(i+1>length)
			{
				return;
			}

			unsigned int v1 = input_persistanceMap[i];
			v1 /= (640*480);

			if(i+1==length)
			{
				output_persistanceMapLenth[v1] = i+1;
				return;
			}

			unsigned int v2 = input_persistanceMap[i+1];
			v2 /= (640*480);

			if(v2>v1)
				output_persistanceMapLenth[v1] = i+1;
		}
	};
	__global__ void computePersistanceMapLength(const PersistanceMapLength mapLength){ mapLength(); }

}


device::SDPFHEstimator sdpfh1;
device::SDPFHEstimator sdpfh2;

device::DFPFHEstimator dfpfh1;
device::DFPFHEstimator dfpfh2;


device::MeanDFPFHBlockEstimator meanBlock1;
device::MeanDFPFHBlockEstimator meanBlock2;

device::MeanDFPFHEstimator<512> mean1;
device::MeanDFPFHEstimator<512> mean2;


device::DivDFPFHEstimator div1;
device::DivDFPFHEstimator div2;

device::SigmaDFPFHBlock sigmaBlock1;
device::SigmaDFPFHBlock sigmaBlock2;

device::SigmaDFPFH<512> sigma1;
device::SigmaDFPFH<512> sigma2;

device::PersistanceDFPFHEstimator persistance;


void
DFPFHEstimator::init()
{
	sdpfh1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	sdpfh1.input_normals = (float4 *)getInputDataPointer(PointNormal);

	sdpfh2.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	sdpfh2.input_normals = (float4 *)getInputDataPointer(PointNormal);

	dfpfh1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	dfpfh2.input_pos = (float4 *)getInputDataPointer(PointCoordinates);

//	dfpfh1.input_bins_sdfpfh = (float *)getTargetDataPointer(SDPFHistogram1);
//	dfpfh1.output_bins = (float *)getTargetDataPointer(DFPFHistogram1);
}


void
DFPFHEstimator::execute()
{

	unsigned int copyIdx = 1;
	unsigned int n_radii = 3;
	float radii[] = {10.f,15.f,20.f,25.f};

	if(n_radii<2)
	{
		printf("Error not enough raddi for persistance analysis! \n");
		exit(0);
	}

	thrust::device_vector<float> d_sdpfh1(n_view*640*480*sdpfh1.bins_n_meta);
	thrust::device_vector<float> d_sdpfh2(n_view*640*480*sdpfh2.bins_n_meta);

	sdpfh1.radius = radii[0];
	sdpfh1.invalidToNormalPointRatio = 0.5;
	sdpfh1.output_bins = thrust::raw_pointer_cast(d_sdpfh1.data());

	sdpfh2.radius = radii[1];
	sdpfh2.invalidToNormalPointRatio = 0.5;
	sdpfh2.output_bins = thrust::raw_pointer_cast(d_sdpfh2.data());


	dim3 block(sdpfh1.dx);
	dim3 grid(640*480/sdpfh1.points_per_block,1,n_view);
	device::computeSDPFH<<<grid,block>>>(sdpfh1);
	device::computeSDPFH<<<grid,block>>>(sdpfh2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_vector<float> d_dfpfh1(n_view*640*480*dfpfh1.bins_n_meta);
	thrust::device_vector<float> d_dfpfh2(n_view*640*480*sdpfh2.bins_n_meta);

	dfpfh1.radius = radii[0];
	dfpfh1.maxReconstructuionLevel = 0;
	dfpfh1.input_bins_sdfpfh = sdpfh1.output_bins;
	dfpfh1.output_bins = thrust::raw_pointer_cast(d_dfpfh1.data());

	dfpfh2.radius = radii[1];
	dfpfh2.maxReconstructuionLevel = 0;
	dfpfh2.input_bins_sdfpfh = sdpfh2.output_bins;
	dfpfh2.output_bins = thrust::raw_pointer_cast(d_dfpfh2.data());


	device::computeDFPFH<<<grid,block>>>(dfpfh1);
	device::computeDFPFH<<<grid,block>>>(dfpfh2);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_ptr<float> dfpfh_ptr = thrust::device_pointer_cast( (float *) getTargetDataPointer(DFPFHistogram) );
	if(copyIdx==0)
		thrust::copy(d_dfpfh1.data(),d_dfpfh1.data()+d_dfpfh1.size(),dfpfh_ptr);
	if(copyIdx==1)
		thrust::copy(d_dfpfh2.data(),d_dfpfh2.data()+d_dfpfh2.size(),dfpfh_ptr);

	thrust::device_vector<float> d_mean_block1((n_view*640*480)/meanBlock1.dx * meanBlock1.bins_n_meta);
	thrust::device_vector<float> d_mean_block2((n_view*640*480)/meanBlock2.dx * meanBlock2.bins_n_meta);

	meanBlock1.input_bins = dfpfh1.output_bins;
	meanBlock1.output_block_mean = thrust::raw_pointer_cast(d_mean_block1.data());

	meanBlock2.input_bins = dfpfh2.output_bins;
	meanBlock2.output_block_mean = thrust::raw_pointer_cast(d_mean_block2.data());

	dim3 meanPatchBlock(meanBlock1.dx);
	dim3 meanPatchGrid((640*480)/meanBlock1.dx,1,n_view);
	device::computeBlockMeanDFPFH<<<meanPatchGrid,meanPatchBlock>>>(meanBlock1);
	device::computeBlockMeanDFPFH<<<meanPatchGrid,meanPatchBlock>>>(meanBlock2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_vector<float> d_mean1(n_view*mean1.bins_n_meta);
	thrust::device_vector<float> d_mean2(n_view*mean2.bins_n_meta);

	mean1.input_bins = meanBlock1.output_block_mean;
	mean1.length = meanPatchGrid.x;
	mean1.output_block_mean = thrust::raw_pointer_cast(d_mean1.data());

	mean2.input_bins = meanBlock2.output_block_mean;
	mean2.length = meanPatchGrid.x;
	mean2.output_block_mean = thrust::raw_pointer_cast(d_mean2.data());


	dim3 meanGrid(1,1,n_view);
	device::computeMeanDFPFH<<<meanGrid,mean1.dx>>>(mean1);
	device::computeMeanDFPFH<<<meanGrid,mean2.dx>>>(mean2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_vector<float> d_div1(n_view*640*480);
	thrust::device_vector<float> d_div2(n_view*640*480);

	div1.input_dfpfh_bins = dfpfh1.output_bins;
	div1.input_mean_bins = mean1.output_block_mean;
	div1.output_div = thrust::raw_pointer_cast(d_div1.data());

	div2.input_dfpfh_bins = dfpfh2.output_bins;
	div2.input_mean_bins = mean2.output_block_mean;
	div2.output_div = thrust::raw_pointer_cast(d_div2.data());


	dim3 divGrid((640*480)/div1.dx,1,n_view);
	device::computeDivDFPFH<<<divGrid,div1.dx>>>(div1);
	device::computeDivDFPFH<<<divGrid,div2.dx>>>(div2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_vector<float> d_sigma_block1(n_view*(640*480)/sigmaBlock1.dx);
	thrust::device_vector<float> d_sigma_block2(n_view*(640*480)/sigmaBlock2.dx);

	sigmaBlock1.input_div = div1.output_div;
	sigmaBlock1.output_div_block = thrust::raw_pointer_cast(d_sigma_block1.data());
	sigmaBlock1.length = 640*480;

	sigmaBlock2.input_div = div2.output_div;
	sigmaBlock2.output_div_block = thrust::raw_pointer_cast(d_sigma_block2.data());
	sigmaBlock2.length = 640*480;

	dim3 sigGrid((640*480)/(sigmaBlock1.dx*2),1,n_view);
	device::computeSigmaDFPFHBlock<<<sigGrid,sigmaBlock1.dx>>>(sigmaBlock1);
	device::computeSigmaDFPFHBlock<<<sigGrid,sigmaBlock1.dx>>>(sigmaBlock2);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_vector<float> d_sigma1(n_view);
	thrust::device_vector<float> d_sigma2(n_view);

	sigma1.input_sig_block = sigmaBlock1.output_div_block;
	sigma1.input_mean_data = mean1.output_block_mean;
	sigma1.output_sigmas = thrust::raw_pointer_cast(d_sigma1.data());
	sigma1.length = sigGrid.x;

	sigma2.input_sig_block = sigmaBlock2.output_div_block;
	sigma2.input_mean_data = mean2.output_block_mean;
	sigma2.output_sigmas = thrust::raw_pointer_cast(d_sigma2.data());
	sigma2.length = sigGrid.x;


	device::computeSigmaDFPFH<<<dim3(1,1,n_view),sigma1.dx>>>(sigma1);
	device::computeSigmaDFPFH<<<dim3(1,1,n_view),sigma2.dx>>>(sigma2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_vector<int> d_persistance_map(n_view*640*480);

	persistance.input_div1 = div1.output_div;
	persistance.input_div2 = div2.output_div;

	persistance.input_sigma1 = sigma1.output_sigmas;
	persistance.input_sigma2 = sigma2.output_sigmas;

	persistance.input_bins_n_meta1 = dfpfh1.output_bins;
	persistance.input_bins_n_meta2 = dfpfh2.output_bins;

	persistance.persistence_map = thrust::raw_pointer_cast(d_persistance_map.data());

	persistance.beta = 1.f;
	persistance.intersection = true;
	if(persistance.intersection)
		cudaMemset(persistance.persistence_map,0,n_view*640*480);

	device::computePersistanceDFPFHFeatures<<<dim3((640*480)/persistance.dx,1,n_view),persistance.dx>>>(persistance);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if(outputlevel>2)
		printf("persistence: 1 \n");

	for(int il=2;il<n_radii;il++)
	{
		bool toggle = (il%2==0);
		if(toggle)
		{
			sdpfh1.radius = radii[il];
			device::computeSDPFH<<<grid,block>>>(sdpfh1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			dfpfh1.radius = radii[il];
			device::computeDFPFH<<<grid,block>>>(dfpfh1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeBlockMeanDFPFH<<<meanPatchGrid,meanPatchBlock>>>(meanBlock1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeMeanDFPFH<<<meanGrid,mean1.dx>>>(mean1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeDivDFPFH<<<divGrid,div1.dx>>>(div1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeSigmaDFPFHBlock<<<sigGrid,sigmaBlock1.dx>>>(sigmaBlock1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeSigmaDFPFH<<<dim3(1,1,n_view),sigma1.dx>>>(sigma1);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			if(copyIdx==il)
				thrust::copy(d_dfpfh1.data(),d_dfpfh1.data()+d_dfpfh1.size(),dfpfh_ptr);
		}
		else
		{
			sdpfh2.radius = radii[il];
			device::computeSDPFH<<<grid,block>>>(sdpfh2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			dfpfh2.radius = radii[il];
			device::computeDFPFH<<<grid,block>>>(dfpfh2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeBlockMeanDFPFH<<<meanPatchGrid,meanPatchBlock>>>(meanBlock2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeMeanDFPFH<<<meanGrid,mean1.dx>>>(mean2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeDivDFPFH<<<divGrid,div1.dx>>>(div2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeSigmaDFPFHBlock<<<sigGrid,sigmaBlock1.dx>>>(sigmaBlock2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			device::computeSigmaDFPFH<<<dim3(1,1,n_view),sigma1.dx>>>(sigma2);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			if(copyIdx==il)
				thrust::copy(d_dfpfh2.data(),d_dfpfh2.data()+d_dfpfh2.size(),dfpfh_ptr);
		}

		device::computePersistanceDFPFHFeatures<<<dim3((640*480)/persistance.dx,1,n_view),persistance.dx>>>(persistance);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		if(outputlevel>2)
			printf("persistence: %d \n",il);


	}



//	thrust::host_vector<int> h_persistance_map = d_persistance_map;

	thrust::device_ptr<int> last = thrust::remove(d_persistance_map.data(),d_persistance_map.data()+d_persistance_map.size(),-1);
//	thrust::device_vector<int> d_clearedPersiatnceMap(d_persistance_map.data(),last);

//	thrust::device_ptr idxLength_ptr = thrust::device_pointer_cast((unsigned int *)getTargetDataPointer(PDFPFHIdxLength));
	thrust::device_vector<unsigned int> d_persistanceMapLength(n_view);

//	printf("d_clearedPersiatnceMap.size(): %d \n",d_clearedPersiatnceMap.size());

//	device::PersistanceMapLength mapLength;
//	mapLength.input_persistanceMap = thrust::raw_pointer_cast(d_clearedPersiatnceMap.data());
//	mapLength.output_persistanceMapLenth = thrust::raw_pointer_cast(d_persistanceMapLength.data());
//	mapLength.length = d_clearedPersiatnceMap.size();
//	device::computePersistanceMapLength<<< ((d_clearedPersiatnceMap.size()-1)/mapLength.dx+1),mapLength.dx >>>(mapLength);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int persistanceMapLength = (unsigned int) (last - d_persistance_map.data());
	device::PersistanceMapLength mapLength;
	mapLength.input_persistanceMap = thrust::raw_pointer_cast(d_persistance_map.data());
	mapLength.output_persistanceMapLenth = thrust::raw_pointer_cast(d_persistanceMapLength.data());
	mapLength.length = persistanceMapLength;
	device::computePersistanceMapLength<<< ((persistanceMapLength-1)/mapLength.dx+1),mapLength.dx >>>(mapLength);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_ptr<int> idxList_ptr = thrust::device_pointer_cast((int *)getTargetDataPointer(PDFPFHIdxList));
	thrust::copy(d_persistance_map.data(),d_persistance_map.data()+mapLength.length,idxList_ptr);

	thrust::device_ptr<unsigned int> idxLength_ptr = thrust::device_pointer_cast((unsigned int *)getTargetDataPointer(PDFPFHIdxLength));
	thrust::copy(d_persistanceMapLength.data(),d_persistanceMapLength.data()+n_view,idxLength_ptr);


	if(outputlevel>1)
	{
		printf("length: %d -> %d \n",d_persistance_map.size(),persistanceMapLength);
		thrust::host_vector<unsigned int> h_persistanceMapLength = d_persistanceMapLength;
		for(int l=0;l<h_persistanceMapLength.size();l++)
		{
			int length = h_persistanceMapLength.data()[l];
			printf("%d -> %d \n",l,length);
		}

		thrust::host_vector<int> h_clearedPersiatnceMap = d_persistance_map;
		int *dataLength = h_clearedPersiatnceMap.data();
		int lidx = h_persistanceMapLength.data()[0];
		printf("%d %d %d \n",dataLength[lidx-1]/(640*480),dataLength[lidx]/(640*480),dataLength[lidx+1]/(640*480));
	}

	char path[50];
	bool output = outputlevel>1;
	if(output)
	{
		thrust::host_vector<int> h_persistance_map = d_persistance_map;
//
//		thrust::device_ptr<int> last = thrust::remove(d_persistance_map.data(),d_persistance_map.data()+d_persistance_map.size(),-1);

//		int length_pMap = last - d_persistance_map.data();
//		printf("length pMap: %d \n",length_pMap);

		thrust::device_vector<int> d_clearedPersiatnceMap(d_persistance_map.data(),last);
		thrust::host_vector<int> h_clearedPersiatnceMap = d_clearedPersiatnceMap;


		thrust::device_ptr<float4> dptr = thrust::device_pointer_cast(dfpfh1.input_pos);
		thrust::device_vector<float4> d_pos(dptr,dptr+n_view*640*480);
		thrust::host_vector<float4> h_pos = d_pos;


		float4 *pos = h_pos.data();
		int *persistance_map = h_persistance_map.data();
		uchar4 *h_map = (uchar4 *)malloc(640*480*sizeof(uchar4));
		int *clearedData = h_clearedPersiatnceMap.data();

		for(int v=0;v<n_view;v++)
		{
			for(int i=0;i<640*480;i++)
			{
				unsigned char g = (pos[v*640*480+i].z/10000.f)*255.f;
//				if(persistance_map[v*640*480+i]>=0)
//				{
//					if(persistance_map[v*640*480+i]!=v*640*480+i)
//						printf("%d != %d \n",persistance_map[v*640*480+i],v*640*480+i);
//
//					g=255;
//				}
				h_map[i] = make_uchar4(g,g,g,128);
			}

			for(int i=0;i<h_clearedPersiatnceMap.size();i++)
			{
				int idx = clearedData[i];
				if(idx >= 0)
				{
					unsigned int vi = idx/(640*480);
					if(vi==v)
						h_map[idx-v*640*480] = make_uchar4(255,0,0,128);
				}
				else
					printf("oOOOO \n");
			}

			sprintf(path,"/home/avo/pcds/persistance_r0_r1%d.ppm",v);
			sdkSavePPM4ub(path,(unsigned char*)h_map,640,480);

		}
	}


	printf("done dfpfh! \n");
}



DFPFHEstimator::DFPFHEstimator(unsigned int n_view,unsigned int outputlevel): n_view(n_view), outputlevel(outputlevel)
{
	DeviceDataParams paramsDFPFH;
	paramsDFPFH.elements = 640*480*n_view;
	paramsDFPFH.element_size = (sdpfh1.bins_n_meta) * sizeof(float);
	paramsDFPFH.elementType = FLOAT1;
	paramsDFPFH.dataType = Histogramm;
	addTargetData(addDeviceDataRequest(paramsDFPFH),DFPFHistogram);

	DeviceDataParams paramsPersistenceIdxList;
	paramsPersistenceIdxList.elements = 640*480*n_view;
	paramsPersistenceIdxList.element_size = sizeof(int);
	paramsPersistenceIdxList.elementType = INT1;
	paramsPersistenceIdxList.dataType = Indice;
	addTargetData(addDeviceDataRequest(paramsPersistenceIdxList),PDFPFHIdxList);

	DeviceDataParams paramsPersistenceIdxLength;
	paramsPersistenceIdxLength.elements = n_view;
	paramsPersistenceIdxLength.element_size = sizeof(unsigned int);
	paramsPersistenceIdxLength.elementType = UINT1;
	paramsPersistenceIdxLength.dataType = MetaData;
	addTargetData(addDeviceDataRequest(paramsPersistenceIdxLength),PDFPFHIdxLength);

}

DFPFHEstimator::~DFPFHEstimator()
{
	// TODO Auto-generated destructor stub
}


void
DFPFHEstimator::TestDFPFHE()
{

	thrust::host_vector<float4> h_inp_pos(n_view*640*480);
	thrust::host_vector<float4> h_inp_normals(n_view*640*480);
	for(int i=0;i<h_inp_pos.size();i++)
	{
		unsigned int v = i/(640*480);
		unsigned int y = (i-v*(640*480))/640;
		unsigned int x = i-v*(640*480)-y*640;

		h_inp_pos[i] = make_float4(x,y,1000+v,0);
		device::setValid(h_inp_pos[i].w);
		device::setForeground(h_inp_pos[i].w);
//		device::unsetReconstructed(h_inp_pos[i].w);
		device::setReconstructed(h_inp_pos[i].w,0);
		h_inp_normals[i] = make_float4(1,0,0,0);
	}

	thrust::device_vector<float4> d_inp_pos = h_inp_pos;
	thrust::device_vector<float4> d_inp_normal = h_inp_normals;
	thrust::device_vector<float> d_dspfph(n_view*640*480*sdpfh1.bins_n_meta);

	sdpfh1.input_pos = thrust::raw_pointer_cast(d_inp_pos.data());
	sdpfh1.input_normals = thrust::raw_pointer_cast(d_inp_normal.data());
	sdpfh1.output_bins = thrust::raw_pointer_cast(d_dspfph.data());
	sdpfh1.view = 0;
	sdpfh1.radius = 15.0f;
	sdpfh1.invalidToNormalPointRatio = 0.5;

	printf("dspdf \n");
	dim3 block(sdpfh1.dx);
	dim3 grid(640*480/sdpfh1.points_per_block,1,1);
	device::computeSDPFH<<<grid,block>>>(sdpfh1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

//	thrust::host_vector<float> h_out_histo = d_dspfph;

//	float *data = h_out_histo.data();
//	for(int i=0;i<640*480*n_view;i+=10000)
//	{
//		if(data[i*33]!=-1)
//		{
//			printf("(%d) ",i);
//			for(int j=0;j<3;j++)
//			{
//				for(int h=0;h<11;h++)
//				{
//					printf("%f ",data[i*33+j*11+h]);
//
//				}
//				printf(" || ");
//			}
//			printf("\n");
//		}
//	}

	thrust::device_vector<float> d_dfpfh(n_view*640*480*dfpfh1.bins_n_meta);

	dfpfh1.input_pos = thrust::raw_pointer_cast(d_inp_pos.data());
	dfpfh1.input_bins_sdfpfh = thrust::raw_pointer_cast(d_dspfph.data());
	dfpfh1.output_bins = thrust::raw_pointer_cast(d_dfpfh.data());
	dfpfh1.radius = 15.f;
	dfpfh1.maxReconstructuionLevel = 0;

	printf("dfpdf \n");
	device::computeDFPFH<<<grid,block>>>(dfpfh1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float4> h_pos_test = d_inp_pos;
	for(int i=0;i<33;i++)
	{
		unsigned int oy = i/640;
		unsigned int ox = i - oy*640;
		printf("(%d %d) %f %f %f %f \n",ox,oy,h_pos_test[i].x,h_pos_test[i].y,h_pos_test[i].z,h_pos_test[i].w);
	}


	thrust::host_vector<float> h_out_dfpfh = d_dfpfh;
	printf("dfpfh.size: %d \n",h_out_dfpfh.size());
	float *data = h_out_dfpfh.data();
//	for(int v=0;v<n_view;v++)
//	{
//		for(int i=0;i<640*480;i+=10000)
//		{
//			if(data[v*640*480*(sdpfhEstimator1.bins_n_meta)+sdpfhEstimator1.bins*640*480+i]!=0)
//			{
//				printf("(%d) ",i);
//				for(int j=0;j<3;j++)
//				{
//					for(int h=0;h<11;h++)
//					{
//	//					printf("%f ",data[i*33+j*11+h]);
//						//TODO
//						printf("%f ",data[v*640*480*(sdpfhEstimator1.bins_n_meta)+(j*11+h)*640*480]);
//
//					}
//					printf(" || ");
//				}
//				printf("\n");
//			}
//		}
//	}

	for(int v=0;v<1;v++)
	{
		for(int i=0;i<5;i+=1)
		{

			for(int j=0;j<3;j++)
			{
				for(int h=0;h<11;h++)
				{
	//					printf("%f ",data[i*33+j*11+h]);
					//TODO
					printf("%f ",data[v*640*480*(sdpfh1.bins_n_meta)+(j*11+h)*640*480+i]);

				}
				printf(" || ");
			}

			printf("|| meta: %f \n",data[v*640*480*(sdpfh1.bins_n_meta)+sdpfh1.bins*640*480+i]);
		}
	}



//	thrust::host_vector<float> h_test_dfpfh(n_view*640*480*blockMean1.bins_n_meta);
//	for(int i=0;i<blockMean1.bins;i++)
//		for(int p=0;p<640*480;p++)
//			h_test_dfpfh[i*640*480+p] = i+1;
//
//	for(int p=0;p<640*480;p++)
//				h_test_dfpfh[blockMean1.bins*640*480+p] = 1;
//
//	thrust::device_vector<float> d_test_dfpfh = h_test_dfpfh;
//	blockMean1.input_bins = thrust::raw_pointer_cast(d_test_dfpfh.data());
	meanBlock1.input_bins = dfpfh1.output_bins;

	thrust::device_vector<float> d_test_output((640*480)/meanBlock1.dx * meanBlock1.bins_n_meta);
	meanBlock1.output_block_mean = thrust::raw_pointer_cast(d_test_output.data());

	dim3 meanBlock(meanBlock1.dx);
	dim3 meanGrid(640*480/meanBlock1.dx,1,n_view);
	device::computeBlockMeanDFPFH<<<meanGrid,meanBlock>>>(meanBlock1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float> h_out_mean_dfpfh = d_test_output;
	printf("dfpfh.size: %d meanGrid: %d \n",h_out_mean_dfpfh.size(),meanGrid.x);

	float *data3 = h_out_mean_dfpfh.data();
//	printf("test: %f \n",data3[0]);
	for(int v=0;v<1;v++)
	{
		for(int i=0;i<1;i++)
		{

			for(int j=0;j<3;j++)
			{
				for(int h=0;h<11;h++)
				{
	//					printf("%f ",data[i*33+j*11+h]);
					//TODO
					printf("%f ",data3[v*meanGrid.x*meanBlock1.bins_n_meta+(j*11+h)*meanGrid.x+i]);

				}
				printf(" || ");
			}

			printf("|| meta: %f \n",data3[v*meanGrid.x*(meanBlock1.bins_n_meta)+meanBlock1.bins*meanGrid.x+i]);
		}
	}


	thrust::device_vector<float> d_outputMean(n_view*mean1.bins_n_meta);
	mean1.output_block_mean = thrust::raw_pointer_cast(d_outputMean.data());

	mean1.input_bins = meanBlock1.output_block_mean;
	mean1.length = meanGrid.x;

	device::computeMeanDFPFH<<<n_view,mean1.dx>>>(mean1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::host_vector<float> h_meanHisto = d_outputMean;
	float *data4 = h_meanHisto.data();
	for(int v=0;v<1;v++)
	{
		printf("mean: ");
		for(int i=0;i<mean1.bins_n_meta;i++)
		{
			if(i>0 && i%mean1.bins_per_feature==0)
				printf("|| ");
			printf("%f ",data4[v*mean1.bins_n_meta+i]);

		}
		printf("\n");

	}

	thrust::device_vector<float> d_div(n_view*640*480);
	div1.input_dfpfh_bins = dfpfh1.output_bins;
	div1.input_mean_bins = mean1.output_block_mean;
	div1.output_div = thrust::raw_pointer_cast(d_div.data());
	device::computeDivDFPFH<<<((640*480)/div1.dx),div1.dx>>>(div1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_vector<float> d_sigma_block((640*480)/sigmaBlock1.dx);
	sigmaBlock1.input_div = div1.output_div;
	sigmaBlock1.output_div_block = thrust::raw_pointer_cast(d_sigma_block.data());
	sigmaBlock1.length = 640*480;
//	dim3 sigBlock((640*480)/sigma1.dx);
	dim3 sigGrid((640*480)/(sigmaBlock1.dx*2),1,n_view);
//	dim3 sigGrid(128,1,n_view);
	device::computeSigmaDFPFHBlock<<<sigGrid,sigmaBlock1.dx>>>(sigmaBlock1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_vector<float> d_sigma(n_view);
	sigma1.input_sig_block = sigmaBlock1.output_div_block;
	sigma1.input_mean_data = mean1.output_block_mean;
	sigma1.output_sigmas = thrust::raw_pointer_cast(d_sigma.data());
	sigma1.length = sigGrid.x;
	device::computeSigmaDFPFH<<<n_view,sigma1.dx>>>(sigma1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



//	thrust::device_vector<int> persistance_map(n_view*640*480);
//	persistance.input_bins_n_meta1

}

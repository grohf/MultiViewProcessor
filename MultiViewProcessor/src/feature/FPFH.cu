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
//		__constant__ float radius;
	}

    __device__ __forceinline__ float
    dotf43(const float4& v1,const float4& v2)
    {
      return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
    }

    __device__ __forceinline__ float4
    minusf43(const float4& v1, const float4& v2)
    {
      return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0);
    }

    __device__ __host__ __forceinline__ float4
    crossf43(const float4& v1, const float4& v2)
    {
      return make_float4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x,0);
    }

    __device__ __forceinline__ float
    lengthf43(const float4& v)
    {
      return sqrt(dotf43(v, v));
    }

    __device__ __forceinline__ float
    kldivergence(float *cur,float *mean,int bins)
    {
    	float div = 0.f;

    	for(int i=0;i<bins;i++)
    	{
    		div += (cur[i]-mean[i])*__logf(cur[i]/mean[i]);
    	}
    	return div;
    }

    __device__ __forceinline__ float
    euclideandivergence(float *cur,float *mean,int bins)
    {
    	float div = 0.f;

    	for(int i=0;i<bins;i++)
    	{
    		float d = (cur[i]-mean[i]);
    		div += d*d;
    	}
    	return sqrtf(div);
//    	return div;
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

				nbins = 8
			};

			float4 *input_pos;
			float4 *input_normals;

			float *output_bins;

			unsigned int view;
			float radius;

			__device__ __forceinline__ void
			operator () () const
			{
				__shared__ float4 shm_pos[kx*ky];
				__shared__ float4 shm_normal[kx*ky];
//				__shared__ unsigned int shm_bins[kx*ky*8];


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
				int mid_bin[8];
				for(int i=0;i<8;i++)
					mid_bin[i]=0;

				if(mid_pos.z==0)
				{
//					for(int i=0;i<8;i++)
//						mid_bin[i]=-1;

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
//						if(f2 >= 20)
//							idx += 2;

						if(dotf43(u,minusf43(pt,ps))/f2 >= 0)
							idx += 2;

						if(atan2f(dotf43(w,nt),dotf43(u,nt)) >= 0)
							idx += 4;

						mid_bin[idx]++;

//						if( !( (oy+sy) < 0 || (oy+sy) > 479 || (ox+sx) < 0 || (ox+sx) > 639) )
//						{
//							atomicAdd(&output_bins[((oy+sy)*640+(ox+sx))*8+idx],1.0f);
//						}
//							atomicAdd(&output_bins[mid_global_off*8+idx],1.0f);

					}
				}

				for(int i=0;i<8;i++)
					output_bins[mid_global_off*8+i] = ((float)mid_bin[i])/((float)point_count);


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

				int off = (sy*640+sx)*8;

//				if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
//				{
//					for(int i=0;i<8;i++)
//						printf("%f | ",inline_bins[off+i]);
//
//					printf("\n");
//				}

				float sum = 0.0;
				for(int i=0;i<8;i++)
					sum += inline_bins[off+i];

				for(int i=0;i<8;i++)
					inline_bins[off+i] /= sum;



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

			unsigned int view;
			float radius;


//		    __device__ __forceinline__ float4
//		    minusf43(const float4& v1, const float4& v2) const
//		    {
//		      return make_float4(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z,0);
//		    }
//
//		    __device__ __forceinline__ float
//		    dotf43(const float4& v1,const float4& v2) const
//		    {
//		      return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
//		    }
//
//		    __device__ __forceinline__ float
//		    lengthf43(const float4& v) const
//		    {
//		      return sqrt(dotf43(v, v));
//		    }

		    __device__ __forceinline__ void
			operator () () const
			{
				const int oy = blockIdx.y*blockDim.y-kyr;
				const int ox = blockIdx.x*blockDim.x-kxr;

				const unsigned int gx = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int gy = threadIdx.y + blockIdx.y * blockDim.y;

				float mid_bins[8];
				for(int i=0;i<8;i++)
					mid_bins[i] = 0;

				float4 mid_pos = input_pos[view*640*480+gy*640+gx];
				if(mid_pos.z==0)
				{
					for(int i=0;i<8;i++)
					{
						fpfh_output[(gy*640+gx)*8+i] = 0.f;
					}
					return;
				}
				unsigned int point_count = 1;
				for(int sy=(oy<0)?(-oy):0;sy<kl;sy++)
				{
					for(int sx=(ox<0)?(-ox):0;sx<kl ;sx++)
					{
						if(sy==kxr && sx==kyr)
							continue;

						float4 cur_pos = input_pos[view*640*480+(oy+sy+threadIdx.y)*640+(ox+sx+threadIdx.x)];
						if(cur_pos.z==0)
							continue;

						float length = lengthf43(minusf43(mid_pos,cur_pos));
						if(length>radius)
							continue;

						float weight = 1.f/(1.f+length);
//						if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
//						{
//								printf("(%d/%d) ->mid_pos: %f %f %f | cur_pos: %f %f %f | weight: %f \n",sx,sy,mid_pos.x,mid_pos.y,mid_pos.z,
//										cur_pos.x,cur_pos.y,cur_pos.z,weight);
//						}

						for(int i=0;i<8;i++)
						{
							mid_bins[i] += weight * spfh_input[((oy+sy+threadIdx.y)*640+(ox+sx+threadIdx.x))*8+i];
							if(spfh_input[((oy+sy+threadIdx.y)*640+(ox+sx+threadIdx.x))*8+i]<0)
								printf("OOOOOOOOOoooooooooooooooooooOOOOOOOOOOOOOOOOOOOOoooooooooooo!!! \n");
						}

						point_count++;
					}
				}

				for(int i=0;i<8;i++)
				{
					mid_bins[i] = spfh_input[(gy*640+gx)*8+i] + mid_bins[i]/((float) point_count);
				}

				float sum = 0.f;
				for(int i=0;i<8;i++)
				{
					sum+=mid_bins[i];
				}

				for(int i=0;i<8;i++)
				{
					fpfh_output[(gy*640+gx)*8+i] = mid_bins[i]/sum;
				}

//				if(blockIdx.x==10 && blockIdx.y==12 && threadIdx.x==10 && threadIdx.y==10)
//				{
//					printf("sum: %f \n",sum);
//					for(int i=0;i<8;i++)
//					{
//						printf("%f | ",mid_bins[i]/sum);
//
//					}
//					printf("\n");
//				}
			}
		};
		__global__ void computeFPFH(const FPFHEstimator fpfhe){ fpfhe(); }


		template <unsigned int blockSize>
		struct Mean8BinHistogramEstimator
		{
			float *input_fpfh;
			float *output_meanHisto;
			unsigned int n;

		    __device__ __forceinline__ void
			operator () () const
			{
		    	extern __shared__ float shm[];

		        unsigned int tid = threadIdx.x;
		        unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
		        unsigned int gridSize = blockSize*2*gridDim.x;

		    	float sum = 0.f;
		    	while(i<n)
		    	{
		    		sum += input_fpfh[i];

//		    		if(threadIdx.x==0)
//		    			printf("sum: %f inp: %f i: %d \n",sum,input_fpfh[i],i);

		            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		            if ( (i + blockSize) < n)
		            	sum += input_fpfh[i+blockSize];

		            i += gridSize;
		    	}

		        // each thread puts its local sum into shared memory
		    	shm[tid] = sum;
		        __syncthreads();

		        // do reduction in shared mem
		        if (blockSize >= 1024)
		        {
		            if (tid < 512)
		            {
		                shm[tid] = sum = sum + shm[tid + 512];
		            }

		            __syncthreads();
		        }


		        if (blockSize >= 512)
		        {
		            if (tid < 256)
		            {
		                shm[tid] = sum = sum + shm[tid + 256];
		            }

		            __syncthreads();
		        }

		        if (blockSize >= 256)
		        {
		            if (tid < 128)
		            {
		            	shm[tid] = sum = sum + shm[tid + 128];
		            }

		            __syncthreads();
		        }

		        if (blockSize >= 128)
		        {
		            if (tid <  64)
		            {
		            	shm[tid] = sum = sum + shm[tid +  64];
		            }

		            __syncthreads();
		        }

				if (tid < 32)
				{
					 // now that we are using warp-synchronous programming (below)
					 // we need to declare our shared memory volatile so that the compiler
					 // doesn't reorder stores to it and induce incorrect behavior.
					 volatile float *smem = shm;

					 if (blockSize >=  64)
					 {
						 smem[tid] = sum = sum + smem[tid + 32];
					 }

					 if (blockSize >=  32)
					 {
						 smem[tid] = sum = sum + smem[tid + 16];
					 }

					 if (blockSize >=  16)
					 {
						 smem[tid] = sum = sum + smem[tid +  8];
					 }

					if (blockSize >=   8)
					{
						smem[tid] += smem[tid +  4];
					}

					if (blockSize >=   4)
					{
						smem[tid] += smem[tid +  2];
					}

					if (blockSize >=   2)
					{
						smem[tid] += smem[tid +  1];
					}

				}

			if(tid<8)
				output_meanHisto[tid] = sum/shm[0];
			}

		};
		__global__ void estimateMean8BitHistogram(const Mean8BinHistogramEstimator<1024> mhe){ mhe(); }


		struct DivHistogramEstimator
		{
			float *input_fpfh;
			float *input_mean;

			float *output_div;

		    __device__ __forceinline__ void
			operator () () const
			{
				const unsigned int gx = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int gy = threadIdx.y + blockIdx.y * blockDim.y;

//				output_div[gy*640+gx] = kldivergence(&input_fpfh[(gy*640+gx)*8],&input_mean[0],8);
				output_div[gy*640+gx] = euclideandivergence(&input_fpfh[(gy*640+gx)*8],&input_mean[0],8);

			}
		};
		__global__ void computeDivHistogram(const DivHistogramEstimator de){ de(); }



		template <unsigned int blockSize>
		struct SigmaEstimator
		{
			float *input_div;
			float4 *input_pos;
			float *output_sigma;

			unsigned int n;
			unsigned int view;

			 __device__ __forceinline__ void
			operator () () const
			{
				extern __shared__ float shm[];

//				unsigned int *shm_count = &shm[blockDim.x*blockDim.y];

				unsigned int tid = threadIdx.x;
				unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
				unsigned int gridSize = blockSize*2*gridDim.x;

				float sum = 0.f;
				float count = 0.f;
				while(i<n)
				{
					float tmp = input_div[i];
					sum += tmp*tmp;
					count += (input_pos[view*640*480+i].z==0)?0:1;

					if(threadIdx.x==0)
						printf("sum: %f inp: %f i: %d \n",sum,input_div[i],i);

					// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
					if ( (i + blockSize) < n)
					{
						tmp = input_div[i+blockSize];
						sum += tmp*tmp;
						count += (input_pos[view*640*480+i+blockSize].z==0)?0:1;
					}
					i += gridSize;
				}

				// each thread puts its local sum into shared memory
				shm[tid] = sum;
				shm[blockDim.x*blockDim.y+tid] = count;
				__syncthreads();

				// do reduction in shared mem
				if (blockSize >= 1024)
				{
					if (tid < 512)
					{
						shm[tid] = sum = sum + shm[tid + 512];
						shm[blockDim.x*blockDim.y+tid] = count = count + shm[blockDim.x*blockDim.y+tid + 512];
					}

					__syncthreads();
				}


				if (blockSize >= 512)
				{
					if (tid < 256)
					{
						shm[tid] = sum = sum + shm[tid + 256];
						shm[blockDim.x*blockDim.y+tid] = count = count + shm[blockDim.x*blockDim.y+tid + 256];
					}

					__syncthreads();
				}

				if (blockSize >= 256)
				{
					if (tid < 128)
					{
						shm[tid] = sum = sum + shm[tid + 128];
						shm[blockDim.x*blockDim.y+tid] = count = count + shm[blockDim.x*blockDim.y+tid + 128];
					}

					__syncthreads();
				}

				if (blockSize >= 128)
				{
					if (tid <  64)
					{
						shm[tid] = sum = sum + shm[tid +  64];
						shm[blockDim.x*blockDim.y+tid] = count = count + shm[blockDim.x*blockDim.y+tid + 64];
					}

					__syncthreads();
				}

				if (tid < 32)
				{
					 // now that we are using warp-synchronous programming (below)
					 // we need to declare our shared memory volatile so that the compiler
					 // doesn't reorder stores to it and induce incorrect behavior.
					 volatile float *smem = shm;

					 if (blockSize >=  64)
					 {
						 smem[tid] = sum = sum + smem[tid + 32];
						 smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 32];
					 }

					 if (blockSize >=  32)
					 {
						 smem[tid] = sum = sum + smem[tid + 16];
						 smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 16];
					 }

					 if (blockSize >=  16)
					 {
						 smem[tid] = sum = sum + smem[tid +  8];
						 smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 8];
					 }

					if (blockSize >=   8)
					{
						smem[tid] = sum = sum + smem[tid +  4];
						smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 4];
					}

					if (blockSize >=   4)
					{
						smem[tid] = sum = sum + smem[tid +  2];
						smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 2];
					}

					if (blockSize >=   2)
					{
						smem[tid] = sum = sum + smem[tid +  1];
						smem[blockDim.x*blockDim.y+tid] = count = count + smem[blockDim.x*blockDim.y+tid + 1];
					}
				}

				if(tid==0)
				{
					float sigma = sqrt(sum/(count-1));
					printf("sum: %f | count: %f | sigma: %f \n",sum,count,sigma);
					output_sigma[tid] = sigma;
				}
			}

		};
		__global__ void computeSigma(const SigmaEstimator<1024> se){ se(); }


		struct PFPFHEstimator
		{

			float4 *input_pos;
			unsigned int view;
			float beta;


			float *input_div1;
			float *input_sigma1;

//			unsigned int *output_persistence_map;

			uchar4 *output_persistence_map;

		    __device__ __forceinline__ void
			operator () () const
			{
				const unsigned int gx = threadIdx.x + blockIdx.x * blockDim.x;
				const unsigned int gy = threadIdx.y + blockIdx.y * blockDim.y;


				float z = input_pos[view*640*480+gy*640+gx].z;

				if( z==0 || input_div1[gy*640+gx]< beta*input_sigma1[0] )
				{
					output_persistence_map[view*640*480+gy*640+gx] = make_uchar4(z/30.,z/30.,z/30.,0);
					return;
				}
				output_persistence_map[view*640*480+gy*640+gx] = make_uchar4(input_div1[gy*640+gx]*200,0,0,0);

			}

		};
		__global__ void computePersistenceFPFH(const PFPFHEstimator pfpfh){ pfpfh(); }


}

device::SPFHEstimator spfhEstimator1;
device::SPFHEstimator spfhEstimator2;

//device::SPFHNormalizer spfhNormailizer1;
//device::SPFHNormalizer spfhNormailizer2;

device::FPFHEstimator fpfhEstimator1;
device::FPFHEstimator fpfhEstimator2;

device::Mean8BinHistogramEstimator<1024> meanEstimator1;

device::DivHistogramEstimator divEstimator1;

device::SigmaEstimator<1024> sigEstimator1;


device::PFPFHEstimator pfpfhEstimator;

void FPFH::init()
{
	spfhEstimator1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	spfhEstimator1.input_normals = (float4 *)getInputDataPointer(PointNormal);
	spfhEstimator1.output_bins = (float *)getTargetDataPointer(SPFHistogram1);

//	spfhEstimator2.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
//	spfhEstimator2.input_normals = (float4 *)getInputDataPointer(PointNormal);
//	spfhEstimator2.output_bins = (float *)getTargetDataPointer(SPFHistogram2);


	fpfhEstimator1.spfh_input = (float *)getTargetDataPointer(SPFHistogram1);
	fpfhEstimator1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	fpfhEstimator1.fpfh_output = (float *)getTargetDataPointer(FPFHistogram1);

//	fpfhEstimator2.spfh_input = (float *)getTargetDataPointer(SPFHistogram2);
//	fpfhEstimator2.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
//	fpfhEstimator2.fpfh_output = (float *)getTargetDataPointer(FPFHistogram2);


	meanEstimator1.input_fpfh = (float *)getTargetDataPointer(FPFHistogram1);
	meanEstimator1.output_meanHisto = (float *)getTargetDataPointer(MeanHistogram1);

	divEstimator1.input_fpfh = (float *)getTargetDataPointer(FPFHistogram1);
	divEstimator1.input_mean = (float *)getTargetDataPointer(MeanHistogram1);
	divEstimator1.output_div = (float *)getTargetDataPointer(DivHistogram1);


	sigEstimator1.input_div = (float *)getTargetDataPointer(DivHistogram1);
	sigEstimator1.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	sigEstimator1.output_sigma = (float *)getTargetDataPointer(Sigma1);


	pfpfhEstimator.input_div1 = (float *)getTargetDataPointer(DivHistogram1);
	pfpfhEstimator.input_pos = (float4 *)getInputDataPointer(PointCoordinates);
	pfpfhEstimator.input_sigma1 = (float *)getTargetDataPointer(Sigma1);
	pfpfhEstimator.output_persistence_map = (uchar4 *)getTargetDataPointer(TestMap);

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y);


}

void FPFH::execute()
{

	float radii[] = {15.0f,20.f,25.f};

//	device::SPFHEstimator *cur_estimator = &spfhEstimator1;

	for(int v=0;v<1;v++)
	{
		spfhEstimator1.view = v;
//		spfhEstimator2.view = v;

//		radius = radii[0];
//		cudaMemcpy(&device::constant::radius,&radius,sizeof(float),cudaMemcpyHostToDevice);

		spfhEstimator1.radius=radii[0];
		device::computeSPFH<<<grid,block>>>(spfhEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

//		device::normalizeSPFH<<<grid,block>>>(spfhNormailizer1);
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());

		fpfhEstimator1.view = v;
		fpfhEstimator1.radius = radii[0];
		device::computeFPFH<<<grid,block>>>(fpfhEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());


//		float *h_testdata = (float *)malloc(640*480*8*sizeof(float));
//		for(int i=0;i<640*480;i++)
//		{
//			for(int b=0;b<8;b++)
//				h_testdata[i*8+b]=b;
//		}
//
//		checkCudaErrors(cudaMemcpy(meanEstimator1.input_fpfh,h_testdata,640*480*8*sizeof(float),cudaMemcpyHostToDevice));
//		checkCudaErrors(cudaGetLastError());
//		checkCudaErrors(cudaDeviceSynchronize());

		dim3 meanblock(32*32);
		meanEstimator1.n = 640*480*8;
		device::estimateMean8BitHistogram<<<1,meanblock,32*32*sizeof(float)>>>(meanEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		float *h_mean_testdata = (float *)malloc(8*sizeof(float));
		checkCudaErrors(cudaMemcpy(h_mean_testdata,meanEstimator1.output_meanHisto,8*sizeof(float),cudaMemcpyDeviceToHost));

		for(int b=0;b<8;b++)
			printf("%f |",h_mean_testdata[b]);

		device::computeDivHistogram<<<grid,block>>>(divEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

//		float *h_f_div_data = (float *)malloc(640*480*sizeof(float));
//		checkCudaErrors(cudaMemcpy(h_f_div_data,divEstimator1.output_div,640*480*sizeof(float),cudaMemcpyDeviceToHost));
//
//		for(int i=640*200;i<640*210;i++)
//		{
//			printf(" %f ",h_f_div_data[i]);
//			if(i%40==0) printf("\n");
//		}



		sigEstimator1.n = 640*480;
		sigEstimator1.view = v;
		device::computeSigma<<<1,meanblock,32*32*2*sizeof(float)>>>(sigEstimator1);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		pfpfhEstimator.view = v;
		pfpfhEstimator.beta = 1.6f;
		device::computePersistenceFPFH<<<grid,block>>>(pfpfhEstimator);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		uchar4 *h_uc4_persistence_data = (uchar4 *)malloc(640*480*sizeof(uchar4));
		checkCudaErrors(cudaMemcpy(h_uc4_persistence_data,pfpfhEstimator.output_persistence_map,640*480*sizeof(uchar4),cudaMemcpyDeviceToHost));

		char path[50];
		sprintf(path,"/home/avo/pcds/persitence_map%d.ppm",0);
		sdkSavePPM4ub(path,(unsigned char*)h_uc4_persistence_data,640,480);


		printf("\n");

		printf("fpfh done... \n");

		for(int ri=0;ri<1;ri++)
		{

		}

	}

//	sp

}



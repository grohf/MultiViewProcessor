/*
 * RigidBodyTransformationAdvancedEstimatior.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: avo
 */

#include "RigidBodyTransformationAdvancedEstimatior.h"

#include <curand.h>

#include <helper_cuda.h>
#include <helper_image.h>

#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/random.h>

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

	struct RigidBodyTransformationBaseKernel : public FeatureBaseKernel
	{
		enum
		{
			corresp_list_length = 8,
		};
	};


	struct CorrespondenceListEstimator : public RigidBodyTransformationBaseKernel
	{
		enum
		{
			local_points_per_block = 32, //local_points_per_block
			global_points_per_sweep = 32, //global_points_per_sweep
			dx = local_points_per_block*global_points_per_sweep,
			minDelta = 10000,
		};


		float *input_feature_histo;
		int *input_idxList;
		unsigned int *input_idxLength;
		float *input_rndList;

		unsigned int *output_s_idxList;

		float *output_corresp_prob;
		unsigned int *output_corresp_idx;

		unsigned int n_view;

		__device__ __forceinline__ void compareDiv(
			float  &val1,
			float  &val2,
			unsigned int  &idx1,
			unsigned int  &idx2,
			unsigned int  dir
		) const
		{
		    float tmp_val;
		    unsigned int tmp_idx;

		    if ((val1 > val2) == dir)
		    {
		    	tmp_val = val1;
		        val1 = val2;
		        val2 = tmp_val;
		        tmp_idx = idx1;
		        idx1 = idx2;
		        idx2 = tmp_idx;
		    }
		}

		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float shm_local_histo[local_points_per_block*bins];
			__shared__ unsigned int shm_local_idx[local_points_per_block];

			__shared__ float shm_global_histo[global_points_per_sweep*bins];
			__shared__ unsigned int shm_global_idx[global_points_per_sweep];

			//TODO: extra buffer for list
			__shared__ float shm_dist_buffer[local_points_per_block*global_points_per_sweep*2];
			__shared__ unsigned int shm_idx_buffer[local_points_per_block*global_points_per_sweep*2];

//			__shared__ float shm_dist_sorted[local_points_per_block*corresp_list_length];
//			__shared__ unsigned int shm_idx_sorted[local_points_per_block*corresp_list_length];

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			unsigned int tid = threadIdx.x;

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

//			if(blockIdx.x==0 && tid==0)
//			{
//				printf("(%d) - > src: %d target: %d \n",blockIdx.z,view_src,view_target);
//			}

			if(threadIdx.x<local_points_per_block)
			{
				float f = input_rndList[(blockIdx.z*gridDim.x+blockIdx.x)*local_points_per_block+threadIdx.x];
				unsigned int begin = (view_src>0)?input_idxLength[view_src-1]:0;
				unsigned int length = input_idxLength[view_src] - begin;
				unsigned int idx = input_idxList[begin+(unsigned int)(f*length)];

//				if(tid==0)
//					printf("begin: %d | length: %d \n",begin,length);

				idx -= (view_src*640*480);
//				unsigned int oy = idx/640;
//				unsigned int ox = idx - oy*640;
//				printf("(%d) %f -> %d -> %d / %d \n",tid,f,idx,oy,ox);

				shm_local_idx[threadIdx.x] = idx;
				output_s_idxList[(blockIdx.z*gridDim.x+blockIdx.x)*local_points_per_block+threadIdx.x] = idx;

			}
			__syncthreads();

			for(int i=threadIdx.x; i<local_points_per_block*bins; i+=blockDim.x)
			{
				unsigned int b = i/local_points_per_block;
				unsigned int sl = i-b*local_points_per_block;
				shm_local_histo[b*local_points_per_block + sl] = input_feature_histo[view_src*640*480*bins_n_meta+b*640*480+shm_local_idx[sl]];
			}

			//TODO: delete
			__syncthreads();
/*			if(blockIdx.x==0 && tid==7)
			{
				for(int p=0;p<local_points_per_block;p++)
				{
					float sum = 0.f;
					printf("L: (%d -> %d): ",p,shm_local_idx[p]);
					for(int f=0;f<features;f++)
					{
						sum = 0.f;
						for(int b=0;b<bins_per_feature;b++)
						{
							float tmp = shm_local_histo[(f*bins_per_feature+b)*local_points_per_block+p];
							printf("%f ",tmp);
							sum += tmp;
						}
						printf(" (sum: %f) || ",sum);
					}
					printf("\n");
				}
			}
*/

			unsigned int target_begin = input_idxLength[view_target-1];
			unsigned int target_length = input_idxLength[view_target]-target_begin;
			if(blockIdx.x==0 && tid==0)
				printf("target length: %d \n",target_length);
			//TODO: handle rest points
			for(int l=0;l<target_length/global_points_per_sweep;l++)
//			for(int l=0;l<2;l++)
			{

				//Get the index of the k global points;
				for(int i=threadIdx.x;i<global_points_per_sweep;i+=blockDim.x)
				{
					shm_global_idx[i] = input_idxList[target_begin+l*global_points_per_sweep+i]-view_target*640*480;
				}
				__syncthreads();

				for(int i=threadIdx.x; i<global_points_per_sweep*bins; i+=blockDim.x)
				{
					unsigned int b = i/global_points_per_sweep;
					unsigned int sl = i - b*global_points_per_sweep;

					shm_global_histo[b*global_points_per_sweep+sl] = input_feature_histo[view_target*640*480*bins_n_meta+b*640*480+shm_global_idx[sl]];
				}
				__syncthreads();

				for(int i=threadIdx.x; i<global_points_per_sweep*local_points_per_block; i+=blockDim.x)
				{
					unsigned int wl = i/local_points_per_block;
					shm_idx_buffer[local_points_per_block*global_points_per_sweep+i] = shm_global_idx[wl];
				}
				__syncthreads();

/*
				__syncthreads();
				if(blockIdx.x==0 && tid==7)
				{
					for(int p=0;p<global_points_per_sweep;p++)
					{
						float sum = 0.f;
						printf("(G: %d -> %d): ",p,shm_global_idx[p]);
						for(int f=0;f<features;f++)
						{
							sum = 0.f;
							for(int b=0;b<bins_per_feature;b++)
							{
								float tmp = shm_global_histo[(f*bins_per_feature+b)*global_points_per_sweep+p];
								printf("%f ",tmp);
								sum += tmp;
							}
							printf(" (sum: %f) || ",sum);
						}
						printf("\n");
					}
				}
*/

				unsigned int lid = tid/local_points_per_block;
				unsigned int ltid = tid - lid*local_points_per_block;
				float div = 0.f;
//				div = klEuclideanDivergence(&(shm_local_histo[ltid]),&(shm_global_histo[lid]),features,bins_per_feature,local_points_per_block,global_points_per_sweep);
//				div = chiSquaredDivergence(&(shm_local_histo[ltid]),&(shm_global_histo[lid]),features,bins_per_feature,local_points_per_block,global_points_per_sweep);
				div = chiSquaredEuclideanDivergence(&(shm_local_histo[ltid]),&(shm_global_histo[lid]),features,bins_per_feature,local_points_per_block,global_points_per_sweep);
				shm_dist_buffer[local_points_per_block*global_points_per_sweep+lid*local_points_per_block+ltid] = div;

//				__syncthreads();


				//Bitonic sort Buffer
				for(int size = 2; size < global_points_per_sweep; size <<= 1)
				{
					unsigned int dir = 1 ^ ((lid & (size/2)) != 0);
					for(int stride = size/2; stride > 0; stride >>= 1)
					{
						__syncthreads();
						if(lid<global_points_per_sweep/2)
						{
							unsigned int pos = (2 * lid - (lid & (stride - 1))) * 32 + ltid + local_points_per_block*global_points_per_sweep;
							compareDiv(shm_dist_buffer[pos], shm_dist_buffer[pos+stride*32], shm_idx_buffer[pos], shm_idx_buffer[pos+stride*32], dir);
						}
					}
				}

				for(int stride = global_points_per_sweep/2; stride > 0; stride >>= 1)
				{
					__syncthreads();
					if(lid<global_points_per_sweep/2)
					{
						unsigned int pos = (2 * lid - (lid & (stride - 1))) * 32 + ltid + local_points_per_block*global_points_per_sweep;
						compareDiv(shm_dist_buffer[pos], shm_dist_buffer[pos+stride*32], shm_idx_buffer[pos], shm_idx_buffer[pos+stride*32], 1);
					}
				}
//				__syncthreads();
//				if(blockIdx.x==0 && tid==7)
//				{
//					for(int j=0;j<local_points_per_block;j++)
//					{
//						bool fine = true;
//						float old = 0.f;
//						printf("(%d): ",j);
//						for(int g=0;g<global_points_per_sweep;g++)
//						{
//							fine = (old <= shm_dist_buffer[g*local_points_per_block+j]);
//							printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//							old = shm_dist_buffer[g*local_points_per_block+j];
//							if(!fine)
//								printf("! (%d/%d) ! ",j,g);
//						}
//						printf(" \n");
//						fine = true;
//					}
//				}

				__syncthreads();
				if(l==0)
				{
					for(int i=threadIdx.x; i<local_points_per_block*global_points_per_sweep; i+=blockDim.x)
					{
						shm_dist_buffer[i] = shm_dist_buffer[local_points_per_block*global_points_per_sweep+i];
						shm_idx_buffer[i] = shm_idx_buffer[local_points_per_block*global_points_per_sweep+i];
					}
				}
				else
				{
					// Merge 2 sorted lists together
					unsigned int stride = global_points_per_sweep;
					unsigned int offset = lid & (stride - 1);
//
					__syncthreads();

					unsigned int pos = (2 * lid - (lid & (stride - 1)))*32 + ltid;
					compareDiv(shm_dist_buffer[pos], shm_dist_buffer[pos+stride*32], shm_idx_buffer[pos], shm_idx_buffer[pos+stride*32],1);

					for(stride >>= 1; stride > 0; stride >>=1)
					{
						__syncthreads();
						unsigned int pos = (2 * lid - (lid & (stride - 1)))*32 + ltid;
						if(offset >= stride)
						{
							compareDiv(shm_dist_buffer[pos-stride*32], shm_dist_buffer[pos], shm_idx_buffer[pos-stride*32], shm_idx_buffer[pos],1);
						}

					}
				}

//				__syncthreads();
//				if(blockIdx.x==0 && tid==7)
//				{
//					if(l%100==0)
//						for(int j=0;j<1;j++)
//						{
//							bool fine = true;
//							float old = 0.f;
//							printf("(%d/%d): ",l,j);
//							for(int g=0;g<global_points_per_sweep;g++)
//							{
//								fine = (old <= shm_dist_buffer[g*local_points_per_block+j]);
//								printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//								old = shm_dist_buffer[g*local_points_per_block+j];
//								if(!fine)
//									printf("! (%d/%d) ! ",j,g);
//							}
//							printf(" \n");
//							for(int g=0;g<global_points_per_sweep;g++)
//							{
//								printf(" %d ",shm_idx_buffer[g*local_points_per_block+j]);
//							}
//							printf(" \n");
//							fine = true;
//						}
//				}

			}
			__syncthreads();


//			__syncthreads();
//			if(blockIdx.x==0 && tid==7)
//			{
//					for(int j=0;j<8;j++)
//					{
//						printf("(%d): ",j);
//						for(int g=0;g<global_points_per_sweep;g++)
//						{
//							printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//
//						}
//						printf(" \n");
//						for(int g=0;g<global_points_per_sweep;g++)
//						{
//							printf(" %d ",shm_idx_buffer[g*local_points_per_block+j]);
//						}
//						printf(" \n");
//					}
//			}

//			if(blockIdx.x==0 && tid==7)
//			{
//					for(int j=0;j<1;j++)
//					{
//						printf("(error: %d): ",j);
//						for(int g=0;g<corresp_list_length;g++)
//						{
//							printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//
//						}
//						printf("\n");
//					}
//					printf("---------------------------- \n");
//			}

//			for(int i=threadIdx.x; i<local_points_per_block*corresp_list_length; i+=blockDim.x)
//			{
//				unsigned int l = i/local_points_per_block;
//				unsigned int p = i-l*local_points_per_block;
//
//				float errorDelta = shm_dist_buffer[p*local_points_per_block+l];
//
//				if(errorDelta < 1.f/minDelta)
//					errorDelta = 1.f/minDelta;
//
//				shm_dist_buffer[p*local_points_per_block+l] = 1.f/(errorDelta);
//			}
//
//			__syncthreads();
//
//			if(blockIdx.x==0 && tid==7)
//			{
//					for(int j=0;j<1;j++)
//					{
//						printf("(prob: %d): ",j);
//						for(int g=0;g<corresp_list_length;g++)
//						{
//							printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//
//						}
//						printf("\n");
//					}
//					printf("---------------------------- \n");
//			}

			if(tid<local_points_per_block)
			{
				float cumSum = 0.f;
				for(int i=0;i<corresp_list_length;i++)
				{
					float tmp = shm_dist_buffer[i*local_points_per_block+tid];
					tmp = (tmp>0)?1.f/(tmp*tmp):1.f/minDelta;
					shm_dist_buffer[i*local_points_per_block+tid] = cumSum += tmp;
				}
			}

			__syncthreads();

//			if(blockIdx.x==0 && tid==7)
//			{
//					for(int j=0;j<1;j++)
//					{
//						printf("(cum: %d): ",j);
//						for(int g=0;g<corresp_list_length;g++)
//						{
//							printf(" %f ",shm_dist_buffer[g*local_points_per_block+j]);
//
//						}
//						printf("\n");
//					}
//					printf("---------------------------- \n");
//			}


			for(int i=threadIdx.x; i<local_points_per_block*corresp_list_length; i+=blockDim.x)
			{
				unsigned int l = i/local_points_per_block;
				unsigned int p = i-l*local_points_per_block;

				output_corresp_prob[(blockIdx.z*gridDim.x+blockIdx.x+p)*local_points_per_block + l] = shm_dist_buffer[p*local_points_per_block+l]/shm_dist_buffer[(corresp_list_length-1)*local_points_per_block+l];
				output_corresp_idx[(blockIdx.z*gridDim.x+blockIdx.x+p)*local_points_per_block + l] = shm_idx_buffer[p*local_points_per_block+l];
			}

		}
	};
	__global__ void estimateBestCorrespondences(const CorrespondenceListEstimator cle){ cle(); }


	struct TransformationBaseKernel : public RigidBodyTransformationBaseKernel
	{
		enum
		{
			n_corresp = 4,
			n_combinations = (n_corresp*(n_corresp-1))/2,

			groups_per_warp = WARP_SIZE/n_corresp,
			group_length = WARP_SIZE/groups_per_warp,
		};
	};


	struct CombinationErrorListEstimator : public TransformationBaseKernel
	{
		enum
		{
			dx = 1024,

			n_transformations = dx/n_corresp,
		};


		float4 			*input_pos;
		unsigned int 	*input_src_idx_corresp;
		unsigned int 	*input_target_idx_corresp;
		float 			*input_target_prob_corresp;

		float 			*input_src_rnd;
		float			*input_target_rnd;

		float			*output_combinationError;
		unsigned int	*output_combinationIdx_src;
		unsigned int	*output_combinationIdx_target;

		unsigned int s_length;
		unsigned int n_view;
		unsigned int combinationListOffset;

		__device__ __forceinline__ void
		operator () () const
		{


			__shared__ float combLength[n_transformations*n_combinations];
			__shared__ float points[dx*6];

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			unsigned int tid = threadIdx.x;

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


			unsigned int idx = (unsigned int)(input_src_rnd[(blockIdx.z*gridDim.x+blockIdx.x)*dx+tid]*s_length);
			unsigned int pidx = input_src_idx_corresp[idx];
			output_combinationIdx_src[(blockIdx.z*gridDim.x+blockIdx.x)*dx+tid] = pidx;
			float4 tmp = input_pos[view_src*640*480+pidx];

			points[dx*0 + tid] = tmp.x;
			points[dx*1 + tid] = tmp.y;
			points[dx*2 + tid] = tmp.z;


			float rnd_t = input_target_rnd[(blockIdx.z*gridDim.x+blockIdx.x)*dx+tid];
			unsigned int g = 0;
			while( rnd_t > input_target_prob_corresp[view_target*s_length*corresp_list_length + g*corresp_list_length + idx] )
			{
				g++;
			}

			pidx = input_target_idx_corresp[view_target*s_length*corresp_list_length + g*corresp_list_length + idx];
			output_combinationIdx_target[(blockIdx.z*gridDim.x+blockIdx.x)*dx+tid] = pidx;

			tmp = input_pos[view_target*640*480+pidx];
			points[dx*3 + tid] = tmp.x;
			points[dx*4 + tid] = tmp.y;
			points[dx*5 + tid] = tmp.z;

			__syncthreads();


			unsigned int wid = tid/WARP_SIZE;
			unsigned int wtid = tid - wid * WARP_SIZE;
			unsigned int gid = wtid/group_length;
			unsigned int gtid = wtid - gid * group_length;

			for(int i=n_corresp-1;i>0;i++)
			{
				if(gtid<i)
				{

					//to be continued...
				}
			}

		}
	};

	struct TransformationMatrixEstimator : public TransformationBaseKernel
	{
		enum
		{
			threads = 512,
//			n_corresp = 4,
//			n_combinations = (n_corresp*(n_corresp-1))/2,
			n_matrices = threads/n_corresp,

			groups_per_warp = WARP_SIZE/n_corresp,
			group_length = WARP_SIZE/groups_per_warp,
		};

		float4 			*input_pos;
		unsigned int 	*input_src_idx_corresp;
		unsigned int 	*input_target_idx_corresp;
		float 			*input_target_prob_corresp;

		float 			*input_src_rnd;
		float			*input_target_rnd;

		float			*output_transformationMatrices;

		unsigned int s_length;
		unsigned int n_view;

		__device__ __forceinline__
		unsigned int hash(unsigned int a) const
		{
			a = (a+0x7ed55d16) + (a<<12);
			a = (a^0xc761c23c) ^ (a>>19);
			a = (a+0x165667b1) + (a<<5);
			a = (a+0xd3a2646c) ^ (a<<9);
			a = (a+0xfd7046c5) + (a<<3);
			a = (a^0xb55a4f09) ^ (a>>16);
			return a;
		}


		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float points[threads*6*2];
			__shared__ float combLength[n_matrices*n_combinations];
			__shared__ float buffer[threads];

			__shared__ float centroids[n_matrices*6];
			__shared__ float matricesH[n_matrices*9];

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			unsigned int tid = threadIdx.x;

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


			unsigned int seed = hash(tid);
			 // seed a random number generator
			thrust::default_random_engine rng(seed);

			// create a mapping from random numbers to [0,1)
			thrust::uniform_real_distribution<float> u01(0,1);

			float x = u01(rng);
			float y = u01(rng);

			if(blockIdx.x==0 && tid==0)
				printf("%f %f \n",x,y);




/*
			unsigned int idx = (unsigned int)(input_src_rnd[blockIdx.x*threads+tid]*s_length);
			unsigned int pidx = input_src_idx_corresp[idx];
			float4 tmp = input_pos[view_src*640*480+pidx];

			points[threads*0 + tid] = tmp.x;
			points[threads*1 + tid] = tmp.y;
			points[threads*2 + tid] = tmp.z;

			pidx = input_target_idx_corresp[view_target*s_length*corresp_list_length + (unsigned int)(input_target_rnd[blockIdx.x*threads+tid]*corresp_list_length) + idx];
			tmp = input_pos[view_target*640*480+pidx];
			points[threads*3 + tid] = tmp.x;
			points[threads*4 + tid] = tmp.y;
			points[threads*5 + tid] = tmp.z;


			unsigned int wid = tid/WARP_SIZE;
			unsigned int wtid = tid - wid * WARP_SIZE;
			unsigned int gid = wtid/group_length;
			unsigned int gtid = wtid - gid * group_length;




			for(int d=0;d<6;d++)
			{

				buffer[threadIdx.x] = points[d*blockDim.x+threadIdx.x];
				__syncthreads();

				volatile float *warpLine = &buffer[wid*WARP_SIZE];

				if(gtid < 16){
					unsigned int posb = gid * group_length;
					if(groups_per_warp<=1) 	warpLine[posb + gtid] += warpLine[posb + gtid + 16];
					if(groups_per_warp<=2) 	warpLine[posb + gtid] += warpLine[posb + gtid + 8];
					if(groups_per_warp<=4) 	warpLine[posb + gtid] += warpLine[posb + gtid + 4];
					if(groups_per_warp<=8) 	warpLine[posb + gtid] += warpLine[posb + gtid + 2];
					if(groups_per_warp<=16) warpLine[posb + gtid] += warpLine[posb + gtid + 1];

					if(gtid<groups_per_warp)
					{
						centroids[d*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length]/group_length;
					}
				}
			}
			__syncthreads();

			points[threads*0 + threadIdx.x] -= centroids[0*n_matrices+wid*groups_per_warp+gid];
			points[threads*1 + threadIdx.x] -= centroids[1*n_matrices+wid*groups_per_warp+gid];
			points[threads*2 + threadIdx.x] -= centroids[2*n_matrices+wid*groups_per_warp+gid];

			points[threads*3 + threadIdx.x] -= centroids[3*n_matrices+wid*groups_per_warp+gid];
			points[threads*4 + threadIdx.x] -= centroids[4*n_matrices+wid*groups_per_warp+gid];
			points[threads*5 + threadIdx.x] -= centroids[5*n_matrices+wid*groups_per_warp+gid];


			__syncthreads();
			for(int sd=0;sd<3;sd++)
			{
				for(int td=0;td<3;td++)
				{

					buffer[threadIdx.x] = points[sd*threads+threadIdx.x] * points[(3+td)*threads+threadIdx.x];
					__syncthreads();

					volatile float *warpLine = &buffer[wid*WARP_SIZE];
					if(gtid < WARP_SIZE/2){
						unsigned int posb = gid * group_length;
						if(groups_per_warp<=1) 	warpLine[posb + gtid] += warpLine[posb + gtid + 16];
						if(groups_per_warp<=2) 	warpLine[posb + gtid] += warpLine[posb + gtid + 8];
						if(groups_per_warp<=4) 	warpLine[posb + gtid] += warpLine[posb + gtid + 4];
						if(groups_per_warp<=8) 	warpLine[posb + gtid] += warpLine[posb + gtid + 2];
						if(groups_per_warp<=16) warpLine[posb + gtid] += warpLine[posb + gtid + 1];


						if(gtid<groups_per_warp)
						{
							matricesH[(sd*3+td)*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length];
//							output_correlationMatrixes[ (sd*3+td)*n_rsac+blockIdx.x*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length];

						}
					}

				}


			}

			__syncthreads();
*/
		}

	};
	__global__ void estimateTransformations(const TransformationMatrixEstimator tfe){ tfe(); }
}


device::CorrespondenceListEstimator correspondanceList;
device::TransformationMatrixEstimator transformEstimator;

void
RigidBodyTransformationAdvancedEstimatior::init()
{

	correspondanceList.input_feature_histo = (float *)getInputDataPointer(HistogramMap);
	correspondanceList.input_idxList = (int *)getInputDataPointer(IdxList);
	correspondanceList.input_idxLength = (unsigned int *)getInputDataPointer(InfoList);

	transformEstimator.input_pos = (float4 *)getInputDataPointer(Coordiantes);
}

void
RigidBodyTransformationAdvancedEstimatior::execute()
{

	unsigned int view_combinations = ((n_view-1)*n_view)/2;

	printf("combinations: %d \n",view_combinations);

	thrust::device_vector<float> d_rndSList(s*view_combinations);
	correspondanceList.input_rndList = thrust::raw_pointer_cast(d_rndSList.data());

	thrust::device_vector<unsigned int> d_sIdxList(s*view_combinations);
	thrust::device_vector<float> d_corresp_prob(s*correspondanceList.corresp_list_length*view_combinations);
	thrust::device_vector<unsigned int> d_corresp_idx(s*correspondanceList.corresp_list_length*view_combinations);

	correspondanceList.output_s_idxList = thrust::raw_pointer_cast(d_sIdxList.data());
	correspondanceList.output_corresp_prob = thrust::raw_pointer_cast(d_corresp_prob.data());
	correspondanceList.output_corresp_idx = thrust::raw_pointer_cast(d_corresp_idx.data());


	correspondanceList.n_view = n_view;
	dim3 correspBlock(correspondanceList.dx);
	dim3 correspGrid(s/correspondanceList.local_points_per_block,1,view_combinations);

	curandGenerator_t gen ;
	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));

	checkCudaErrors(curandGenerateUniform(gen,correspondanceList.input_rndList,correspGrid.x*correspondanceList.local_points_per_block*view_combinations));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	device::estimateBestCorrespondences<<<correspGrid,correspBlock>>>(correspondanceList);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::host_vector<float> h_corresp_prob = d_corresp_prob;
	thrust::host_vector<unsigned int> h_cooresp_idx = d_corresp_idx;

	for(int i=0;i<8;i++)
	{
		printf("%d :",i);
		for(int k=0;k<correspondanceList.corresp_list_length;k++)
		{
			printf("%f | ",h_corresp_prob[k*s+i]);
		}
		printf("\n");
//		for(int k=0;k<correspondanceList.corresp_list_length;k++)
//		{
//			printf("%d | ",h_cooresp_idx[k*s+i]);
//		}
//		printf("\n");
	}


	transformEstimator.input_src_idx_corresp = correspondanceList.output_s_idxList;
	transformEstimator.input_target_idx_corresp = correspondanceList.output_corresp_idx;
	transformEstimator.input_target_prob_corresp = correspondanceList.output_corresp_prob;
	transformEstimator.n_view = n_view;
	transformEstimator.s_length = s;

	device::estimateTransformations<<<1,transformEstimator.threads>>>(transformEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



//	thrust::host_vector<float> h_rndSList = d_rndSList;
//
//	for(int i=0;i<s*view_combinations;i+=8)
//	{
//		printf("%d -> %f \n",i,h_rndSList.data()[i]);
//	}

}



RigidBodyTransformationAdvancedEstimatior::RigidBodyTransformationAdvancedEstimatior(unsigned int n_view, unsigned int rn, unsigned int s,unsigned int k) : n_view(n_view), rn(rn), s(s), k(k)
{
	DeviceDataParams transformationmatrixesParams;
	transformationmatrixesParams.elements = rn * ((n_view-1)*n_view)/2;
	transformationmatrixesParams.element_size = 12 * sizeof(float);
	transformationmatrixesParams.elementType = FLOAT1;
	transformationmatrixesParams.dataType = Matrix;
	addTargetData(addDeviceDataRequest(transformationmatrixesParams),TransformationMatrices);

	DeviceDataParams transformationMetaDataList;
	transformationMetaDataList.elements = ((n_view-1)*n_view)/2;
	transformationMetaDataList.element_size = sizeof(int);
	transformationMetaDataList.elementType = TransformationInfoListItem;
	transformationMetaDataList.dataType = Indice;
	addTargetData(addDeviceDataRequest(transformationMetaDataList),TransformationMetaDataList);
}

void RigidBodyTransformationAdvancedEstimatior::TestRBTAFct()
{
//	correspondanceList.bins_n_meta;

	n_view = 4;

	unsigned int s = 32;
	unsigned int view_combinations = ((n_view-1)*n_view)/2;

	printf("combinations: %d \n",view_combinations);

	curandGenerator_t gen ;
	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));

	thrust::device_vector<float> d_histos(n_view*640*480*correspondanceList.bins_n_meta);
	thrust::device_vector<int> d_idxList(n_view*640*480);
	thrust::device_vector<unsigned int> d_idxLength(n_view);
	thrust::device_vector<float> d_rndSList(s*view_combinations);

	correspondanceList.input_feature_histo = thrust::raw_pointer_cast(d_histos.data());
	correspondanceList.input_idxList = thrust::raw_pointer_cast(d_idxList.data());
	correspondanceList.input_idxLength = thrust::raw_pointer_cast(d_idxLength.data());
	correspondanceList.input_rndList = thrust::raw_pointer_cast(d_rndSList.data());

	correspondanceList.n_view = n_view;
	dim3 TestBlock(correspondanceList.dx);
	dim3 TestGrid(s/correspondanceList.local_points_per_block,1,view_combinations);


	checkCudaErrors(curandGenerateUniform(gen,correspondanceList.input_rndList,TestGrid.x*correspondanceList.local_points_per_block*view_combinations));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


//	correspondanceList.

	device::estimateBestCorrespondences<<<TestGrid,TestBlock>>>(correspondanceList);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



}

RigidBodyTransformationAdvancedEstimatior::~RigidBodyTransformationAdvancedEstimatior()
{
	// TODO Auto-generated destructor stub
}



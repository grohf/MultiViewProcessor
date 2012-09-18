/*
 * RigidBodyTransformationEstimator.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#include <curand.h>
#include <helper_cuda.h>

#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>

#include "SVDEstimatorCPU.h"
#include "RigidBodyTransformationEstimator.h"
#include "utils.hpp"


#define CURAND_CALL ( x ) do { if (( x ) != CURAND_STATUS_SUCCESS ) { \
printf (" Error at % s :% d \ n " , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)

#ifndef MAX_VIEWS
#define MAX_VIEWS 8
#endif

namespace device
{
	namespace constant
	{
//		__constant__ unsigned int idx2xyTriangle[(MAX_VIEWS*(MAX_VIEWS+1))/2][2];
	}

	struct OMEstimator
	{

		__device__ __forceinline__ void
		operator () () const
		{

		}

	};

	struct RigidBodyTransformationMatrixEstimator
	{
		enum
		{
			kernels = 1,
		};

		unsigned int n_ransac;

		float *input_correletationmatrices;
		float *output_rotationMatrices;
		float *output_translationVector;

		__device__ __forceinline__ void
		operator () () const
		{
			float m[9];
			for(int d=0;d<9;d++)
			{
				m[d] = input_correletationmatrices[d*n_ransac+blockIdx.x*blockDim.x+threadIdx.x];
			}

			float cov[6];
			cov[0] = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
			cov[1] = m[0]*m[3] + m[1]*m[4] + m[2]*m[5];
			cov[2] = m[0]*m[6] + m[1]*m[7] + m[2]*m[8];

			cov[3] = m[3]*m[3] + m[4]*m[4] + m[5]*m[5];
			cov[4] = m[3]*m[6] + m[4]*m[7] + m[5]*m[8];

			cov[5] = m[6]*m[6] + m[7]*m[7] + m[8]*m[8];

			typedef Eigen33::Mat33 Mat33;
			Eigen33 eigen33(cov);

			Mat33 tmp;
			Mat33 vec_tmp;
			Mat33 evecs;
			float3 evals;

			eigen33.compute(tmp, vec_tmp, evecs, evals);

			if(evals.x==0 || evals.y == 0 || evals.z == 0)
			{
				output_rotationMatrices[blockIdx.x*n_ransac+blockIdx.x*blockDim.x+threadIdx.x] = -1.f;
				return;
			}

			for(int i=0;i<3;i++)
			{
				tmp[i].x = 0;
				tmp[i].y = 0;
				tmp[i].z = 0;

				vec_tmp[i].x = 0;
				vec_tmp[i].y = 0;
				vec_tmp[i].z = 0;
			}

			float eval_sqrt = sqrtf(evals.x);

			tmp[0].x += (evecs[0].x*evecs[0].x)/eval_sqrt;
			tmp[1].x += (evecs[0].x*evecs[0].y)/eval_sqrt;
			tmp[2].x += (evecs[0].x*evecs[0].z)/eval_sqrt;

			tmp[0].y += (evecs[0].y*evecs[0].x)/eval_sqrt;
			tmp[1].y += (evecs[0].y*evecs[0].y)/eval_sqrt;
			tmp[2].y += (evecs[0].y*evecs[0].z)/eval_sqrt;

			tmp[0].z += (evecs[0].z*evecs[0].x)/eval_sqrt;
			tmp[1].z += (evecs[0].z*evecs[0].y)/eval_sqrt;
			tmp[2].z += (evecs[0].z*evecs[0].z)/eval_sqrt;

			eval_sqrt = sqrtf(evals.y);
			tmp[0].x += (evecs[1].x*evecs[1].x)/eval_sqrt;
			tmp[1].x += (evecs[1].x*evecs[1].y)/eval_sqrt;
			tmp[2].x += (evecs[1].x*evecs[1].z)/eval_sqrt;

			tmp[0].y += (evecs[1].y*evecs[1].x)/eval_sqrt;
			tmp[1].y += (evecs[1].y*evecs[1].y)/eval_sqrt;
			tmp[2].y += (evecs[1].y*evecs[1].z)/eval_sqrt;

			tmp[0].z += (evecs[1].z*evecs[1].x)/eval_sqrt;
			tmp[1].z += (evecs[1].z*evecs[1].y)/eval_sqrt;
			tmp[2].z += (evecs[1].z*evecs[1].z)/eval_sqrt;

			eval_sqrt = sqrtf(evals.z);
			tmp[0].x += (evecs[2].x*evecs[2].x)/eval_sqrt;
			tmp[1].x += (evecs[2].x*evecs[2].y)/eval_sqrt;
			tmp[2].x += (evecs[2].x*evecs[2].z)/eval_sqrt;

			tmp[0].y += (evecs[2].y*evecs[2].x)/eval_sqrt;
			tmp[1].y += (evecs[2].y*evecs[2].y)/eval_sqrt;
			tmp[2].y += (evecs[2].y*evecs[2].z)/eval_sqrt;

			tmp[0].z += (evecs[2].z*evecs[2].x)/eval_sqrt;
			tmp[1].z += (evecs[2].z*evecs[2].y)/eval_sqrt;
			tmp[2].z += (evecs[2].z*evecs[2].z)/eval_sqrt;



		}

	};

	struct DeMeanedCorrelationMatrixEstimator
	{
		enum
		{
			threads = 512,
			n_corresp = 8,
			n_matrices = threads/n_corresp,

			WARP_SIZE = 32,
			groups_per_warp = WARP_SIZE/n_corresp,
			group_length = WARP_SIZE/groups_per_warp,
		};

		unsigned int view_src;
		unsigned int view_target;

		float4 *pos;

		float* correspondanceErrorList;
		unsigned int* correspondanceErrorIdxList;

		unsigned int* correspondanceIdxList;

		float* rnd_src;
		float* rnd_target;

		unsigned int n_src;
		unsigned int n_target;

		unsigned int n_rsac;
		float *output_correlationMatrixes;

		float *output_transformationMatrices;
		int *output_transformationMetaData;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float points[threads*6];
//			__shared__ float ps_y[n_matrices*n_corresp];
//			__shared__ float ps_z[n_matrices*n_corresp];

//			__shared__ float points_target[n_matrices*n_corresp*3];
//			__shared__ float pt_y[n_matrices*n_corresp];
//			__shared__ float pt_z[n_matrices*n_corresp];

//			__shared__ float b_x[n_matrices*n_corresp];
//			__shared__ float b_y[n_matrices*n_corresp];
//			__shared__ float b_z[n_matrices*n_corresp];

			__shared__ float buffer[threads];

//			__shared__ float c_x[n_matrices];
//			__shared__ float c_y[n_matrices];
//			__shared__ float c_z[n_matrices];

			__shared__ float centroids[n_matrices*6];
			__shared__ float matricesH[n_matrices*9];

//			unsigned int tid = threadIdx.x;

//			for(int i=threadIdx.x; i<n_matrices*n_corresp;i+=blockDim.x)
			{
				unsigned int tid = threadIdx.x;
				unsigned int idx = (unsigned int)(rnd_src[blockIdx.x*threads+tid]*n_src);
				unsigned int pidx = correspondanceIdxList[idx];
				float4 tmp = pos[view_src*640*480+pidx];
//				points[threads*0 + tid] = idx;
				points[threads*0 + tid] = tmp.x;
				points[threads*1 + tid] = tmp.y;
				points[threads*2 + tid] = tmp.z;


				pidx = correspondanceErrorIdxList[idx*n_target + (unsigned int)(rnd_target[blockIdx.x*threads+tid]*n_target)];
				tmp = pos[view_target*640*480+pidx];
				points[threads*3 + tid] = tmp.x;
				points[threads*4 + tid] = tmp.y;
				points[threads*5 + tid] = tmp.z;


//				points[threads*0 + tid] = tid;
//				points[threads*1 + tid] = tid;
//				points[threads*2 + tid] = tid;
//				points[threads*3 + tid] = tid;
//				points[threads*4 + tid] = tid;
//				points[threads*5 + tid] = tid;

			}
			__syncthreads(); //TODO: necessary?




//			__syncthreads();
//			if(blockIdx.x == 0 && threadIdx.x==0)
//			{
//				for(int i=0;i<threads;i++)
//				{
//					printf("(%d) %f | ",i,points[i]);
//				}
//				printf(" \n");
//			}



//			unsigned int gid = threadIdx.x/n_corresp;
//			unsigned int tid = threadIdx.x - gid * n_corresp;

			unsigned int wid = threadIdx.x/WARP_SIZE;
			unsigned int wtid = threadIdx.x - wid * WARP_SIZE;
			unsigned int gid = wtid/group_length;
			unsigned int gtid = wtid - gid * group_length;
/*
//			unsigned int stride = groups_per_warp*2;

//			if(threadIdx.x==0)
//				printf("groups per warp: %d | stride: %d \n",groups_per_warp,stride);

//			//TEST
//			if(blockIdx.x == 0 && threadIdx.x<WARP_SIZE)
//			{
//				ps_x[threadIdx.x] = threadIdx.x;
//				ps_y[threadIdx.x] = threadIdx.x;
//				ps_z[threadIdx.x] = threadIdx.x;
//
//				pt_x[threadIdx.x] = threadIdx.x+100;
//				pt_y[threadIdx.x] = threadIdx.x+100;
//				pt_z[threadIdx.x] = threadIdx.x+100;
//
//			}

//			if(wtid<(WARP_SIZE/2))
//			{
//				unsigned int pos = wid*32 + wgid*stride + gtid;
////				unsigned int pos_load = wid*32+
//				b_x[pos] = ps_x[wid*32+wtid*2] + ps_x[wid*32+wtid*2+1];
//				b_y[pos] = ps_y[wid*32+wtid*2] + ps_y[wid*32+wtid*2+1];
//				b_z[pos] = ps_z[wid*32+wtid*2] + ps_z[wid*32+wtid*2+1];
//
//			}
//			else
//			{
//				unsigned int pos = wid*32 + ( (wtid-(WARP_SIZE/2))/groups_per_warp)*stride + groups_per_warp + gtid;
//
//				b_x[pos] = pt_x[wid*32 + (wtid-WARP_SIZE/2)*2] + pt_x[wid*32 + (wtid-WARP_SIZE/2)*2+1];
//				b_y[pos] = pt_y[wid*32 + (wtid-WARP_SIZE/2)*2] + pt_y[wid*32 + (wtid-WARP_SIZE/2)*2+1];
//				b_z[pos] = pt_z[wid*32 + (wtid-WARP_SIZE/2)*2] + pt_z[wid*32 + (wtid-WARP_SIZE/2)*2+1];
//
//			}

//			__syncthreads();
//			if(blockIdx.x == 0 && threadIdx.x==0)
//			{
//				for(int i=0;i<n_matrices*n_corresp/32;i++)
//				{
//					for(int w=0;w<32;w++)
//					{
//						printf(" %f | ", b_x[i*32+w]);
//					}
//				printf(" \n ");
//
//				}
//			}
*/

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

	//				if(groups_per_warp*2 <= 8) warpLine[wtid] += warpLine[wtid + 8];
	//				if(groups_per_warp*2 <= 4) warpLine[wtid] += warpLine[wtid + 4];
	//				if(groups_per_warp*2 <= 2) warpLine[wtid] += warpLine[wtid + 2];
	//				if(groups_per_warp < 2) warpLine[wtid] += warpLine[wtid + 1];
					if(gtid<groups_per_warp)
					{
						centroids[d*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length]/group_length;
					}
				}
			}
			__syncthreads();

//			if(blockIdx.x == 0 && threadIdx.x==0)
//			{
//				for(int d=0;d<2;d++)
//				{
//					for(int l=0;l<2;l++)
//					{
//						printf("l: %d \n",l);
//						for(int w=0;w<groups_per_warp;w++)
//						{
//							printf("%f | ", centroids[d*n_matrices + l*groups_per_warp+w]);
//						}
//						printf(" \n ");
//					}
//				}
//			}

			points[threads*0 + threadIdx.x] -= centroids[0*n_matrices+wid*groups_per_warp+gid];
			points[threads*1 + threadIdx.x] -= centroids[1*n_matrices+wid*groups_per_warp+gid];
			points[threads*2 + threadIdx.x] -= centroids[2*n_matrices+wid*groups_per_warp+gid];

			points[threads*3 + threadIdx.x] -= centroids[3*n_matrices+wid*groups_per_warp+gid];
			points[threads*4 + threadIdx.x] -= centroids[4*n_matrices+wid*groups_per_warp+gid];
			points[threads*5 + threadIdx.x] -= centroids[5*n_matrices+wid*groups_per_warp+gid];

//			__syncthreads();
//			if(blockIdx.x == 0 && threadIdx.x==0)
//			{
//				for(int i=0;i<36;i++)
//				{
//					printf("(%d) %f | ",i,points[i]);
//				}
//				printf(" \n ");
//			}


			__syncthreads();
			for(int sd=0;sd<3;sd++)
			{
				for(int td=0;td<3;td++)
				{

					buffer[threadIdx.x] = points[sd*threads+threadIdx.x] * points[(3+td)*threads+threadIdx.x];
					__syncthreads();

//					if(threadIdx.x==32 && sd==0 && td==0)
//					{
//						for(int i=32;i<48;i++)
//						{
//							printf("i: %d -> %f = %f * %f \n",i,buffer[i],points[sd*threads+i],points[(3+td)*threads+i]);
//						}
//					}

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
							matricesH[(sd*3+td)*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length];

//							if(gtid==0 && sd==0 && td==0)
//								printf("r: %f | gtid: %d \n",warpLine[gtid*group_length],gtid);
//							output_correlationMatrixes[ (sd*3+td)*n_rsac+blockIdx.x*n_matrices+wid*groups_per_warp+gtid] = warpLine[gtid*group_length];
//							if(blockIdx.x == 0 && threadIdx.x==0)
//							{
//								for(int i=0;i<groups_per_warp;i++)
//								{
//									printf("(%d) %f | ",i,warpLine[gtid*group_length]);
//								}
//								printf(" \n");
//							}

						}
					}

				}


			}

			__syncthreads();

			//Calculate TransformationMatrix via OM
//			if(threadIdx.x<n_matrices)
			if(threadIdx.x<n_matrices)
			{
//				printf("%d \n",threadIdx.x);
				volatile float *cov = &points[threadIdx.x*6];

				cov[0] = matricesH[0*n_matrices + threadIdx.x]*matricesH[0*n_matrices + threadIdx.x] + matricesH[1*n_matrices + threadIdx.x]*matricesH[1*n_matrices + threadIdx.x] + matricesH[2*n_matrices + threadIdx.x]*matricesH[2*n_matrices + threadIdx.x];
				cov[1] = matricesH[0*n_matrices + threadIdx.x]*matricesH[3*n_matrices + threadIdx.x] + matricesH[1*n_matrices + threadIdx.x]*matricesH[4*n_matrices + threadIdx.x] + matricesH[2*n_matrices + threadIdx.x]*matricesH[5*n_matrices + threadIdx.x];
				cov[2] = matricesH[0*n_matrices + threadIdx.x]*matricesH[6*n_matrices + threadIdx.x] + matricesH[1*n_matrices + threadIdx.x]*matricesH[7*n_matrices + threadIdx.x] + matricesH[2*n_matrices + threadIdx.x]*matricesH[8*n_matrices + threadIdx.x];

				cov[3] = matricesH[3*n_matrices + threadIdx.x]*matricesH[3*n_matrices + threadIdx.x] + matricesH[4*n_matrices + threadIdx.x]*matricesH[4*n_matrices + threadIdx.x] + matricesH[5*n_matrices + threadIdx.x]*matricesH[5*n_matrices + threadIdx.x];
				cov[4] = matricesH[3*n_matrices + threadIdx.x]*matricesH[6*n_matrices + threadIdx.x] + matricesH[4*n_matrices + threadIdx.x]*matricesH[7*n_matrices + threadIdx.x] + matricesH[5*n_matrices + threadIdx.x]*matricesH[8*n_matrices + threadIdx.x];

				cov[5] = matricesH[6*n_matrices + threadIdx.x]*matricesH[6*n_matrices + threadIdx.x] + matricesH[7*n_matrices + threadIdx.x]*matricesH[7*n_matrices + threadIdx.x] + matricesH[8*n_matrices + threadIdx.x]*matricesH[8*n_matrices + threadIdx.x];

//				if()

                typedef Eigen33::Mat33 Mat33;
                Eigen33 eigen33(cov);

                Mat33& tmp 		= (Mat33&)points[n_matrices*6+threadIdx.x*9];
                Mat33& vec_tmp 	= (Mat33&)points[n_matrices*(6+9)+threadIdx.x*9];
                Mat33& evecs 	= (Mat33&)points[n_matrices*(6+18)+threadIdx.x*9];
                float3 evals;

                eigen33.compute(tmp, vec_tmp, evecs, evals);

    			if(evals.x==0 || evals.y == 0 || evals.z == 0)
    			{
//    				printf("out: %d \n",threadIdx.x);

//    				output_transformationMatrices[blockIdx.x*n_rsac+threadIdx.x] = -1.f;

    				output_transformationMetaData[blockIdx.x*n_matrices+threadIdx.x] = -1;
    				output_transformationMetaData[n_rsac + blockIdx.x*n_matrices +threadIdx.x] = -1;
    				return;
    			}

    			for(int i=0;i<3;i++)
    			{
    				tmp[i].x = 0;
    				tmp[i].y = 0;
    				tmp[i].z = 0;

    				vec_tmp[i].x = 0;
    				vec_tmp[i].y = 0;
    				vec_tmp[i].z = 0;
    			}
/*
			tmp[0].x += (evecs[0].x*evecs[0].x)/eval_sqrt;
			tmp[1].x += (evecs[0].x*evecs[0].y)/eval_sqrt;
			tmp[2].x += (evecs[0].x*evecs[0].z)/eval_sqrt;

			tmp[0].y += (evecs[0].y*evecs[0].x)/eval_sqrt;
			tmp[1].y += (evecs[0].y*evecs[0].y)/eval_sqrt;
			tmp[2].y += (evecs[0].y*evecs[0].z)/eval_sqrt;

			tmp[0].z += (evecs[0].z*evecs[0].x)/eval_sqrt;
			tmp[1].z += (evecs[0].z*evecs[0].y)/eval_sqrt;
			tmp[2].z += (evecs[0].z*evecs[0].z)/eval_sqrt;
*/
			float eval_sqrt = sqrtf(evals.x);
			tmp[0].x += (evecs[0].x*evecs[0].x)/eval_sqrt;
			tmp[0].y += (evecs[0].x*evecs[0].y)/eval_sqrt;
			tmp[0].z += (evecs[0].x*evecs[0].z)/eval_sqrt;

			tmp[1].x += (evecs[0].y*evecs[0].x)/eval_sqrt;
			tmp[1].y += (evecs[0].y*evecs[0].y)/eval_sqrt;
			tmp[1].z += (evecs[0].y*evecs[0].z)/eval_sqrt;

			tmp[2].x += (evecs[0].z*evecs[0].x)/eval_sqrt;
			tmp[2].y += (evecs[0].z*evecs[0].y)/eval_sqrt;
			tmp[2].z += (evecs[0].z*evecs[0].z)/eval_sqrt;

			eval_sqrt = sqrtf(evals.y);
			tmp[0].x += (evecs[1].x*evecs[1].x)/eval_sqrt;
			tmp[0].y += (evecs[1].x*evecs[1].y)/eval_sqrt;
			tmp[0].z += (evecs[1].x*evecs[1].z)/eval_sqrt;

			tmp[1].x += (evecs[1].y*evecs[1].x)/eval_sqrt;
			tmp[1].y += (evecs[1].y*evecs[1].y)/eval_sqrt;
			tmp[1].z += (evecs[1].y*evecs[1].z)/eval_sqrt;

			tmp[2].x += (evecs[1].z*evecs[1].x)/eval_sqrt;
			tmp[2].y += (evecs[1].z*evecs[1].y)/eval_sqrt;
			tmp[2].z += (evecs[1].z*evecs[1].z)/eval_sqrt;

			eval_sqrt = sqrtf(evals.z);
			tmp[0].x += (evecs[2].x*evecs[2].x)/eval_sqrt;
			tmp[0].y += (evecs[2].x*evecs[2].y)/eval_sqrt;
			tmp[0].z += (evecs[2].x*evecs[2].z)/eval_sqrt;

			tmp[1].x += (evecs[2].y*evecs[2].x)/eval_sqrt;
			tmp[1].y += (evecs[2].y*evecs[2].y)/eval_sqrt;
			tmp[1].z += (evecs[2].y*evecs[2].z)/eval_sqrt;

			tmp[2].x += (evecs[2].z*evecs[2].x)/eval_sqrt;
			tmp[2].y += (evecs[2].z*evecs[2].y)/eval_sqrt;
			tmp[2].z += (evecs[2].z*evecs[2].z)/eval_sqrt;

			vec_tmp[0].x = matricesH[0*n_matrices + threadIdx.x]*tmp[0].x + matricesH[3*n_matrices + threadIdx.x]*tmp[1].x + matricesH[6*n_matrices + threadIdx.x]*tmp[2].x;
			vec_tmp[0].y = matricesH[0*n_matrices + threadIdx.x]*tmp[0].y + matricesH[3*n_matrices + threadIdx.x]*tmp[1].y + matricesH[6*n_matrices + threadIdx.x]*tmp[2].y;
			vec_tmp[0].z = matricesH[0*n_matrices + threadIdx.x]*tmp[0].z + matricesH[3*n_matrices + threadIdx.x]*tmp[1].z + matricesH[6*n_matrices + threadIdx.x]*tmp[2].z;

			vec_tmp[1].x = matricesH[1*n_matrices + threadIdx.x]*tmp[0].x + matricesH[4*n_matrices + threadIdx.x]*tmp[1].x + matricesH[7*n_matrices + threadIdx.x]*tmp[2].x;
			vec_tmp[1].y = matricesH[1*n_matrices + threadIdx.x]*tmp[0].y + matricesH[4*n_matrices + threadIdx.x]*tmp[1].y + matricesH[7*n_matrices + threadIdx.x]*tmp[2].y;
			vec_tmp[1].z = matricesH[1*n_matrices + threadIdx.x]*tmp[0].z + matricesH[4*n_matrices + threadIdx.x]*tmp[1].z + matricesH[7*n_matrices + threadIdx.x]*tmp[2].z;

			vec_tmp[2].x = matricesH[2*n_matrices + threadIdx.x]*tmp[0].x + matricesH[5*n_matrices + threadIdx.x]*tmp[1].x + matricesH[8*n_matrices + threadIdx.x]*tmp[2].x;
			vec_tmp[2].y = matricesH[2*n_matrices + threadIdx.x]*tmp[0].y + matricesH[5*n_matrices + threadIdx.x]*tmp[1].y + matricesH[8*n_matrices + threadIdx.x]*tmp[2].y;
			vec_tmp[2].z = matricesH[2*n_matrices + threadIdx.x]*tmp[0].z + matricesH[5*n_matrices + threadIdx.x]*tmp[1].z + matricesH[8*n_matrices + threadIdx.x]*tmp[2].z;

//			if(blockIdx.x==0) printf("%d \n",threadIdx.x);

			float det = 0.f;
			det += vec_tmp[0].x * vec_tmp[1].y * vec_tmp[2].z;
			det += vec_tmp[0].y * vec_tmp[1].z * vec_tmp[2].x;
			det += vec_tmp[0].z * vec_tmp[1].x * vec_tmp[2].y;

			det -= vec_tmp[0].z * vec_tmp[1].y * vec_tmp[2].x;
			det -= vec_tmp[0].y * vec_tmp[1].x * vec_tmp[2].z;
			det -= vec_tmp[0].x * vec_tmp[1].z * vec_tmp[2].y;

			int ret = (det < 0.9f || det > 1.1f)?(int)-1:(int)(blockIdx.x*n_matrices+threadIdx.x);

			output_transformationMetaData[1+blockIdx.x*n_matrices+threadIdx.x] = ret;
			output_transformationMetaData[1+n_rsac + blockIdx.x*n_matrices +threadIdx.x] = view_src;
			output_transformationMetaData[1+2*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = view_target;

			if(ret==-1)
				return;


			output_transformationMatrices[0*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[0].x;
			output_transformationMatrices[1*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[0].y;
			output_transformationMatrices[2*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[0].z;

			output_transformationMatrices[3*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[1].x;
			output_transformationMatrices[4*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[1].y;
			output_transformationMatrices[5*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[1].z;

			output_transformationMatrices[6*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[2].x;
			output_transformationMatrices[7*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[2].y;
			output_transformationMatrices[8*n_rsac + blockIdx.x*n_matrices +threadIdx.x] = vec_tmp[2].z;


			output_transformationMatrices[9*n_rsac 	+ blockIdx.x*n_matrices +threadIdx.x]  	= centroids[3*n_matrices+threadIdx.x] - (vec_tmp[0].x*centroids[0*n_matrices+threadIdx.x] + vec_tmp[0].y*centroids[1*n_matrices+threadIdx.x] + vec_tmp[0].z*centroids[2*n_matrices+threadIdx.x]);
			output_transformationMatrices[10*n_rsac + blockIdx.x*n_matrices +threadIdx.x]  	= centroids[4*n_matrices+threadIdx.x] - (vec_tmp[1].x*centroids[0*n_matrices+threadIdx.x] + vec_tmp[1].y*centroids[1*n_matrices+threadIdx.x] + vec_tmp[1].z*centroids[2*n_matrices+threadIdx.x]);
			output_transformationMatrices[11*n_rsac + blockIdx.x*n_matrices +threadIdx.x]  	= centroids[5*n_matrices+threadIdx.x] - (vec_tmp[2].x*centroids[0*n_matrices+threadIdx.x] + vec_tmp[2].y*centroids[1*n_matrices+threadIdx.x] + vec_tmp[2].z*centroids[2*n_matrices+threadIdx.x]);

//			printf("(%d) det: %f \n",threadIdx.x,det);

//			__syncthreads();
//			if(threadIdx.x==0 && blockIdx.x==0)
//			{
//
//			}

//			__syncthreads();
//			if(threadIdx.x==0 && blockIdx.x==0)
//			{
////				printf("R: %d \n",n_matrices);
////				for(int j=0;j<3;j++)
////				{
////					printf("%f | %f | %f \n",vec_tmp[j].x,vec_tmp[j].y,vec_tmp[j].z);
////				}
//
//				for(int i=0;i<n_matrices;i++)
//				{
//	                Mat33& r 	= (Mat33&)points[n_matrices*(6+9)+i*9];
//
//				float det = 0.f;
//				det += r[0].x * r[1].y * r[2].z;
//				det += r[0].y * r[1].z * r[2].x;
//				det += r[0].z * r[1].x * r[2].y;
//
//				det -= r[0].z * r[1].y * r[2].x;
//				det -= r[0].y * r[1].x * r[2].z;
//				det -= r[0].x * r[1].z * r[2].y;
//
//				printf("(%d) det: %f \n",i,det);
//				}
//
//			}

		}


//			for(int i=threadIdx.x; i<n_matrices*n_corresp;i+=blockDim.x)
//			{
//				unsigned int idx = correspondanceErrorIdxList[(unsigned int)(rnd_target[blockDim.x*n_matrices*n_corresp+i]*n_target)];
//				float4 tmp = pos[view_target*640*480+idx];
//				ps_x[i] = tmp.x;
//				ps_y[i] = tmp.y;
//				ps_z[i] = tmp.z;
//
//				b_x[i] = tmp.x;
//				b_y[i] = tmp.y;
//				b_z[i] = tmp.z;
//
//			}


		}


	};
	__global__ void estimateCorrelationMatrix(const DeMeanedCorrelationMatrixEstimator cme) {cme (); }

	template<unsigned int bins>
	struct SKCorrespondanceEstimator
	{
		enum
		{
			sp = 32,
			k = 32,
		};

		unsigned int view_src;
		unsigned int view_target;

		float *histoMap;
//		float *histoMap2;

		unsigned int *idxList;
//		unsigned int *idxList2;

		float *rndList;
		unsigned int *infoList;

		unsigned int *output_s_idx;

		unsigned int *output_idx;
		float *output_prob;


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

//		__device__ __forceinline__ void compareDiv(
//			volatile float  &val1,
//			volatile float  &val2,
//			volatile unsigned int  &idx1,
//			volatile unsigned int  &idx2,
//			unsigned int  dir
//		) const
//		{
//		    float tmp_val;
//		    unsigned int tmp_idx;
//
//		    if ((val1 > val2) == dir)
//		    {
//		    	tmp_val = val1;
//		        val1 = val2;
//		        val2 = tmp_val;
//		        tmp_idx = idx1;
//		        idx1 = idx2;
//		        idx2 = tmp_idx;
//		    }
//		}
//
//		__device__ __forceinline__ void compareDivLists(
//				float *val1,
//				float *val2,
//				unsigned int *idx1,
//				unsigned int *idx2,
//				unsigned int length,
//				unsigned int pos1,
//				unsigned int pos2,
//				unsigned int dir
//				) const
//		{
//			float *a_val = (pos1<length)?&val1[pos1]:&val2[pos1-length];
//			unsigned int *a_idx = (pos1<length)?&idx1[pos1]:&idx1[pos1-length];
//
//			float *b_val = (pos2<length)?&val1[pos2]:&val2[pos2-length];
//			unsigned int *b_idx = (pos2<length)?&idx2[pos2]:&idx2[pos2-length];
//
//
//			if ((a_val[0] > b_val[0]) == dir)
//			{
//				float tmp_val;
//				tmp_val = a_val[0];
//				a_val[0] = b_val[0];
//				b_val[0] = tmp_val;
//
//				unsigned int tmp_idx;
//				tmp_idx = a_idx[0];
//				a_idx[0] = b_idx[0];
//				b_idx[0] = tmp_idx;
//			}
//		}

		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float shm_local_histo[sp*bins];
			__shared__ unsigned int shm_local_idx[sp];

			__shared__ float shm_global_buffer[k*bins];
			__shared__ unsigned int shm_global_idx[k];


			__shared__ float shm_dist[k*sp*2];
			float *shm_dist_buffer = &(shm_dist[k*sp]);


			__shared__ unsigned int shm_idx[k*sp*2];
			unsigned int *shm_idx_buffer = &(shm_idx[k*sp]);

			if(threadIdx.x<sp)
			{
				float f = rndList[blockIdx.x*sp+threadIdx.x];
				unsigned int idx = idxList[view_src*640*480+(unsigned int)(f*infoList[view_src+1])];
				shm_local_idx[threadIdx.x] = idx;
				output_s_idx[blockIdx.x*sp+threadIdx.x] = idx;

//				if(threadIdx.x==10)
//					printf("rnd: %f idx_pos: %d idx: %d \n",f,(unsigned int)(f*infoList[view_src+1]),idx);

			}
			__syncthreads();

			for(int i=threadIdx.x; i<sp*bins; i+=blockDim.x)
			{
				unsigned int sl = i/bins;
				unsigned int b = i-sl*bins;

//				float f = rndList[blockIdx.x*sp+sl];
//
//				unsigned int idx = idxList1[(unsigned int)(f*infoList[view_src+1])];
//				output_idx[blockIdx.x*sp+sl] = idx;
//				unsigned int idx =  idxList1[blockIdx.x*sp+sl];


				shm_local_histo[sl + b*sp] = histoMap[view_src*640*480+shm_local_idx[sl]*bins+b];
			}


			//TODO Check last iterations idx
//			for(int l=0;l<(infoList[1]-1)/k+1;l+=k)
			for(int l=0;l<(infoList[view_target+1])/k;l+=k)
//			for(int l=0;l<3;l++)
			{

				//Get the index of the k global points;
				for(int i=threadIdx.x;i<k;i+=blockDim.x)
				{
					shm_global_idx[i] = idxList[view_src*640*480+i+l*k];
				}
				__syncthreads();

				for(int i=threadIdx.x; i<k*bins; i+=blockDim.x)
				{
					unsigned int kl = i/bins;
					unsigned int idx = shm_global_idx[kl];
					unsigned int b = i-kl*bins;

					shm_global_buffer[kl + b*k] = histoMap[view_target*640*480+idx*bins+b];
				}
				__syncthreads();

				for(int i=threadIdx.x; i<k*sp; i+=blockDim.x)
				{
					unsigned int wl = i/sp;
					shm_idx_buffer[i] = shm_global_idx[wl];
				}
				__syncthreads();

//				if(threadIdx.x==0)
//				{
//					bool check = true;
//					for(int kl=0;kl<k;kl++)
//					{
//						unsigned int idx = idxList2[kl+l*k];
//						for(int b=0;b<bins;b++)
//						{
//							if(!(histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]))
//								printf("%d -> %f = %f \n", (histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]),shm_global_buffer[kl+k*b],histoMap2[idx*bins+b]);
//
//							check = check && (histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]);
//						}
//
//					}
//					printf("check global buffer: %d  \n",check);
//
////					for(int kl=0;kl<k;kl++)
////					{
////						for(int sl=0;sl<s;sl++)
////						{
////							printf("%d | ",shm_idx_buffer[kl*s+sl]);
////						}
////						printf(" \n");
////					}
//
//				}
//				__syncthreads();


				unsigned int wid = threadIdx.x/32;
				unsigned int tid = threadIdx.x - wid*32;

				float div = 0.f;
				for(int b=0;b<bins;b++)
				{
					float d = shm_local_histo[tid+sp*b] - shm_global_buffer[wid+k*b];
					div += d*d;
				}
				div = sqrtf(div);
				shm_dist_buffer[wid*sp+tid] = div;
				__syncthreads();

//				for(int b=0;b<bins;b++)
//				{
//					float d = shm_histo_local[tid+s*b] - shm_global_buffer[wid*2+k*b];
//					div += d*d;
//				}
//				div = sqrtf(div);
//				shm_dist_buffer[wid*2*s+tid] = div;
//				__syncthreads();

//				__syncthreads();
//				if(threadIdx.x==0)
//				{
//					for(int kl=0;kl<k;kl++)
//					{
//						for(int sl=0;sl<s;sl++)
//						{
//							printf("%f | ",shm_dist_buffer[kl*s+sl]);
//						}
//						printf(" \n");
//					}
//				}
				//Bitonic sort

//				if(wid<k/2)
//				{
					for(int size = 2; size < k; size <<= 1)
					{
						unsigned int dir = 1 ^ ((wid & (size/2)) != 0);
						for(int stride = size/2; stride > 0; stride >>= 1)
						{
							__syncthreads();
							if(wid<k/2)
							{
								unsigned int pos = (2 * wid - (wid & (stride - 1))) * 32 + tid;
								compareDiv(shm_dist_buffer[pos], shm_dist_buffer[pos+stride*32], shm_idx_buffer[pos], shm_idx_buffer[pos+stride*32], dir);
							}
						}
					}


					for(int stride = k/2; stride > 0; stride >>= 1)
					{
						__syncthreads();
						if(wid<k/2)
						{
							unsigned int pos = (2 * wid - (wid & (stride - 1))) * 32 + tid;
							compareDiv(shm_dist_buffer[pos], shm_dist_buffer[pos+stride*32], shm_idx_buffer[pos], shm_idx_buffer[pos+stride*32], 1);
						}
					}
					__syncthreads();
//
////
////					__syncthreads();
////					if(threadIdx.x==0)
////					{
////						printf("---------------------------------- \n");
////						for(int kl=0;kl<k;kl++)
////						{
////							for(int sl=0;sl<s;sl++)
////							{
////								printf("%d | ",shm_idx_buffer[kl*s+sl]);
////							}
////							printf(" \n");
////						}
////					}
//
//
//
//				}
//				else
//				{
//					if(tid==0)
//						printf("wid: %d \n",wid);
//
//					__syncthreads();
//				}
//				__syncthreads();

//				__syncthreads();
//				if(threadIdx.x==120)
//				{
//					printf("l: %d ---------------------------------- \n",l);
//					for(int kl=0;kl<k;kl++)
//					{
//						for(int sl=0;sl<s;sl++)
//						{
////								int sl = 5;
//							printf("%f | ",shm_dist_buffer[kl*s+sl]);
//						}
//						printf(" \n");
//					}
//				}

				__syncthreads();
				if(l==0)
				{
					for(int i=threadIdx.x; i<k*sp; i+=blockDim.x)
					{
						shm_dist[i] = shm_dist_buffer[i];
						shm_idx[i] = shm_idx_buffer[i];
					}
					__syncthreads();

//					if(threadIdx.x==0)
//					{
//						printf("\n");
//						printf("---------------- %d. iteration: buffer ------------------ \n",l);
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist_buffer[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//						printf("\n");
//
//						printf("\n");
//						printf("---------------- %d. iteration: sorted ------------------ \n",l);
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//						printf("\n");
//					}
//
				}

				else
				{
//					__syncthreads();
//					if(threadIdx.x==0)
//					{
//						printf("\n");
//						printf("---------------- before ------------------ \n");
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist_sorted[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//						printf("\n");
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist_buffer[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//					}


//					if(threadIdx.x==0)
//						printf("l: %d \n",l);

					// Merge 2 sorted lists together

					unsigned int stride = k;
					unsigned int offset = wid & (stride - 1);
//
					__syncthreads();
//					if(wid<k/2)
//					{
						unsigned int pos = (2 * wid - (wid & (stride - 1)))*32 + tid;
						compareDiv(shm_dist[pos], shm_dist[pos+stride*32], shm_idx[pos], shm_idx[pos+stride*32],1);
//					}
//						compareDivLists(shm_dist_sorted,shm_dist_buffer,shm_idx_sorted,shm_idx_buffer,k*sp,
//							pos,pos+stride*32,1);
//
					for(stride >>= 1; stride > 0; stride >>=1)
					{
						__syncthreads();
//						if(wid<k/2)
//						{
							unsigned int pos = (2 * wid - (wid & (stride - 1)))*32 + tid;
							if(offset >= stride)
							{
								compareDiv(shm_dist[pos-stride*32], shm_dist[pos], shm_idx[pos-stride*32], shm_idx[pos],1);
	//							compareDivLists(shm_dist_sorted,shm_dist_buffer,shm_idx_sorted,shm_idx_buffer,k*sp,
	//									pos-stride*32,pos,1);
							}
//						}
					}
					__syncthreads();


//					__syncthreads();
//					if(threadIdx.x==0)
//					{
//						printf("\n");
//						printf("\n");
//						printf("---------------- after ------------------ \n");
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//						printf("\n");
//						for(int kl=0;kl<k;kl++)
//						{
//							for(int sl=0;sl<s;sl++)
//							{
//								printf("%f | ",shm_dist_buffer[kl*s+sl]);
//							}
//							printf(" \n");
//						}
//					}
				}
				__syncthreads();

				output_prob[blockIdx.x*sp*k+wid*32+tid] = shm_dist[wid*32+tid];
				output_idx[blockIdx.x*sp*k+wid*32+tid] = shm_idx[wid*32+tid];

			}



		}


	};
	__global__ void estimateCorrespondance(const SKCorrespondanceEstimator<8> ce) {ce (); }

	template<typename T, int Value>
	struct invalid
	{
		template<typename Tuple>
		__device__ bool operator()(const Tuple& tuple) const
		{
			const T idx = thrust::get<0>(tuple);

			if(idx==Value)
				return true;

			return false;
		}
	};


}

device::SKCorrespondanceEstimator<8> correspondanceEstimator;
device::DeMeanedCorrelationMatrixEstimator deMeanedCorrelatonMEstimator;

void RigidBodyTransformationEstimator::init()
{

	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));


	correspondanceEstimator.histoMap = (float *)getInputDataPointer(HistogramMap);
	correspondanceEstimator.idxList = (unsigned int *)getInputDataPointer(IdxList);

//	correspondanceEstimator.histoMap2 = (float *)getInputDataPointer(HistogramMap);
//	correspondanceEstimator.idxList2 = (unsigned int *)getInputDataPointer(IdxList);

	correspondanceEstimator.infoList = (unsigned int *)getInputDataPointer(InfoList);

	correspondanceEstimator.rndList = (float *)getTargetDataPointer(RndIndices);

	correspondanceEstimator.output_prob = (float *)getTargetDataPointer(ProbList);
	correspondanceEstimator.output_idx = (unsigned int *)getTargetDataPointer(ProbIdxList);
	correspondanceEstimator.output_s_idx = (unsigned int *)getTargetDataPointer(SIndices);



	block = dim3(correspondanceEstimator.sp*correspondanceEstimator.k,1);
	grid = dim3(s/correspondanceEstimator.sp,1);

	deMeanedCorrelatonMEstimator.pos = (float4 *)getInputDataPointer(Coordiantes);

	deMeanedCorrelatonMEstimator.correspondanceErrorIdxList = (unsigned int *)getTargetDataPointer(ProbIdxList);
	deMeanedCorrelatonMEstimator.correspondanceErrorList = (float *)getTargetDataPointer(ProbList);
	deMeanedCorrelatonMEstimator.correspondanceIdxList = (unsigned int *)getTargetDataPointer(SIndices);

	deMeanedCorrelatonMEstimator.n_src = s;
	deMeanedCorrelatonMEstimator.n_target = k;
	deMeanedCorrelatonMEstimator.rnd_src = (float *)getTargetDataPointer(RndSrcIndices);
	deMeanedCorrelatonMEstimator.rnd_target = (float *)getTargetDataPointer(RndTargetIndices);

	deMeanedCorrelatonMEstimator.n_rsac = rn;
	deMeanedCorrelatonMEstimator.output_correlationMatrixes = (float *)getTargetDataPointer(CorrelationMatrices);
	deMeanedCorrelatonMEstimator.output_transformationMatrices = (float *)getTargetDataPointer(TransformationMatrices);
	deMeanedCorrelatonMEstimator.output_transformationMetaData = (int *)getTargetDataPointer(TransformationMetaDataList);

	deMeanBlock = dim3(deMeanedCorrelatonMEstimator.threads,1);
	deMeanGrid = dim3(rn/deMeanedCorrelatonMEstimator.n_matrices,1);
}

void RigidBodyTransformationEstimator::execute()
{
//	printf("hmmm \n");
//	SVDEstimator_CPU *svd = new SVDEstimator_CPU();
//	svd->execute();

//	printf("d_infoList: %d \n",correspondanceEstimator.infoList);

	checkCudaErrors(curandGenerateUniform(gen,correspondanceEstimator.rndList,s));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	correspondanceEstimator.view_src = 0;
	correspondanceEstimator.view_target = 1;
	device::estimateCorrespondance<<<grid,block>>>(correspondanceEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	checkCudaErrors(curandGenerateUniform(gen,deMeanedCorrelatonMEstimator.rnd_src,rn*s*k));
	checkCudaErrors(curandGenerateUniform(gen,deMeanedCorrelatonMEstimator.rnd_target,rn*s*k));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	deMeanedCorrelatonMEstimator.view_src = 0;
	deMeanedCorrelatonMEstimator.view_target = 1;
	device::estimateCorrelationMatrix<<<deMeanGrid,deMeanBlock>>>(deMeanedCorrelatonMEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::device_ptr<int> metaData = thrust::device_pointer_cast(deMeanedCorrelatonMEstimator.output_transformationMetaData+1);

//	size_t length = thrust::remove_if(
//			thrust::make_zip_iterator(thrust::make_tuple(metaData,metaData+rn,metaData+2*rn)),
//			thrust::make_zip_iterator(thrust::make_tuple(metaData+rn,metaData+2*rn,metaData+3*rn)),
//					device::invalid<int,-1>())
//			- thrust::make_zip_iterator(thrust::make_tuple(metaData,metaData+rn,metaData+2*rn));
//

	size_t length = thrust::remove(metaData,metaData+rn,-1)- metaData;

	checkCudaErrors(cudaMemcpy(deMeanedCorrelatonMEstimator.output_transformationMetaData,&length,sizeof(int),cudaMemcpyHostToDevice));
	printf("length: %u \n",length);
//
//	int *h_metadata = (int *)malloc(rn*3*sizeof(int));
//	checkCudaErrors( cudaMemcpy(h_metadata,deMeanedCorrelatonMEstimator.output_transformationMetaData,rn*3*sizeof(int),cudaMemcpyDeviceToHost));

//	for(int off=0;off<length;off++)
//	{
//		printf(" %u  \n",h_metadata[off]);
//	}


//	float *h_corrmatrices = (float *)malloc(rn*9*sizeof(float));
//	checkCudaErrors( cudaMemcpy(h_corrmatrices,deMeanedCorrelatonMEstimator.output_correlationMatrixes,rn*9*sizeof(float),cudaMemcpyDeviceToHost));


//	for(int off=0;off<64;off++)
//	{
//		for(int j=0;j<3;j++)
//		{
//			for(int i=0;i<3;i++)
//			{
//				float p = h_corrmatrices[(j*3+i)*rn+off];
//				printf("%f ",p);
//			}
//			printf(" | ");
//		}
//		printf("\n");
//	}

	/*
	float *h_tmp = (float *)malloc(s*k*sizeof(float));
	checkCudaErrors( cudaMemcpy(h_tmp,correspondanceEstimator.output_prob,s*k*sizeof(float),cudaMemcpyDeviceToHost));

	unsigned int *h_idx = (unsigned int *)malloc(s*k*sizeof(unsigned int));
	checkCudaErrors( cudaMemcpy(h_idx,correspondanceEstimator.output_idx,s*k*sizeof(unsigned int),cudaMemcpyDeviceToHost));

	printf("\n");
	printf("\n");
	int st = 5;
	for(int j=0;j<32;j++)
	{
		for(int i=0;i<32;i++)
		{
			printf("%f | ",h_tmp[st*32*32+j*32+i]);
		}
		printf("\n");
	}

	printf("\n");
	printf("\n");
	for(int j=0;j<32;j++)
	{
		for(int i=0;i<32;i++)
		{
			printf("%d | ",h_idx[st*32*32+j*32+i]);
		}
		printf("\n");
	}

	printf("\n");
	printf("\n");

	unsigned int *h_s_idx = (unsigned int *)malloc(s);
	checkCudaErrors(cudaMemcpy(h_s_idx,correspondanceEstimator.output_s_idx,s*sizeof(unsigned int),cudaMemcpyDeviceToHost));

	for(int i=0;i<32;i++)
	{
		printf("%d | ",h_s_idx[i]);
	}
	*/


}


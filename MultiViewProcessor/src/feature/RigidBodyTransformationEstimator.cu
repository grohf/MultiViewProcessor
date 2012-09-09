/*
 * RigidBodyTransformationEstimator.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#include <curand.h>
#include <helper_cuda.h>

#include "SVDEstimatorCPU.h"
#include "RigidBodyTransformationEstimator.h"

# define CURAND_CALL ( x ) do { if (( x ) != CURAND_STATUS_SUCCESS ) { \
printf (" Error at % s :% d \ n " , __FILE__ , __LINE__ ) ;\
return EXIT_FAILURE ;}} while (0)


namespace device
{
	struct OMEstimator
	{

		__device__ __forceinline__ void
		operator () () const
		{

		}

	};

	template<unsigned int bins>
	struct SimilarCorrespondanceEstimator
	{
		enum
		{
			lx = 1024,
		};

		float *histoMap1;
		float *histoMap2;

		unsigned int *idxList1;
		unsigned int *idxList2;

		unsigned int k;
		unsigned int nList2;

		unsigned int output_idx;
		float output_prob;

		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float dist[lx];
			__shared__ unsigned int idx[lx];

			__shared__ float dist_tmp[lx];
			__shared__ unsigned int idx_tmp[lx];



			unsigned int idx_mid = idxList1[blockIdx.x];
			float histo_mid[bins];
			for(int i=0;i<bins;i++)
			{
				histo_mid[i] = histoMap1[idx_mid*bins+i];
			}

			for(int i=threadIdx.x;i<lx;i+=blockDim.x)
			{
				dist[i] = 0.f;
				idx[i] = 0;

				dist_tmp[i] = 0.f;
				idx_tmp[i] = 0;
			}

//			float int histo_cur[bins];
			for(int tx=0;tx<nList2;tx++)
			{

			}


		}
	};

	struct DeMeandCorrelationEstimator
	{

		float *histoMap;
		unsigned int bins;

		unsigned int *idxList;

	};

	template<unsigned int bins>
	struct SKCorrespondanceEstimator
	{
		enum
		{
			s = 32,
			k = 32,
		};

		float *histoMap1;
		float *histoMap2;

		unsigned int *idxList1;
		unsigned int *idxList2;
		unsigned int *s_rnd;

//		unsigned int k;
		unsigned int *infoList;

		unsigned int output_idx;
		float output_prob;

//		template<unsigned int bins>
//		__device__ __forceinline__ void euclidenHistoDist(int l,int g)
//		{
//
//		}

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_histo_local[s*bins];
			__shared__ float shm_global_buffer[k*bins];
			__shared__ unsigned int shm_global_idx[k];

			__shared__ float shm_dist_buffer[k*s];
			__shared__ float shm_dist_sorted[k*s];

			__shared__ unsigned int shm_idx_buffer[k*s];
			__shared__ unsigned int shm_idx_sorted[k*s];

//			__shared__ float shm_histo_local_tmp[s*bins];

			for(int i=threadIdx.x; i<s*bins; i+=blockDim.x)
			{
				unsigned int sl = i/bins;
				unsigned int idx = idxList1[blockIdx.x*s+sl];
				unsigned int b = i-sl*bins;

				shm_histo_local[sl + b*s] = histoMap1[idx*bins+b];

			}

//			for(int i=threadIdx.x; i<s ; i+=blockDim.x)
//			{
//				unsigned int idx = idxList1[blockIdx.x*s+i];
//
//				for(int b=0;b<bins;b++)
//				{
//					shm_histo_local[i+b*s] = histoMap1[idx*bins+b];
//				}
//			}
//
//			if(threadIdx.x==20)
//			{
//				printf("check shm \n");
//				for(int i=0;i<s*bins;i++)
//				{
////					if(shm_histo_local_tmp[i]!=shm_histo_local[i])
////						printf("fail! \n");
//
//					printf("%d -> %f = %f \n",shm_histo_local_tmp[i]==shm_histo_local[i],shm_histo_local[i],shm_histo_local_tmp[i]);
//				}
//			}

//			printf("wid: %d \n",wid);


			//TODO Check last iterations idx
//			for(int l=0;l<(infoList[1]-1)/k+1;l+=k)
			for(int l=0;l<(infoList[1])/k;l+=k)
//			for(int l=0;l<1;l+=k)
			{
				//Get the index of the k global points;
				for(int i=threadIdx.x;i<k;i++)
				{
					shm_global_idx[i] = idxList2[i+l*k];
				}
				__syncthreads();


				for(int i=threadIdx.x; i<k*bins; i+=blockDim.x)
				{
					unsigned int kl = i/bins;
					unsigned int idx = shm_global_idx[kl];
					unsigned int b = i-kl*bins;

					shm_global_buffer[kl + b*k] = histoMap2[idx*bins+b];
				}
				__syncthreads();

				for(int i=threadIdx.x; i<k*s; i+=blockDim.x)
				{
					unsigned int wl = i/s;
					shm_idx_buffer[i] = shm_global_idx[wl];
				}
				__syncthreads();

				if(threadIdx.x==0)
				{
					bool check = true;
					for(int kl=0;kl<k;kl++)
					{
						unsigned int idx = idxList2[kl+l*k];
						for(int b=0;b<bins;b++)
						{
							if(!(histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]))
								printf("%d -> %f = %f \n", (histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]),shm_global_buffer[kl+k*b],histoMap2[idx*bins+b]);

							check = check && (histoMap2[idx*bins+b]==shm_global_buffer[kl+k*b]);
						}

					}
					printf("check global buffer: %d  \n",check);

//					for(int kl=0;kl<k;kl++)
//					{
//						for(int sl=0;sl<s;sl++)
//						{
//							printf("%d | ",shm_idx_buffer[kl*s+sl]);
//						}
//						printf(" \n");
//					}

				}
				__syncthreads();


				unsigned int wid = threadIdx.x/32;
				unsigned int tid = threadIdx.x - wid*32;

				float div = 0.f;
				for(int b=0;b<bins;b++)
				{
					float d = shm_histo_local[tid+s*b] - shm_global_buffer[wid+k*b];
					div += d*d;
				}
				div = sqrtf(div);
				shm_dist_buffer[wid*s+tid] = div;

			}

			//Bitonic sort


		}


	};
	__global__ void estimateCorrespondance(const SKCorrespondanceEstimator<8> ce) {ce (); }
}

device::SKCorrespondanceEstimator<8> correspondanceEstimator;

void RigidBodyTransformationEstimator::init()
{
	size_t n = 100;
	curandGenerator_t gen ;
	float *d_rnd_data;

	checkCudaErrors(cudaMalloc((void**)&d_rnd_data,n*sizeof(float)));
	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));

	checkCudaErrors(curandGenerateUniform(gen,d_rnd_data,n));

	correspondanceEstimator.histoMap1 = (float *)getInputDataPointer(HistogramMap);
	correspondanceEstimator.idxList1 = (unsigned int *)getInputDataPointer(IdxList);

	correspondanceEstimator.histoMap2 = (float *)getInputDataPointer(HistogramMap);
	correspondanceEstimator.idxList2 = (unsigned int *)getInputDataPointer(IdxList);
	correspondanceEstimator.infoList = (unsigned int *)getInputDataPointer(InfoList);

	block = dim3(correspondanceEstimator.s*correspondanceEstimator.k,1);
	grid = dim3(1,1);
}

void RigidBodyTransformationEstimator::execute()
{
//	printf("hmmm \n");
//	SVDEstimator_CPU *svd = new SVDEstimator_CPU();
//	svd->execute();

	printf("d_infoList: %d \n",correspondanceEstimator.infoList);

	device::estimateCorrespondance<<<1,block>>>(correspondanceEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}


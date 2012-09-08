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

			float int histo_cur[bins];
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
}


void RigidBodyTransformationEstimator::init()
{
	size_t n = 100;
	curandGenerator_t gen ;
	float *d_rnd_data;

	checkCudaErrors(cudaMalloc((void**)&d_rnd_data,n*sizeof(float)));
	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));

	checkCudaErrors(curandGenerateUniform(gen,d_rnd_data,n));

//	float *h_data = (float *)malloc(n*sizeof(float));
//	checkCudaErrors(cudaMemcpy(h_data,d_rnd_data,n*sizeof(float),cudaMemcpyDeviceToHost));
//
//	for(int i=0;i<n;i++)
//		printf("%f \n",h_data[i]);


}

void RigidBodyTransformationEstimator::execute()
{
//	printf("hmmm \n");
//	SVDEstimator_CPU *svd = new SVDEstimator_CPU();
//	svd->execute();
}


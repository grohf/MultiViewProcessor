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


#define CURAND_CALL ( x ) do { if (( x ) != CURAND_STATUS_SUCCESS ) { \
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


	struct DeMeanedCorrelationMatrixEstimator
	{
		enum
		{
			n = 32,
			warps = 32,
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

		float *output_correlationMatrixes;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float ps_x[n*warps];
			__shared__ float ps_y[n*warps];
			__shared__ float ps_z[n*warps];

			__shared__ float pt_x[n*warps];
			__shared__ float pt_y[n*warps];
			__shared__ float pt_z[n*warps];




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

		float *histoMap1;
		float *histoMap2;

		unsigned int *idxList1;
		unsigned int *idxList2;

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
				unsigned int idx = idxList1[(unsigned int)(f*infoList[view_src+1])];
				shm_local_idx[threadIdx.x] = idx;
				output_s_idx[blockIdx.x*sp+threadIdx.x] = idx;

				if(threadIdx.x==10)
					printf("rnd: %f idx_pos: %d idx: %d \n",f,(unsigned int)(f*infoList[view_src+1]),idx);

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


				shm_local_histo[sl + b*sp] = histoMap1[shm_local_idx[sl]*bins+b];
			}


			//TODO Check last iterations idx
//			for(int l=0;l<(infoList[1]-1)/k+1;l+=k)
			for(int l=0;l<(infoList[view_target+1])/k;l+=k)
//			for(int l=0;l<3;l++)
			{

				//Get the index of the k global points;
				for(int i=threadIdx.x;i<k;i+=blockDim.x)
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
}

device::SKCorrespondanceEstimator<8> correspondanceEstimator;
device::DeMeanedCorrelationMatrixEstimator deMeanedCorrelatonMEstimator;

void RigidBodyTransformationEstimator::init()
{

	checkCudaErrors(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));


	correspondanceEstimator.histoMap1 = (float *)getInputDataPointer(HistogramMap);
	correspondanceEstimator.idxList1 = (unsigned int *)getInputDataPointer(IdxList);

	correspondanceEstimator.histoMap2 = (float *)getInputDataPointer(HistogramMap);
	correspondanceEstimator.idxList2 = (unsigned int *)getInputDataPointer(IdxList);

	correspondanceEstimator.infoList = (unsigned int *)getInputDataPointer(InfoList);

	correspondanceEstimator.rndList = (float *)getTargetDataPointer(RndIndices);

	correspondanceEstimator.output_prob = (float *)getTargetDataPointer(ProbList);
	correspondanceEstimator.output_idx = (unsigned int *)getTargetDataPointer(ProbIdxList);
	correspondanceEstimator.output_s_idx = (unsigned int *)getTargetDataPointer(SIndices);



	block = dim3(correspondanceEstimator.sp*correspondanceEstimator.k,1);
	grid = dim3(s/correspondanceEstimator.sp,1);

	deMeanedCorrelatonMEstimator.pos = (float4 *)getTargetDataPointer(Coordiantes);

	deMeanedCorrelatonMEstimator.correspondanceErrorIdxList = (unsigned int *)getTargetDataPointer(ProbIdxList);
	deMeanedCorrelatonMEstimator.correspondanceErrorList = (float *)getTargetDataPointer(ProbList);
	deMeanedCorrelatonMEstimator.correspondanceIdxList = (unsigned int *)getTargetDataPointer(SIndices);

	deMeanedCorrelatonMEstimator.n_src = s;
	deMeanedCorrelatonMEstimator.n_target = k;
	deMeanedCorrelatonMEstimator.rnd_src = (float *)getTargetDataPointer(RndIndices);
	deMeanedCorrelatonMEstimator.rnd_target = (float *)getTargetDataPointer(RndIndices2);

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
	correspondanceEstimator.view_target = 0;
	device::estimateCorrespondance<<<grid,block>>>(correspondanceEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



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


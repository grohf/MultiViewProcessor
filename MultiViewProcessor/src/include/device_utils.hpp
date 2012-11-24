/*
 * device_utils.hpp
 *
 *  Created on: Nov 8, 2012
 *      Author: avo
 */

#ifndef DEVICE_UTILS_HPP_
#define DEVICE_UTILS_HPP_

namespace device
{


template<unsigned int blockSize, typename T>
__device__ __forceinline__ void reduceBlock(T* shm)
{

	  unsigned int tid = threadIdx.x;
	  float sum = shm[tid];
	  if (blockSize >= 1024) { if(tid < 512) 	{ shm[tid] = sum = sum + shm[tid + 512]; 	} __syncthreads(); }
	  if (blockSize >=  512) { if(tid < 256) 	{ shm[tid] = sum = sum + shm[tid + 256]; 	} __syncthreads(); }
	  if (blockSize >=  256) { if(tid < 128) 	{ shm[tid] = sum = sum + shm[tid + 128]; 	} __syncthreads(); }
	  if (blockSize >=  128) { if(tid < 64) 	{ shm[tid] = sum = sum + shm[tid + 64]; 	} __syncthreads(); }

	  if(tid < 32)
	  {
		  volatile T* smem = shm;
		  if (blockSize >=  64) { smem[tid] = sum = sum + smem[tid + 32]; 	}
		  if (blockSize >=  32) { smem[tid] = sum = sum + smem[tid + 16]; 	}
		  if (blockSize >=  16) { smem[tid] = sum = sum + smem[tid +  8]; 	}
		  if (blockSize >=   8) { smem[tid] = sum = sum + smem[tid +  4]; 	}
		  if (blockSize >=   4) { smem[tid] = sum = sum + smem[tid +  2]; 	}
		  if (blockSize >=   2) { smem[tid] = sum = sum + smem[tid +  1]; 	}
	  }
}

template<unsigned int blockSize, typename T>
__device__ __forceinline__ void reduceBlockNoReg(T* shm)
{

	  unsigned int tid = threadIdx.x;
	  if (blockSize >= 1024) { if(tid < 512) 	{ shm[tid] += shm[tid + 512]; 	} __syncthreads(); }
	  if (blockSize >=  512) { if(tid < 256) 	{ shm[tid] += shm[tid + 256]; 	} __syncthreads(); }
	  if (blockSize >=  256) { if(tid < 128) 	{ shm[tid] += shm[tid + 128]; 	} __syncthreads(); }
	  if (blockSize >=  128) { if(tid < 64) 	{ shm[tid] += shm[tid + 64]; 	} __syncthreads(); }

	  if(tid < 32)
	  {
		  volatile T* smem = shm;
		  if (blockSize >=  64) { smem[tid] += smem[tid + 32]; 	}
		  if (blockSize >=  32) { smem[tid] += smem[tid + 16]; 	}
		  if (blockSize >=  16) { smem[tid] += smem[tid +  8]; 	}
		  if (blockSize >=   8) { smem[tid] += smem[tid +  4]; 	}
		  if (blockSize >=   4) { smem[tid] += smem[tid +  2]; 	}
		  if (blockSize >=   2) { smem[tid] += smem[tid +  1]; 	}
	  }
}

template<unsigned int blockSize, typename T>
__device__ __forceinline__ void reduceBlockNoReg(T* shm,int lines)
{

	  unsigned int tid = threadIdx.x;
	  if (blockSize >= 1024) { if(tid < 512) 	{ for(int i=0;i<lines;i++){ shm[i*blockSize + tid] += shm[i*blockSize + tid + 512]; } 	} __syncthreads(); }
	  if (blockSize >=  512) { if(tid < 256) 	{ for(int i=0;i<lines;i++){ shm[i*blockSize + tid] += shm[i*blockSize + tid + 256]; } 	} __syncthreads(); }
	  if (blockSize >=  256) { if(tid < 128) 	{ for(int i=0;i<lines;i++){ shm[i*blockSize + tid] += shm[i*blockSize + tid + 128]; }	} __syncthreads(); }
	  if (blockSize >=  128) { if(tid < 64) 	{ for(int i=0;i<lines;i++){ shm[i*blockSize + tid] += shm[i*blockSize + tid +  64]; } 	} __syncthreads(); }

	  if(tid < 32)
	  {
		  volatile T* smem = shm;
		  if (blockSize >=  64) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid + 32]; } 	}
		  if (blockSize >=  32) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid + 16]; } 	}
		  if (blockSize >=  16) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid +  8]; }	}
		  if (blockSize >=   8) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid +  4]; } 	}
		  if (blockSize >=   4) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid +  2]; } 	}
		  if (blockSize >=   2) { for(int i=0;i<lines;i++){ smem[i*blockSize + tid] += smem[i*blockSize + tid +  1]; } 	}
	  }
}


template<class It>
__device__ __forceinline__ float3 fetch(It ptr, int index)
{
    //return tr(ptr[index]);
    return *(float3*)&ptr[index];
}

template<class It>
__device__ __forceinline__ float3 fetch(It &ptr)
{
    //return tr(ptr[index]);
    return *(float3*)&ptr;
}

__device__  __forceinline__ void compareAndSwap(float *a, float *b, unsigned int *ai, unsigned int *bi,unsigned int dir)
{
	if( (a[0] > b[0]) == dir )
	{
		float tmp = b[0];
		b[0] = a[0];
		a[0] = tmp;

		unsigned int tmpIdx = bi[0];
		bi[0] = ai[0];
		ai[0] = tmpIdx;
	}
}

__device__  __forceinline__ void compareAndSwap(float &a, float &b, unsigned int &ai, unsigned int &bi,unsigned int dir)
{
	if( (a > b) == dir )
	{
		float tmp = b;
		b = a;
		a = tmp;

		unsigned int tmpIdx = bi;
		bi = ai;
		ai = tmpIdx;
	}
}



__device__  __forceinline__ float
klDivergence(float *feature, float *mean, unsigned int bins_count, unsigned int offset_feature, unsigned int offset_mean)
{

	float div = 0.f;
	for(int i=0;i<bins_count;i++)
	{
		float p = feature[i*offset_feature];
		float m = mean[i*offset_mean];

		if(p/m>0)
			div += (p - m) * __logf(p/m);
	}

	return div;
}

__device__  __forceinline__ float
klEuclideanDivergence(float *feature_P, float *featur_Q, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_P, unsigned int offset_Q)
{
	float div = 0.f;

	for(int f=0;f<feature_count;f++)
	{
		float tmpDiv = 0.f;
		for(int i=0;i<bin_count_per_feature;i++)
		{
			float a = feature_P[(f*bin_count_per_feature+i)*offset_P];
			float b = featur_Q[(f*bin_count_per_feature+i)*offset_Q];

			if( !(a==0 && b==0) && a/b>0)
			{
//				tmpDiv += (a - b) * __logf(a/b);
				tmpDiv += (a - b) * logf(a/b);
			}
		}
		div += (tmpDiv * tmpDiv);
	}

	return sqrtf(div);
}

__device__  __forceinline__ float
divergence(float *feature_P, float *featur_Q, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_P, unsigned int offset_Q)
{
	float div = 0.f;


	return div;
}

__device__  __forceinline__ float
chiSquaredDivergence(float *feature_P, float *featur_Q, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_P, unsigned int offset_Q)
{
	float div = 0.f;

	for(int f=0;f<feature_count*bin_count_per_feature;f++)
	{
		float a = feature_P[f*offset_P];
		float b = featur_Q[f*offset_Q];

		if(a+b != 0)
		{
			div += ((a-b)*(a-b)) / (a+b) ;
		}
	}

	return div;
}

__device__  __forceinline__ float
chiSquaredEuclideanDivergence(float *feature_P, float *featur_Q, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_P, unsigned int offset_Q)
{
	float div = 0.f;


	for(int f=0;f<feature_count;f++)
	{
		float tmpDiv = 0.f;
		for(int i=0;i<bin_count_per_feature;i++)
		{
			float a = feature_P[(f*bin_count_per_feature+i)*offset_P];
			float b = featur_Q[(f*bin_count_per_feature+i)*offset_Q];

			if(a+b != 0)
			{
				div += ((a-b)*(a-b)) / (a+b) ;
			}
		}
		div += (tmpDiv * tmpDiv);
	}

	return sqrtf(div);
}


template <typename DivType>
class Div
{
	enum Norm
	{
		ChiSquared,
		KL,
		L2,
	};

public:
	__device__ __forceinline__ void operator() (float *feature_P, float *featur_Q, unsigned int feature_count, unsigned int bin_count_per_feature, unsigned int offset_P, unsigned int offset_Q)
	{

	}

};



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


#endif /* DEVICE_UTILS_HPP_ */

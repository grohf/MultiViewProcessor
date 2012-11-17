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

class Div
{
public:
	__device__ __forceinline__ void operator() (float *a, float *b,unsigned int off)
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

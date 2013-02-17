/*
 * HistogramThresholdSegmentation.cpp
 *
 *  Created on: Dec 2, 2012
 *      Author: avo
 */

#include "HistogramThresholdSegmentation.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>

#include "point_info.hpp"
#include "device_utils.hpp"

template <typename T>
struct threshold_stats_data
{
    T n1;
    T n2;
//    T min;
//    T max;
    T mean1;
    T mean2;

    T M1;
    T M2;

//    T M3;
//    T M4;

    // initialize to the identity element
    void initialize()
    {
      n1 = n2 = mean1 = mean2 = 0;
      M1 = M2 = 0;
//      min = std::numeric_limits<T>::max();
//      max = std::numeric_limits<T>::min();
    }

    T threshold()   { return 0.5 * (mean2 - mean1); }
    T threshold_var()   { return 0.5 * (M1/(n1-1))/(M2/(n2-1)) * (mean2 - mean1); }

//    T variance()   { return M2 / (n - 1); }
//    T variance_n() { return M2 / n; }
//    T skewness()   { return std::sqrt(n) * M3 / std::pow(M2, (T) 1.5); }
//    T kurtosis()   { return n * M4 / (M2 * M2); }
};

template <typename T>
struct threshold_stats_unary_op
{
	float thresh;

    __host__ __device__
    threshold_stats_data<T> operator()(const float4 & p) const
    {
         threshold_stats_data<T> result;

         result.M1 = 0;
         result.M2 = 0;

         if(p.z==0)
         {
 			result.n1    = 0;
 			result.n2    = 0;

          	result.mean1 = 0;
          	result.mean2 = 0;

          	return result;
         }

         if(p.z<=thresh)
         {
			result.n1    = 1;
			result.n2    = 0;

         	result.mean1 = p.z;
         	result.mean2 = 0;
         }
         else
         {
        	 result.n1    = 0;
			 result.n2    = 1;

			result.mean1 = 0;
			result.mean2 = p.z;
         }
//         result.n1    = (p.z>thresh)?1:0;
//         result.min  = p.z;
//         result.max  = p.z;
//         result.mean = p.z;
//         result.M2   = 0;
//         result.M3   = 0;
//         result.M4   = 0;

         return result;
    }
};

template <typename T>
struct threshold_stats_binary_op
    : public thrust::binary_function<const threshold_stats_data<T>&,
                                     const threshold_stats_data<T>&,
                                           threshold_stats_data<T> >
{
    __host__ __device__
    threshold_stats_data<T> operator()(const threshold_stats_data<T>& x, const threshold_stats_data <T>& y) const
    {
        threshold_stats_data<T> result;

        // precompute some common subexpressions
        T n1  = x.n1 + y.n1;
        T n2  = x.n2 + y.n2;

//        T n2 = n  * n;
//        T n3 = n2 * n;

        T delta1  = y.mean1 - x.mean1;
        T delta2  = y.mean2 - x.mean2;
//        T delta2 = delta  * delta;
//        T delta3 = delta2 * delta;
//        T delta4 = delta3 * delta;

        //Basic number of samples (n), min, and max
        result.n1   = n1;
        result.n2   = n2;

//        result.min = thrust::min(x.min, y.min);
//        result.max = thrust::max(x.max, y.max);

        result.mean1 =(n1>0)?(x.mean1 + delta1 * y.n1 / n1):0;
        result.mean2 =(n2>0)?(x.mean2 + delta2 * y.n2 / n2):0;

        result.M1  = (n1>0)?x.M1 + y.M1:0;
        result.M2  = (n2>0)?x.M2 + y.M2:0;

//        result.M2  = x.M2 + y.M2;
//        result.M2 += delta2 * x.n * y.n / n;
//
//        result.M3  = x.M3 + y.M3;
//        result.M3 += delta3 * x.n * y.n * (x.n - y.n) / n2;
//        result.M3 += (T) 3.0 * delta * (x.n * y.M2 - y.n * x.M2) / n;
//
//        result.M4  = x.M4 + y.M4;
//        result.M4 += delta4 * x.n * y.n * (x.n * x.n - x.n * y.n + y.n * y.n) / n3;
//        result.M4 += (T) 6.0 * delta2 * (x.n * x.n * y.M2 + y.n * y.n * x.M2) / n2;
//        result.M4 += (T) 4.0 * delta * (x.n * y.M3 - y.n * x.M3) / n;

        return result;
    }
};



namespace device
{
	struct ThresholdFilter
	{
		enum
		{
			Min = 300,
		};

		float4* pos;
		float *max;

		__device__ __forceinline__ void
		operator () () const
		{
			int sx = blockIdx.x*blockDim.x+threadIdx.x;
			int sy = blockIdx.y*blockDim.y+threadIdx.y;

			int off = blockIdx.z*640*480+sy*640+sx;

			float4 wc = pos[off];
			SegmentationPointInfo spi;
			spi = TBD;
			if(wc.z > Min)
			{
				spi = (wc.z > max[blockIdx.z])?Background:Foreground;
			}
			device::setSegmentationPointInfo(wc.w,spi);
			pos[off] = wc;
		}

	};
	__global__ void filterTruncateThreshold(ThresholdFilter tt) { tt (); }

}
device::ThresholdFilter threshFilter;

void
HistogramThresholdSegmentation::init()
{
	threshFilter.pos = (float4 *)getInputDataPointer(0);
}

void
HistogramThresholdSegmentation::execute()
{
	thrust::device_ptr<float4> d_ptr = thrust::device_pointer_cast((float4 *)getInputDataPointer(WorldCoordinates));
	thrust::device_vector<float4> d_pos(d_ptr,d_ptr+n_view*640*480);

//    thrust::host_vector<float4> h_p(10);
//    for(int i=0;i<h_p.size();i++)
//    	h_p[i].z = i;

	thrust::device_vector<float> d_max(n_view);
    for(int v=0;v<n_view;v++)
    {
		threshold_stats_unary_op<float>  unary_op;
		threshold_stats_binary_op<float> binary_op;
		threshold_stats_data<float>      init;
		init.initialize();
		unary_op.thresh = 5000.f;
		threshold_stats_data<float> result;

		float old_tresh = -50.f;
//		unsigned int c = 0;
		while(abs(result.threshold()-old_tresh)>5.0f)
		{
//			printf("round: %d \n",c++,result.threshold());
			old_tresh = unary_op.thresh;
			result = thrust::transform_reduce(d_pos.data()+v*640*480, d_pos.data()+(v+1)*640*480,unary_op, init, binary_op);
			unary_op.thresh = result.threshold();
			unary_op.thresh = result.threshold_var();
//	//        printf("round: %d \n",c++,result.threshold());
//			std::cout <<"n1              : "<< result.n1 << std::endl;
//			std::cout <<"n2             : "<< result.n2 << std::endl;
//			std::cout <<"mean1              : "<< result.mean1 << std::endl;
//			std::cout <<"mean2             : "<< result.mean2 << std::endl;
//			std::cout <<"new Threshold              : "<< result.threshold() << std::endl;
		}

		if(outputlevel>0)
			printf("view: %d thresh: %f \n",v,result.threshold());
		d_max[v] = result.threshold();
    }

    threshFilter.max = thrust::raw_pointer_cast(d_max.data());

	dim3 block = dim3(32,24);
	dim3 grid = dim3(640/block.x,480/block.y,n_view);
	device::filterTruncateThreshold<<<grid,block>>>(threshFilter);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if(outputlevel>1)
	{
		size_t uc4s = 640*480*sizeof(uchar4);
		char path[50];
		for(int v=0;v<n_view;v++)
		{
			float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
			checkCudaErrors(cudaMemcpy(h_f4_depth,threshFilter.pos+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

		//	uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
		//	for(int i=0;i<640*480;i++)
		//	{
		//		unsigned char g = h_f4_depth[i].z/20;
		//		h_uc4_depth[i] = make_uchar4(g,g,g,128);
		//
		//		if(!device::isValid(h_f4_depth[i].w)) h_uc4_depth[i].x = 255;
		//
		//		if(device::isReconstructed(h_f4_depth[i].w)) h_uc4_depth[i].y = 255;
		//	}
		//
		//	sprintf(path,"/home/avo/pcds/src_depth_valid_map%d.ppm",0);
		//	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);



			uchar4 *h_uc4_depth2 = (uchar4 *)malloc(uc4s);
			for(int i=0;i<640*480;i++)
			{
				unsigned char g = h_f4_depth[i].z/20;
				h_uc4_depth2[i] = make_uchar4(g,g,g,128);

				if(device::isForeground(h_f4_depth[i].w)) h_uc4_depth2[i].x = 255;

				if(device::isBackground(h_f4_depth[i].w)) h_uc4_depth2[i].y = 255;

				if(!device::isSegmented(h_f4_depth[i].w)) h_uc4_depth2[i].z = 255;
			}

			sprintf(path,"/home/avo/pcds/threshold_map%d.ppm",v);
			sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth2,640,480);
		}
	}
}

HistogramThresholdSegmentation::HistogramThresholdSegmentation(unsigned int n_view,unsigned int outputlevel) : n_view(n_view), outputlevel(outputlevel)
{
	// TODO Auto-generated constructor stub

}

HistogramThresholdSegmentation::~HistogramThresholdSegmentation()
{
	// TODO Auto-generated destructor stub
}


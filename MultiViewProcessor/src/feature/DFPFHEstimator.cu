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

	struct SDPFHEstimator
	{
		enum
		{
			points_per_block = 32,
			WARP_SIZE = 32,
			dx = points_per_block * WARP_SIZE,
			dy = 1,
			features = 3,
			bins_per_feature = 11,
			bins = features * bins_per_feature,

			rx = 2,
			ry = rx,
			lx = 2*rx+1,
			ly = 2*ry+1,

		};

		float4 *input_pos;
		float4 *input_normals;

		float *output_bins;

		unsigned int view;
		float radius;


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

				shm_pos[tid] = input_pos[view*640*480+oy*640+ox];
				shm_normal[tid] = input_normals[view*640*480+oy*640+ox];
			}

			for(int i=tid;i<bins*points_per_block;i+=dx)
			{
				shm_histo[i] = 0.f;
			}
			__syncthreads();

			if(shm_pos[wid].z==0 || isBackground(shm_pos[wid].w))
			{
				if(wtid==0)
					shm_histo[wid] = -1.f;

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
				oy = j/ly;
				ox = j - oy*ly;

				oy = shm_off[wid] - ry + oy;
				ox = shm_off[points_per_block+wid] - rx + ox;

				if(oy<0 || oy>=480 || ox <0 || ox>=640 || j==(lx*ly)/2)
					continue;

				float4 p = input_pos[view*640*480+oy*640+ox];
				float4 n = input_normals[view*640*480+oy*640+ox];

				if(!isValid(p.w))
				{
					invalid_points++;
					continue;
				}

//				norm(*(float3 *)&p);

				if(lengthf43(minusf43(shm_pos[wid],p))>radius || n.w<0)
				{
					continue;
				}

				point_count++;

				float4 ps,pt,ns,nt;
				if( dotf43(shm_normal[wid], minusf43(p,shm_pos[wid]) ) <= dotf43(n, minusf43(shm_pos[wid],p) ) )
				{
					ps = shm_pos[wid];
					ns = shm_normal[wid];

					pt = p;
					nt = n;
				}
				else
				{
					ps = p;
					ns = n;

					pt = shm_pos[wid];
					nt = shm_normal[wid];
				}

				float4 u = ns;
				float4 v = crossf43(minusf43(pt,ps),u);
				float4 w = crossf43(u,v);

				int idx = 0;
				float f = dotf43(v,nt);
				idx = floorf (bins_per_feature * ((f + 1.0f) * 0.5f));
				idx = min(bins_per_feature - 1, max(0, idx))*points_per_block + wid;
				atomicAdd(&shm_histo[idx], 1.f);

				f = dotf43(u,minusf43(pt,ps))/lengthf43( minusf43(pt,ps));
				idx = floorf (bins_per_feature * ((f + 1.0f) * 0.5f));
				idx = min(bins_per_feature - 1, max(0, idx))*points_per_block + wid;
				atomicAdd(&shm_histo[idx], 1.f);

				f = atan2f(dotf43(w,nt),dotf43(u,nt));
				idx = floorf (bins_per_feature * ((f + M_PI) * (1.0f / (2.0f * M_PI))));
				idx = min(bins_per_feature - 1, max(0, idx))*points_per_block + wid;
				atomicAdd(&shm_histo[idx], 1.f);



			}




		}
	};
	__global__ void computeSDPFH(const SDPFHEstimator sdpfhe){ sdpfhe(); }


}


device::SDPFHEstimator sdpfhEstimator1;

void
DFPFHEstimator::init()
{

}


void
DFPFHEstimator::execute()
{

}



DFPFHEstimator::DFPFHEstimator(unsigned int n_view): n_view(n_view)
{
	// TODO Auto-generated constructor stub

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
		h_inp_normals[i] = make_float4(1,0,0,0);
	}

	thrust::device_vector<float4> d_inp_pos = h_inp_pos;
	thrust::device_vector<float4> d_inp_normal = h_inp_normals;
	thrust::device_vector<float> d_out_bins(n_view*640*480*3*11);

	sdpfhEstimator1.input_pos = thrust::raw_pointer_cast(d_inp_pos.data());
	sdpfhEstimator1.input_normals = thrust::raw_pointer_cast(d_inp_normal.data());
	sdpfhEstimator1.output_bins = thrust::raw_pointer_cast(d_out_bins.data());
	sdpfhEstimator1.view = 0;
	sdpfhEstimator1.radius = 15.0f;

	dim3 block(sdpfhEstimator1.dx);
	dim3 grid(640*480/sdpfhEstimator1.points_per_block,1,1);
	device::computeSDPFH<<<grid,block>>>(sdpfhEstimator1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

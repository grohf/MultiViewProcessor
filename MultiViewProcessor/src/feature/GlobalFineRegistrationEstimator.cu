/*
 * GlobalFineRegistrationEstimator.cpp
 *
 *  Created on: Dec 7, 2012
 *      Author: avo
 */

#include "GlobalFineRegistrationEstimator.h"

#include "point_info.hpp"
#include "utils.hpp"
//#include "device_thrust_utils.hpp"

#include <helper_cuda.h>
#include <helper_image.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <thrust/remove.h>
#include <thrust/scan.h>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "math.h"

namespace device
{
	struct GlobalFineRegistrationBaseKernel : public FeatureBaseKernel
	{
		enum
		{
			ForgroundOnly = 1,
			Invalid = -1,
		};
	};

	template<typename T>
	struct invalid_binary : public thrust::binary_function<T,T,T>, public GlobalFineRegistrationBaseKernel
	{
		__device__
		T operator()(T x, T y)
		{
			if(y==Invalid)
				return x;

			return x+1;
		}
	};

	struct SparseCbCreater : public GlobalFineRegistrationBaseKernel
	{
		enum
		{
			dx = 32,
			dy = 32,

		};

		float4 *input_pos;
		float4 *input_normal;
		float *input_tm;
		SensorInfoV2 *input_sinfo;

		int *output_sparseAInfo;
		float *output_sparseAb;
//		float *output_b;

		unsigned int n_view;
		unsigned int view_combinations;

		float dist_threshold;
		float angle_threshold;

		__device__ __forceinline__ void
		operator () () const
		{
			__shared__ float shm_tm_src[TMatrixDim];
			__shared__ float shm_tm_target[TMatrixDim];

			__shared__ float shm_tm[TMatrixDim];
			unsigned int tid = threadIdx.y*blockDim.x+threadIdx.x;

			__shared__ unsigned int view_src;
			__shared__ unsigned int view_target;

			__shared__ SensorInfoV2 sinfo;

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

				sinfo = input_sinfo[view_target];
			}
			__syncthreads();

			if(tid<TMatrixDim)
			{
				shm_tm_src[tid] = input_tm[view_src*TMatrixDim+tid];
				shm_tm_target[tid] = input_tm[view_target*TMatrixDim+tid];
			}
			__syncthreads();

			if(tid==0)
			{
//				unsigned int i = (tid/3)*3;
//				unsigned int j = tid - i;
//				shm_tm[i] = shm_tm_target[i] * shm_tm_src[j] + shm_tm_target[i+1] * shm_tm_src[j+3] + shm_tm_target[i+2] * shm_tm_src[j+6];

				shm_tm[0] = shm_tm_target[0] * shm_tm_src[0] + shm_tm_target[3] * shm_tm_src[3] + shm_tm_target[6] * shm_tm_src[6];
				shm_tm[1] = shm_tm_target[0] * shm_tm_src[1] + shm_tm_target[3] * shm_tm_src[4] + shm_tm_target[6] * shm_tm_src[7];
				shm_tm[2] = shm_tm_target[0] * shm_tm_src[2] + shm_tm_target[3] * shm_tm_src[5] + shm_tm_target[6] * shm_tm_src[8];

				shm_tm[3] = shm_tm_target[1] * shm_tm_src[0] + shm_tm_target[4] * shm_tm_src[3] + shm_tm_target[7] * shm_tm_src[6];
				shm_tm[4] = shm_tm_target[1] * shm_tm_src[1] + shm_tm_target[4] * shm_tm_src[4] + shm_tm_target[7] * shm_tm_src[7];
				shm_tm[5] = shm_tm_target[1] * shm_tm_src[2] + shm_tm_target[4] * shm_tm_src[5] + shm_tm_target[7] * shm_tm_src[8];

				shm_tm[6] = shm_tm_target[2] * shm_tm_src[0] + shm_tm_target[5] * shm_tm_src[3] + shm_tm_target[8] * shm_tm_src[6];
				shm_tm[7] = shm_tm_target[2] * shm_tm_src[1] + shm_tm_target[5] * shm_tm_src[4] + shm_tm_target[8] * shm_tm_src[7];
				shm_tm[8] = shm_tm_target[2] * shm_tm_src[2] + shm_tm_target[5] * shm_tm_src[5] + shm_tm_target[8] * shm_tm_src[8];


				shm_tm[9] 	= shm_tm_target[0] * (shm_tm_src[9] - shm_tm_target[9]) + shm_tm_target[3] * (shm_tm_src[10] - shm_tm_target[10]) + shm_tm_target[6] * (shm_tm_src[11] - shm_tm_target[11]);
				shm_tm[10] 	= shm_tm_target[1] * (shm_tm_src[9] - shm_tm_target[9]) + shm_tm_target[4] * (shm_tm_src[10] - shm_tm_target[10]) + shm_tm_target[7] * (shm_tm_src[11] - shm_tm_target[11]);
				shm_tm[11] 	= shm_tm_target[2] * (shm_tm_src[9] - shm_tm_target[9]) + shm_tm_target[5] * (shm_tm_src[10] - shm_tm_target[10]) + shm_tm_target[8] * (shm_tm_src[11] - shm_tm_target[11]);

			}
			__syncthreads();

			if(blockIdx.x==0 && blockIdx.y==0 && tid==0)
			{
				for(int i=0;i<12;i++)
					printf("%f | ",shm_tm[i]);

				printf("\n");
			}

			unsigned int gid = (blockIdx.y*blockDim.y+threadIdx.y)*640+blockIdx.x*blockDim.x+threadIdx.x;

			float4 pos = input_pos[view_src*640*480+gid];
			float4 normal = input_normal[view_src*640*480+gid];

			gid += blockIdx.z*640*480;
			bool valid = false;

			if(pos.z !=0 && (!ForgroundOnly || isForeground(pos.w)) && isValid(pos.w) && !isReconstructed(pos.w))
			{
				float3 pos_m,n_m;
				pos_m.x = shm_tm[0]*pos.x + shm_tm[1]*pos.y + shm_tm[2]*pos.z + shm_tm[9];
				pos_m.y = shm_tm[3]*pos.x + shm_tm[4]*pos.y + shm_tm[5]*pos.z + shm_tm[10];
				pos_m.x = shm_tm[6]*pos.x + shm_tm[7]*pos.y + shm_tm[8]*pos.z + shm_tm[11];

				n_m.x = shm_tm[0]*normal.x + shm_tm[1]*normal.y + shm_tm[2]*normal.z + shm_tm[9];
				n_m.y = shm_tm[3]*normal.x + shm_tm[4]*normal.y + shm_tm[5]*normal.z + shm_tm[10];
				n_m.x = shm_tm[6]*normal.x + shm_tm[7]*normal.y + shm_tm[8]*normal.z + shm_tm[11];

				int ix = (int) ((pos_m.x*sinfo.fx)/pos_m.z + sinfo.cx);
				int iy = (int) ((pos_m.y*sinfo.fy)/pos_m.z + sinfo.cy);

				if(ix>=0 && ix<640 && iy>=0 && iy<480)
				{
					float3 pos_d = fetchToFloat3(input_pos[view_target*640*480+iy*640+ix]);
					float3 n_d = fetchToFloat3(input_normal[view_target*640*480+iy*640+ix]);

					float3 dv = pos_m - pos_d;
					float angle = n_m.x * n_d.x + n_m.y * n_d.y + n_m.z * n_d.z;

					if( norm(dv) < dist_threshold && ( angle >= angle_threshold) )
					{
						valid = true;
						float3 rp = cross(n_m,pos_d);

						output_sparseAb[0*view_combinations*640*480 + gid] = rp.x;
						output_sparseAb[1*view_combinations*640*480 + gid] = rp.y;
						output_sparseAb[2*view_combinations*640*480 + gid] = rp.z;

						output_sparseAb[3*view_combinations*640*480 + gid] = -n_m.x;
						output_sparseAb[4*view_combinations*640*480 + gid] = -n_m.y;
						output_sparseAb[5*view_combinations*640*480 + gid] = -n_m.z;

//						output_sparseAb[6*view_combinations*640*480 + blockIdx.z*640*480+gid] = -rp.x;
//						output_sparseAb[7*view_combinations*640*480 + blockIdx.z*640*480+gid] = -rp.y;
//						output_sparseAb[8*view_combinations*640*480 + blockIdx.z*640*480+gid] = -rp.z;
//
//						output_sparseAb[9*view_combinations*640*480 + blockIdx.z*640*480+gid] = n_m.x;
//						output_sparseAb[10*view_combinations*640*480 + blockIdx.z*640*480+gid] = n_m.y;
//						output_sparseAb[11*view_combinations*640*480 + blockIdx.z*640*480+gid] = n_m.z;

						output_sparseAb[12*view_combinations*640*480 + gid] = dot(n_m,dv);
					}

				}
			}

			output_sparseAInfo[gid] = (valid)?(int)gid:Invalid;
			if(valid)
			{
				output_sparseAInfo[1*view_combinations*640*480 + gid] = view_src;
				output_sparseAInfo[2*view_combinations*640*480 + gid] = view_target;
			}

		}
	};
	__global__ void computeSparseCb(const SparseCbCreater Cb) { Cb (); }


	struct CompressIdxList : public GlobalFineRegistrationBaseKernel
	{
		enum
		{
			dx = 1024,
		};

		int *input_sparseInfo;
		int *input_prefixSum;
		unsigned int *output_compressedIdxList;

		unsigned int length;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
			if(gid>=length)
				return;

			int id = input_sparseInfo[gid];
			if(id==Invalid)
				return;

			unsigned int idx = input_prefixSum[gid];
			output_compressedIdxList[idx] = id;
		}
	};
	__global__ void compressIdxList(const CompressIdxList cil) { cil (); }

	struct MatrixAbCreater
	{
		enum
		{
			dx = 1024,
		};

		int *input_sparseAInfo;
		unsigned int *input_compressedIdxList;
		float *input_sparseCb;

		float *output_MatrixA;
		float *output_Vectorb;

		unsigned int length;
		unsigned int offset;

		__device__ __forceinline__ void
		operator () () const
		{

			unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;

			if(gid>=length)
				return;

			int idx = input_sparseAInfo[gid];
			int i = input_sparseAInfo[1*offset+idx];
			int j = input_sparseAInfo[2*offset+idx];

			float tmp;

			tmp = input_sparseCb[0*offset+idx];
			output_MatrixA[(i*6+0)*length+gid] = tmp;
			output_MatrixA[(j*6+0)*length+gid] = -tmp;

			tmp = input_sparseCb[1*offset+idx];
			output_MatrixA[(i*6+1)*length+gid] = tmp;
			output_MatrixA[(j*6+1)*length+gid] = -tmp;

			tmp = input_sparseCb[2*offset+idx];
			output_MatrixA[(i*6+2)*length+gid] = tmp;
			output_MatrixA[(j*6+2)*length+gid] = -tmp;


			tmp = input_sparseCb[3*offset+idx];
			output_MatrixA[(i*6+3)*length+gid] = tmp;
			output_MatrixA[(j*6+3)*length+gid] = -tmp;

			tmp = input_sparseCb[4*offset+idx];
			output_MatrixA[(i*6+4)*length+gid] = tmp;
			output_MatrixA[(j*6+4)*length+gid] = -tmp;

			tmp = input_sparseCb[5*offset+idx];
			output_MatrixA[(i*6+5)*length+gid] = tmp;
			output_MatrixA[(j*6+5)*length+gid] = -tmp;


			tmp = input_sparseCb[6*offset+idx];
			output_Vectorb[gid] = tmp;
		}
	};
	__global__ void createAb(const MatrixAbCreater A) { A (); }


}

device::SparseCbCreater sparseCb;
device::CompressIdxList idxListCompressor;
device::MatrixAbCreater MatrixAb;


template <typename T>
struct is_not_valid_transform
{
    template <typename K>  __device__
    bool operator()(const K data) const
    {
        // unpack the tuple into x and y coordinates
//        const T x = thrust::get<0>(tuple);

        if (data == -1)
            return true;
        else
            return false;
    }
};

void
GlobalFineRegistrationEstimator::init()
{
	sparseCb.input_pos = (float4 *)getInputDataPointer(WorldCoordinates);
	sparseCb.input_normal = (float4 *)getInputDataPointer(Normals);
	sparseCb.input_sinfo = (SensorInfoV2 *)getInputDataPointer(SensorInfoListV2);
	sparseCb.input_tm = (float *)getInputDataPointer(GlobalTransformationMatrices);


}

void
GlobalFineRegistrationEstimator::execute()
{

	thrust::device_ptr<float> dptr_trans = thrust::device_pointer_cast(sparseCb.input_tm);
	thrust::device_vector<float> d_trans(dptr_trans,dptr_trans+12*n_view);
	thrust::host_vector<float> h_trans = d_trans;
	for(int v=0;v<n_view;v++)
	{
		for(int i=0;i<12;i++)
			printf(" %f | ",h_trans.data()[v*12+i]);

		printf("\n");
	}

	unsigned int view_combinations = (n_view * (n_view-1))/2;
	thrust::device_vector<int> d_infoA(view_combinations*640*480*3);
	thrust::device_vector<float> d_sparseCb(view_combinations*640*480*13);

	sparseCb.output_sparseAInfo = thrust::raw_pointer_cast(d_infoA.data());
	sparseCb.output_sparseAb = thrust::raw_pointer_cast(d_sparseCb.data());

	sparseCb.dist_threshold = 50.f;
	sparseCb.angle_threshold = cos(M_PI/2);

	dim3 sparseGrid(640/sparseCb.dx,480/sparseCb.dy,view_combinations);
	dim3 sparseBlock(sparseCb.dx,sparseCb.dy);
	device::computeSparseCb<<<sparseGrid,sparseBlock>>>(sparseCb);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	thrust::host_vector<int> h_infoA = d_infoA;
	for(int i=0;i<640*480;i+=10000)
	{
		printf("%d -> %d \n",i,h_infoA.data()[i]);
	}

	thrust::device_vector<int> d_prefSum(view_combinations*640*480);
	thrust::exclusive_scan(d_infoA.data(),d_infoA.data()+view_combinations*640*480,d_prefSum.data(),0,device::invalid_binary<int>());

	unsigned int lengthOfCompressedIdxList = d_prefSum[view_combinations*640*480-1];
	printf("lengthOfCompressedIdxList: %d \n",lengthOfCompressedIdxList);
//	thrust::device_vector<unsigned int> d_compressedidxList(lengthOfCompressedIdxList);
//	idxListCompressor.input_sparseInfo = thrust::raw_pointer_cast(d_infoA.data());
//	idxListCompressor.input_prefixSum = thrust::raw_pointer_cast(d_prefSum.data());
//	idxListCompressor.output_compressedIdxList = thrust::raw_pointer_cast(d_compressedidxList.data());
//	idxListCompressor.length = view_combinations*640*480;

//	device::compressIdxList<<<((view_combinations*640*480-1)/idxListCompressor.dx)+1,idxListCompressor.dx>>>(idxListCompressor);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
/*

	MatrixAb.input_sparseAInfo = thrust::raw_pointer_cast(d_infoA.data());
	MatrixAb.input_compressedIdxList = thrust::raw_pointer_cast(d_compressedidxList.data());
	MatrixAb.input_sparseCb = thrust::raw_pointer_cast(d_sparseCb.data());
	MatrixAb.length = lengthOfCompressedIdxList;
	MatrixAb.offset = view_combinations*640*480;

	thrust::device_vector<float> d_MatrixA(lengthOfCompressedIdxList*n_view*6);
	thrust::device_vector<float> d_Vectorb(lengthOfCompressedIdxList*n_view);
	thrust::fill(d_MatrixA.begin(),d_MatrixA.end(),0);
	MatrixAb.output_MatrixA = thrust::raw_pointer_cast(d_MatrixA.data());
	MatrixAb.output_Vectorb = thrust::raw_pointer_cast(d_Vectorb.data());

	device::createAb<<<(lengthOfCompressedIdxList-1)/MatrixAb.dx,MatrixAb.dx>>>(MatrixAb);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
*/



//	thrust::host_vector<unsigned int> psum_out = d_psum;
//	for(int i=0;i<psum.size();i++)
//		printf("%d | ",psum_out[i]);

//	thrust::device_vector<int> d_test(10);
//	thrust::remove(d_test.data(),d_test.data()+10,-1);
	//TODO Reduce
//	thrust::make_tuple(d_infoA.data(),d_infoA.data()+view_combinations*640*480,d_infoA.data()+view_combinations*640*480*2);

//	typedef thrust::device_ptr<int> IntDevPtr;
//	typedef thrust::tuple<IntDevPtr,IntDevPtr,IntDevPtr> IntTuple;
//	typedef thrust::zip_iterator<IntTuple> IntTupleIterator;
//
//	IntTupleIterator list_beg = thrust::make_zip_iterator(thrust::make_tuple(d_infoA.data()+view_combinations*640*480*0,d_infoA.data()+view_combinations*640*480*1,d_infoA.data()+view_combinations*640*480*2));
//	IntTupleIterator list_end = thrust::make_zip_iterator(thrust::make_tuple(d_infoA.data()+view_combinations*640*480*1,d_infoA.data()+view_combinations*640*480*2,d_infoA.data()+view_combinations*640*480*3));
//	thrust::remove_if(list_beg,list_end,is_not_valid_transform<int>());

//	thrust::remove_if(
//			thrust::make_zip_iterator(thrust::make_tuple(d_infoA.data())),
//			thrust::make_zip_iterator(thrust::make_tuple(d_infoA.data()+10)),
//			device::is_not_valid<int>());

//	thrust::remove_if(
//	thrust::make_zip_iterator(
//			thrust::make_tuple(d_infoA.data()+view_combinations*640*480*0,d_infoA.data()+view_combinations*640*480*1,d_infoA.data()+view_combinations*640*480*2)),
//	thrust::make_zip_iterator(
//			thrust::make_tuple(d_infoA.data()+view_combinations*640*480*1,d_infoA.data()+view_combinations*640*480*2,d_infoA.data()+view_combinations*640*480*3)),
//			device::is_not_valid<int>()
//	);

//	MatrixAb.input_sparseAInfo =
}

GlobalFineRegistrationEstimator::GlobalFineRegistrationEstimator(unsigned int n_view) : n_view(n_view)
{
	// TODO Auto-generated constructor stub

}

GlobalFineRegistrationEstimator::~GlobalFineRegistrationEstimator()
{
	// TODO Auto-generated destructor stub
}


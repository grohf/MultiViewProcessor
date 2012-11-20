/*
 * RigidBodyTransformationAdvancedEstimatior.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: avo
 */

#include "RigidBodyTransformationAdvancedEstimatior.h"


namespace device
{



	struct CorrespondanceListEstimator : public FeatureBaseKernel
	{
		enum
		{
			s = 32, //points_per_block
			k = 32, //global_points_per_sweep
		};


		float *input_feature_histo;
		int *input_idxList;
		unsigned int *input_idxLength;

		float *input_rndList;

		//TODO: get view combination from blockIdx.z
		unsigned int view_src;


		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float shm_local_histo[s*bins];
			__shared__ unsigned int shm_local_idx[s];

			__shared__ float shm_global_buffer[k*bins];
			__shared__ unsigned int shm_global_idx[k];


			__shared__ float shm_dist[k*s*2];
			__shared__ unsigned int shm_idx[k*s*2];

			//TODO: get view combination from blockIdx.z

			if(threadIdx.x<s)
			{
				float f = input_rndList[blockIdx.x*s+threadIdx.x];
				unsigned int idx = input_idxList[view_src*640*480+(unsigned int)(f*input_idxLength[view_src])];

//				shm_local_idx[threadIdx.x] = idx;
//				output_s_idx[blockIdx.x*sp+threadIdx.x] = idx;

			}

		}

	};


}





void
RigidBodyTransformationAdvancedEstimatior::init()
{

}

void
RigidBodyTransformationAdvancedEstimatior::execute()
{

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

RigidBodyTransformationAdvancedEstimatior::~RigidBodyTransformationAdvancedEstimatior()
{
	// TODO Auto-generated destructor stub
}


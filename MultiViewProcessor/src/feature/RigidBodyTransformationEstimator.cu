/*
 * RigidBodyTransformationEstimator.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#include "RigidBodyTransformationEstimator.h"
#include "SVDEstimatorCPU.h"

namespace device
{
	struct OMEstimator
	{

		__device__ __forceinline__ void
		operator () () const
		{

		}

	};


}


void RigidBodyTransformationEstimator::init()
{

}

void RigidBodyTransformationEstimator::execute()
{
	printf("hmmm \n");
	SVDEstimator_CPU *svd = new SVDEstimator_CPU();
	svd->execute();
}


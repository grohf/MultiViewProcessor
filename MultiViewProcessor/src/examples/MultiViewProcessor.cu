/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <processor.h>

// includes, cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <helper_cuda_gl.h>

#include "../sources/SyncFreenectSource.h"
#include "../filter/TestFilter.h"
#include "../filter/TestFilter2.h"
#include "../filter/ATrousFilter.h"
#include "../feature/NormalPCAEstimator.h"
#include "../feature/FPFH.h"
#include "../feature/SVDEstimatorCPU.h"
#include "../feature/RigidBodyTransformationEstimator.h"


void runTestProcessor()
{
	SyncFreenectSource *src = new SyncFreenectSource();
//	SourcePtr src(new SyncFreenectSource());

	Processor p;
	p.setSource(SourcePtr(src));

//	SVDEstimator_CPU *svd_cpu = new SVDEstimator_CPU();
//	p.addFeature(svd_cpu);

	ATrousFilter *atrousfilter = new ATrousFilter();
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));

	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator();
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
//	nPCAestimator->setWorldCoordinates(src->getTargetData(SyncFreenectSource::PointXYZI));
	p.addFeature(nPCAestimator);

	FPFH *fpfhEstimator = new FPFH();
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());

	p.addFeature(fpfhEstimator);

	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(1024,32);
	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
	p.addFeature(rbEstimator);

//	FilterPtr fp = atrousfilter->ptr;

	p.start();

	src->~SyncFreenectSource();

}

void coorespTest()
{
	SyncFreenectSource *src = new SyncFreenectSource();
//	SourcePtr src(new SyncFreenectSource());

	Processor p;
	p.setSource(SourcePtr(src));
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

	runTestProcessor();
	return 0;
}

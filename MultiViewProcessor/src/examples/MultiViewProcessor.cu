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
#include "../filter/TruncateThresholdFilter.h"
#include "../feature/NormalPCAEstimator.h"
#include "../feature/FPFH.h"
#include "../feature/SVDEstimatorCPU.h"
#include "../feature/RigidBodyTransformationEstimator.h"
#include "../feature/TranformationValidator.h"


void runTestProcessor()
{
	SyncFreenectSource *src = new SyncFreenectSource(1);
//	SourcePtr src(new SyncFreenectSource());

	Processor p;
	p.setSource(SourcePtr(src));

//	SVDEstimator_CPU *svd_cpu = new SVDEstimator_CPU();
//	p.addFeature(svd_cpu);

	ATrousFilter *atrousfilter = new ATrousFilter(1,1,5,5);
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));

	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(1);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
//	nPCAestimator->setWorldCoordinates(src->getTargetData(SyncFreenectSource::PointXYZI));
	p.addFeature(nPCAestimator);

	FPFH *fpfhEstimator = new FPFH(1);
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());

	p.addFeature(fpfhEstimator);

	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(1,256,64,32);
	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
	rbEstimator->setCoordinatesMap(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(rbEstimator);

//	FilterPtr fp = atrousfilter->ptr;

	p.start();

	src->~SyncFreenectSource();

}


void runMultiViewTest()
{
	SyncFreenectSource *src = new SyncFreenectSource(2);
//	SourcePtr src(new SyncFreenectSource());

	Processor p;
	p.setSource(SourcePtr(src));

//	TruncateThresholdFilter *truncateThresholdFilter = new TruncateThresholdFilter(2,600.f,2000.f);
//	truncateThresholdFilter->setWorldCoordinates(src->getTargetData(SyncFreenectSource::PointXYZI));
//	p.addFilter(FilterPtr(truncateThresholdFilter));

	ATrousFilter *atrousfilter = new ATrousFilter(2,2,10,5,2);
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));
	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(2);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);


	FPFH *fpfhEstimator = new FPFH(2);
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());
	p.addFeature(fpfhEstimator);

	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(2,256,64,32);
	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
	rbEstimator->setCoordinatesMap(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(rbEstimator);

	TranformationValidator *validator = new TranformationValidator();
	validator->setWorldCooordinates(atrousfilter->getFilteredWorldCoordinates());
	validator->setTransformationmatrices(rbEstimator->getTransformationMatrices());
	validator->setTranformationInfoList(rbEstimator->getTransformationInfoList());
	p.addFeature(validator);

	p.start();

	src->~SyncFreenectSource();
}

void coorespTest()
{
	SyncFreenectSource *src = new SyncFreenectSource(2);
//	SourcePtr src(new SyncFreenectSource());



	Processor p;
	p.setSource(SourcePtr(src));
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

//	runTestProcessor();

	runMultiViewTest();
	return 0;
}

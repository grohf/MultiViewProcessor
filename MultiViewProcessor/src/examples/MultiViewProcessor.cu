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

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "../sources/SyncFreenectSource.h"
#include "../sources/SynthRGBDBenchmarkSource.h"
#include "../filter/TestFilter.h"
#include "../filter/TestFilter2.h"
#include "../filter/ATrousFilter.h"
#include "../filter/TruncateThresholdFilter.h"
#include "../feature/NormalPCAEstimator.h"
#include "../feature/FPFH.h"
#include "../feature/DFPFHEstimator.h"
#include "../feature/SVDEstimatorCPU.h"
#include "../feature/RigidBodyTransformationEstimator.h"
#include "../feature/TranformationValidator.h"

#include "../include/point_info.hpp"

#include "../debug/EigenCheckClass.h"


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

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(1,20);
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

	TruncateThresholdFilter *truncateThresholdFilter = new TruncateThresholdFilter(2,600.f,2000.f);
	truncateThresholdFilter->setWorldCoordinates(src->getTargetData(SyncFreenectSource::PointXYZI));
	p.addFilter(FilterPtr(truncateThresholdFilter));

	ATrousFilter *atrousfilter = new ATrousFilter(2,2,10,5,2);
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));
	atrousfilter->setPointIntensity(src->getTargetData(SyncFreenectSource::PointIntensity));
	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(2,50);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);


	FPFH *fpfhEstimator = new FPFH(2);
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());
	p.addFeature(fpfhEstimator);

//	unsigned int r_ransac = 4096;
//
//	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(2,r_ransac,64,32);
//	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
//	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
//	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
//	rbEstimator->setCoordinatesMap(atrousfilter->getFilteredWorldCoordinates());
//	p.addFeature(rbEstimator);
//
//	TranformationValidator *validator = new TranformationValidator(2,r_ransac);
//	validator->setWorldCooordinates(atrousfilter->getFilteredWorldCoordinates());
//	validator->setNormals(nPCAestimator->getNormals());
//	validator->setSensorInfoList(src->getTargetData(SyncFreenectSource::SensorInfoList));
//	validator->setTransformationmatrices(rbEstimator->getTransformationMatrices());
//	validator->setTranformationInfoList(rbEstimator->getTransformationInfoList());
//	p.addFeature(validator);

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

void TesterFct()
{
	DFPFHEstimator *dfpfhEstimator = new DFPFHEstimator(1);
	dfpfhEstimator->TestDFPFHE();

//	TranformationValidator *validator = new TranformationValidator(2,512);
//	validator->TestMinimumPicker();
//	validator->TestSumCalculator();
//	validator->TestTransform();

//	EigenCheckClass *cpuCheck = new EigenCheckClass();
//	cpuCheck->createTestOMRotation();
//
//	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(2,256,64,32);
//	rbEstimator->TestCorrelationMatrix(cpuCheck->pos_m,cpuCheck->pos_d,cpuCheck->H);

//	unsigned int k = 0;
//	float f = 0;
//	device::setValid(f);
////	device::setReconstructed(f,31);
//
//	printf("%d %d \n",device::isReconstructed(f),device::getReconstructionLevel(f));
//
//	device::unsetReconstructed(f);
//
//	printf("%d \n",device::isReconstructed(f));
//
//	device::setForeground(f);
//
//	printf("%d %d %d",device::isForeground(f),device::isBackground(f),device::isSegmented(f));
}

void TestTransformationerror()
{

	SyncFreenectSource *src = new SyncFreenectSource(1);
	//	SourcePtr src(new SyncFreenectSource());

	Processor p;
	p.setSource(SourcePtr(src));

//	TruncateThresholdFilter *truncateThresholdFilter = new TruncateThresholdFilter(2,600.f,2000.f);
//	truncateThresholdFilter->setWorldCoordinates(src->getTargetData(SyncFreenectSource::PointXYZI));
//	p.addFilter(FilterPtr(truncateThresholdFilter));

	ATrousFilter *atrousfilter = new ATrousFilter(1,2,10,5,2);
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));
	atrousfilter->setPointIntensity(src->getTargetData(SyncFreenectSource::PointIntensity));
	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(1,20);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);

	p.start();


	thrust::device_ptr<float4> d_views_ptr = thrust::device_pointer_cast((float4 *)atrousfilter->getFilteredWorldCoordinates()->getDeviceDataPtr().get());
	thrust::device_ptr<float4> d_normals_ptr = thrust::device_pointer_cast((float4 *)nPCAestimator->getNormals()->getDeviceDataPtr().get());

	thrust::device_vector<float4> d_views(d_views_ptr, d_views_ptr+640*480);
	thrust::device_vector<float4> d_normals(d_normals_ptr, d_normals_ptr+640*480);

	thrust::host_vector<float4> views = d_views;
	thrust::host_vector<float4> normals = d_normals;

	thrust::host_vector<float4> views_synth(views.size());
	thrust::host_vector<float4> normals_synth(normals.size());

	thrust::host_vector<float> transformationMatrix(12);

	EigenCheckClass *cpuCheck = new EigenCheckClass();
	cpuCheck->createRotatedViews(views,views_synth,normals,normals_synth,transformationMatrix);
//	cpuCheck->createRefrenceTransformationMatrices();

//	for(int i=0;i<12;i++)
//	{
//		printf("%f | ",transformationMatrix[i]);
//	}
//	printf("\n");
//	TranformationValidator *validator = new TranformationValidator(2,64);
//	validator->TestTransformError(views,views_synth,normals,normals_synth,transformationMatrix);

	/*
	thrust::host_vector<float4> normals(views.size());
	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(2);
	nPCAestimator->TestNormalCreater(views,normals);



	TranformationValidator *validator = new TranformationValidator(2,1);
//	validator->TestTransformError(synth_view,src_view,transformationMatrix);
*/
	src->~SyncFreenectSource();
}

void TestSynthInput()
{
	Processor p;
	SynthRGBDBenchmarkSource *src = new SynthRGBDBenchmarkSource(2,"/home/avo/Desktop/rgbd_dataset_freiburg3_teddy/",false);
	p.setSource(SourcePtr(src));

	ATrousFilter *atrousfilter = new ATrousFilter(2,2,15,3,2);
	atrousfilter->setInput2DPointCloud(src->getWorldCoordinates());
	atrousfilter->setInputSensorInfo(src->getSensorV2InfoList());
	atrousfilter->setPointIntensity(src->getIntensity());
	p.addFilter(FilterPtr(atrousfilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(2,50);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);

	TruncateThresholdFilter *truncateThresholdFilter = new TruncateThresholdFilter(2,600.f,2000.f);
	truncateThresholdFilter->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFilter(FilterPtr(truncateThresholdFilter));

	FPFH *fpfhEstimator = new FPFH(2);
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());
	p.addFeature(fpfhEstimator);


	unsigned int r_ransac = 1024;

	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(2,r_ransac,512,32);
	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
	rbEstimator->setCoordinatesMap(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(rbEstimator);

	p.start();

//	thrust::device_ptr<float4> dptr = thrust::device_pointer_cast((float4 *)atrousfilter->getFilteredWorldCoordinates()->getDeviceDataPtr().get());
//	thrust::device_vector<float4> d_points(dptr,dptr+2*640*480);
//	thrust::host_vector<float4> h_points = d_points;
//	for(int i=0;i<2*640*480;i++)
//		if(i%5000==0)
//			printf("%f %f %f | ",h_points[i].x,h_points[i].y,h_points[i].z);
//
//	printf("\n");

	rbEstimator->TestCorrelationQuality(src->getGroundTruthTransformation());

//	char **depth_pngs,**rgb_pngs;
//	float *transformationMatrices;
//	unsigned int lns[] = {1,2,3};
//	EigenCheckClass *cpuCheck = new EigenCheckClass();
//	cpuCheck->createRefrenceTransformationMatrices("/home/avo/Desktop/rgbd_dataset_freiburg3_teddy/",2,lns,depth_pngs,rgb_pngs,transformationMatrices);


}

void PointInfoTest()
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
	atrousfilter->setPointIntensity(src->getTargetData(SyncFreenectSource::PointIntensity));
	p.addFilter(FilterPtr(atrousfilter));

	TruncateThresholdFilter *truncateThresholdFilter = new TruncateThresholdFilter(2,600.f,2000.f);
	truncateThresholdFilter->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFilter(FilterPtr(truncateThresholdFilter));

	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(2,20);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);

	FPFH *fpfhEstimator = new FPFH(2);
	fpfhEstimator->setPointCoordinates(atrousfilter->getFilteredWorldCoordinates());
	fpfhEstimator->setNormals(nPCAestimator->getNormals());
	p.addFeature(fpfhEstimator);

	unsigned int r_ransac = 1024;

	RigidBodyTransformationEstimator *rbEstimator = new RigidBodyTransformationEstimator(2,r_ransac,512,32);
	rbEstimator->setPersistanceHistogramMap(fpfhEstimator->getFPFH());
	rbEstimator->setPersistanceIndexList(fpfhEstimator->getPersistanceIndexList());
	rbEstimator->setPersistenceInfoList(fpfhEstimator->getPersistenceInfoList());
	rbEstimator->setCoordinatesMap(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(rbEstimator);

	TranformationValidator *validator = new TranformationValidator(2,r_ransac);
	validator->setWorldCooordinates(atrousfilter->getFilteredWorldCoordinates());
	validator->setNormals(nPCAestimator->getNormals());
	validator->setSensorInfoList(src->getTargetData(SyncFreenectSource::SensorInfoList));
	validator->setTransformationmatrices(rbEstimator->getTransformationMatrices());
	validator->setTranformationInfoList(rbEstimator->getTransformationInfoList());
	p.addFeature(validator);


	p.start();

	src->~SyncFreenectSource();
}

void runNormalTest()
{
	Processor p;

	SyncFreenectSource *src = new SyncFreenectSource(1);
	p.setSource(SourcePtr(src));


	ATrousFilter *atrousfilter = new ATrousFilter(1,3,15,3,2);
	atrousfilter->setInput2DPointCloud(src->getTargetData(SyncFreenectSource::PointXYZI));
	atrousfilter->setInputSensorInfo(src->getTargetData(SyncFreenectSource::SensorInfoList));
	atrousfilter->setPointIntensity(src->getTargetData(SyncFreenectSource::PointIntensity));
	p.addFilter(FilterPtr(atrousfilter));


	NormalPCAEstimator *nPCAestimator = new NormalPCAEstimator(1,50);
	nPCAestimator->setWorldCoordinates(atrousfilter->getFilteredWorldCoordinates());
	p.addFeature(nPCAestimator);

	p.start();
	src->~SyncFreenectSource();
}

int main(int argc, char **argv) {

//	runTestProcessor();
//	runNormalTest();

	TesterFct();
//	TestTransformationerror();
//	TestSynthInput();
//	PointInfoTest();




//	runMultiViewTest();
	return 0;
}

/*
 * processor.h
 *
 *  Created on: Aug 23, 2012
 *      Author: avo
 */

#ifndef PROCESSOR_H_
#define PROCESSOR_H_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <enhancer.hpp>
#include <source.h>
#include <filter.hpp>
#include <feature.hpp>

#include <device_data.h>
#include <vector>

#include <boost/ptr_container/ptr_vector.hpp>

#include <helper_cuda.h>

class Processor
{

SourcePtr srcPtr;
std::vector<EnhancerPtr> alignmentPtrList;
std::vector<EnhancerPtr> viewPtrList;
std::vector<EnhancerPtr> allPtrList;


std::vector<DeviceDataInfoPtr> memList;

public:

	enum Mode
	{
		Alignment,
		View,
		All,
	};

	Processor();

	unsigned int maxAlignFrames;
	unsigned int captureingFrames;
	Processor(unsigned int maxAlignFrames);
	Processor(unsigned int maxAlignFrames, unsigned int captureingFrames);

//	void setSource(Source& source);
	void addFilter(Filter& filter);
	void addEnhancer(Enhancer& enh);

	void start();

	void setSource(SourcePtr srcPtr_)
	{
		srcPtr = srcPtr_;
		addRequestedDeviceData(srcPtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFilter(FilterPtr filterPtr)
	{
		EnhancerPtr ePtr = filterPtr;
		alignmentPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(FeaturePtr featurePtr)
	{
		EnhancerPtr ePtr = featurePtr;
		alignmentPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(Feature *feature)
	{
		addFeature(feature->getFeaturePtr());
	}

	void addFilter(Filter *filter)
	{
		addFilter(filter->getFilterPtr());
	}




	/* Mode */


	void addFilter(FilterPtr filterPtr, Mode m)
	{
		EnhancerPtr ePtr = filterPtr;
		allPtrList.push_back(ePtr);

		if(m == Alignment || m == All)
			alignmentPtrList.push_back(ePtr);

		if(m == View || m == All)
			viewPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(FeaturePtr featurePtr, Mode m)
	{
		EnhancerPtr ePtr = featurePtr;
		allPtrList.push_back(ePtr);

		if(m == Alignment || m == All)
			alignmentPtrList.push_back(ePtr);

		if(m == View || m == All)
			viewPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(Feature *feature,Mode m)
	{
		addFeature(feature->getFeaturePtr(),m);
	}

	void addFilter(Filter *filter,Mode m)
	{
		addFilter(filter->getFilterPtr(),m);
	}

	static bool contRun;
	static bool contTry;
	static bool isAligned;

	static void setAligned()
	{
		isAligned = true;
	}

	static void breakRun()
	{
		contRun = false;
	}

	static void breakTry()
	{
		contTry = false;
	}

private:

	void addRequestedDeviceData(std::vector<DeviceDataInfoPtr> list)
	{
		for(int i=0;i<list.size();i++)
			memList.push_back(list[i]);
	}

	void allocateDeviceMemory()
	{
		unsigned int amount = 0;
		for(int i=0;i<memList.size();i++)
		{
			DeviceDataInfoPtr ddip = memList[i];
			DeviceDataParams params = ddip->getDeviceDataParams();
//			printf("i:%d | elements: %d | element_size: %d \n",i,params.elements,params.element_size);

			void* d;

			checkCudaErrors(cudaMalloc((void**)&d,params.elements*params.element_size));
			amount += params.elements*params.element_size;

			ddip->setDeviceDataPtr(DeviceDataPtr(d));


//			printf("allocated DeviceDataType: %d \n",params.dataType);
		}

		printf("allocated Memory: %f MB \n", ((float)amount)/(1024.f*1024.f));
	}
};


#endif /* PROCESSOR_H_ */

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
std::vector<EnhancerPtr> enhancerPtrList;


std::vector<DeviceDataInfoPtr> memList;

public:

	Processor();

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
		enhancerPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(FeaturePtr featurePtr)
	{
		EnhancerPtr ePtr = featurePtr;
		enhancerPtrList.push_back(ePtr);

		addRequestedDeviceData(ePtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFeature(Feature *feature)
	{
		addFeature(feature->getFeaturePtr());
	}

private:

	void addRequestedDeviceData(std::vector<DeviceDataInfoPtr> list)
	{
		for(int i=0;i<list.size();i++)
			memList.push_back(list[i]);
	}

	void allocateDeviceMemory()
	{
		for(int i=0;i<memList.size();i++)
		{
			DeviceDataInfoPtr ddip = memList[i];
			DeviceDataParams params = ddip->getDeviceDataParams();
			printf("i:%d | %d | %d \n",i,params.elements,params.element_size);

			void* d;

			checkCudaErrors(cudaMalloc((void**)&d,params.elements*params.element_size));

			ddip->setDeviceDataPtr(DeviceDataPtr(d));


			printf("allocated DeviceDataType: %d \n",params.dataType);
		}
	}
};


#endif /* PROCESSOR_H_ */

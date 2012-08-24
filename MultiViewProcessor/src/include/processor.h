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

public:

	Processor();

	void setSource(Source& source);
	void addFilter(Filter& filter);
	void addFeature(Feature& feature);
	void addEnhancer(Enhancer& enh);

	void start();

	void setSource(SourcePtr srcPtr)
	{
		allocateDeviceMemory(srcPtr->getRequestedDeviceDataInfoPtrList());
	}

	void addFilter(FilterPtr filterPtr)
	{
		EnhancerPtr ePtr = filterPtr;
		enhancerPtrList.push_back(ePtr);
	}

private:
	void allocateDeviceMemory(std::vector<DeviceDataInfoPtr> list)
	{
		for(int i=0;i<list.size();i++)
		{
			DeviceDataInfoPtr ddip = list[i];
			DeviceDataParams params = ddip->getDeviceDataParams();
			printf("i:%d | %d | %d \n",i,params.elements,params.element_size);

			void* d = ddip->getDeviceDataPtr().get();

			checkCudaErrors(cudaMalloc((void**)&d,params.elements*params.element_size));

			printf("allocated DeviceDataType: %d \n",params.dataType);
		}
	}
};


#endif /* PROCESSOR_H_ */

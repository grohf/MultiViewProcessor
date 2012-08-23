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

#include <source.h>
#include <filter.hpp>
#include <feature.hpp>

#include <device_data.h>
#include <vector>

#include <helper_cuda.h>

class Processor
{
public:

	Processor();

	void addSource(Source& source);
	void addFilter(Filter& filter);
	void addFeature(Feature& feature);

	void start();

private:
	void allocateDeviceMemory(std::vector<DeviceDataInfoPtr> list)
	{
		for(int i=0;i<list.size();i++)
		{
			DeviceDataInfoPtr ddip = list[i];
			printf("i:%d | %d | %d \n",i,ddip->params.elements,ddip->params.element_size);
			checkCudaErrors(cudaMalloc((void**)&list[i]->ptr,list[i]->params.elements*list[i]->params.element_size));
			printf("allocated DeviceDataType: %d \n",list[i]->params.dataType);
		}
	}
};


#endif /* PROCESSOR_H_ */

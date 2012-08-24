

#include <vector>

// includes, cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <helper_cuda_gl.h>

#include "processor.h"


Processor::Processor()
{

//	findCudaGLDevice(0,(const char**)"");

}

void
Processor::setSource(Source& src_)
{
	printf("addSource() \n");
	std::vector<DeviceDataInfoPtr> qryList = src_.getRequestedDeviceDataInfoPtrList();
//	printf("qryList size: %d \n",qryList.size());
//
//	int i1 = qryList[0]->getDeviceDataParams().element_size;
//	int i2 = qryList[1]->getDeviceDataParams().element_size;

//	printf("%d %d \n",i1,i2);

	allocateDeviceMemory(qryList);

}

void Processor::addFilter(Filter& filter)
{
	Enhancer *enh = &filter;
	EnhancerPtr ePtr(enh);
	enhancerPtrList.push_back(ePtr);
}

void Processor::start()
{
	for(int i=0;i<enhancerPtrList.size();i++)
	{
		enhancerPtrList[i]->execute();
	}
}


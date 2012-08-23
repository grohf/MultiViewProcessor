

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
Processor::addSource(Source& src)
{
	printf("addSource() \n");
	std::vector<DeviceDataInfoPtr> qryList = src.getRequestedDeviceDataInfoPtrList();
	printf("qryList size: %d \n",qryList.size());


	std::vector<DeviceDataInfoPtr> qryList2(1);

	DeviceDataInfo ddi;
	DeviceDataParams params;
	params.elements = 2;
	params.element_size = 1;
	ddi.params = params;

	qryList2[0] = &ddi;

	allocateDeviceMemory(qryList2);

}


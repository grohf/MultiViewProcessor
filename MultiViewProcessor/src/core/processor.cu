

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


void Processor::addFilter(Filter& filter)
{
	Enhancer *enh = &filter;
	EnhancerPtr ePtr(enh);
	enhancerPtrList.push_back(ePtr);
}

void Processor::start()
{
	findCudaDevice(0,(const char**)"");

	allocateDeviceMemory();

	/* INITS */
	srcPtr->init();
	for(int i=0;i<enhancerPtrList.size();i++)
	{
		enhancerPtrList[i]->init();
	}

	/* EXECUTES */
	srcPtr->execute();
	for(int i=0;i<enhancerPtrList.size();i++)
	{
		enhancerPtrList[i]->execute();
	}
}


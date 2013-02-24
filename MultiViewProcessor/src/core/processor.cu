

#include <vector>

// includes, cuda
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <helper_cuda_gl.h>

#include "processor.h"


bool Processor::contRun;
bool Processor::contTry;
bool Processor::isAligned;

Processor::Processor() : maxAlignFrames(1),captureingFrames(1)
{

//	findCudaGLDevice(0,(const char**)"");
//	contRun = true;
//	contTry = true;
//
	isAligned = false;

}

Processor::Processor(unsigned int maxAlignFrames) : maxAlignFrames(maxAlignFrames),captureingFrames(1)
{
	isAligned = false;
}

Processor::Processor(unsigned int maxAlignFrames, unsigned int captureingFrames) : maxAlignFrames(maxAlignFrames),captureingFrames(captureingFrames)
{
	isAligned = false;
}


void Processor::addFilter(Filter& filter)
{
	Enhancer *enh = &filter;
	EnhancerPtr ePtr(enh);
	alignmentPtrList.push_back(ePtr);
}

void Processor::start()
{
	findCudaDevice(0,(const char**)"");

	allocateDeviceMemory();

	/* INITS */
	srcPtr->init();
	for(int i=0;i<allPtrList.size();i++)
	{
		allPtrList[i]->init();
	}

	printf("maxAlignFrames: %d | captureingFrames: %d \n",maxAlignFrames,captureingFrames);

//	for(int l=0;l<25 && contRun;l++)
////	while(contRun)
//	{
//		/* EXECUTES */
//		srcPtr->execute();
//		for(int i=0;i<alignmentPtrList.size()&&contTry;i++)
//		{
//			alignmentPtrList[i]->execute();
//		}
//	}

	for(int f=0;f<maxAlignFrames && !isAligned;f++)
	{
		/* EXECUTES ALIGNMENT*/
		srcPtr->execute();
		for(int i=0;i<alignmentPtrList.size();i++)
		{
			alignmentPtrList[i]->execute();
		}
	}

	for(int f=0;f<captureingFrames && isAligned;f++)
	{
		printf("cp: %d/%d \n",f,captureingFrames);
		/* EXECUTES VIEW*/
		srcPtr->execute();
		for(int i=0;i<viewPtrList.size();i++)
		{
			viewPtrList[i]->execute();
		}
	}


}


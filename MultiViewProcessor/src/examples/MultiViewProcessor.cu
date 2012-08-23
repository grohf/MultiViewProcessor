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

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv) {

//	TestFilter *f = new TestFilter();

//	findCudaGLDevice(argc,(const char**)argv);

//	findCudaDevice(argc,(const char**)argv);

	SyncFreenectSource src;

	Processor p;
	p.addSource(src);

	return 0;
}

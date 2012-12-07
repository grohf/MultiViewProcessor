/*
 * ATrousFilterAdvanced.cpp
 *
 *  Created on: Dec 1, 2012
 *      Author: avo
 */

#include "ATrousFilterAdvanced.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_image.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "point_info.hpp"
#include "../sink/pcd_io.h"

namespace device
{

	struct ATrousFilter
	{
		enum
		{
			dx = 32,
			dy = 24,

			h_r = 2,
			h_length = h_r + 1,
			h_lx = dx + 2*h_r,
			h_ly = dy + 2*h_r,

		};

		float4* input_pos;
		float4* output_pos;


		__device__ __forceinline__ void
		operator () () const
		{

		}
	};


}


void ATrousFilterAdvanced::init()
{

}

void ATrousFilterAdvanced::execute()
{

}



ATrousFilterAdvanced::ATrousFilterAdvanced(unsigned int n_view, unsigned int iterations, float sigma_depth, float sigma_intensity)
: n_view(n_view), iterations(iterations), sigma_depth(sigma_depth), sigma_intensity(sigma_intensity)
{
	// TODO Auto-generated constructor stub

}

ATrousFilterAdvanced::~ATrousFilterAdvanced()
{
	// TODO Auto-generated destructor stub
}


/*
 * Integrator.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: avo
 */

#include "Integrator.h"

namespace device
{
	struct IntegratorFilter
	{
		enum
		{
			dx = 32,
			dy = 32,
		};

		float4* input_pos;
		float* input_transformations;

		__device__ __forceinline__ void
		operator () () const
		{
			unsigned int sx = blockIdx.x*blockDim.x+threadIdx.x;
			unsigned int sy = blockIdx.y*blockDim.y+threadIdx.y;

			float4 pos = input_pos[blockIdx.z*640*480+sy*640+sx];
			float4 tp = make_float4(0,0,0,0);

			__shared__ float tm[12];
			if(sy==0&&sx<12)
			{
				tm[sx] = input_transformations[blockIdx.z*12+sx];
			}
			__syncthreads();

			tp.x = tm[0] * pos.x + tm[1] * pos.y + tm[2] * pos.z + tm[9];
			tp.y = tm[3] * pos.x + tm[4] * pos.y + tm[5] * pos.z + tm[10];
			tp.z = tm[6] * pos.x + tm[7] * pos.y + tm[8] * pos.z + tm[11];

			input_pos[blockIdx.z*640*480+sy*640+sx] = tp;
		}

	};

	__global__ void integrateViews(const IntegratorFilter inte){ inte(); }

}

device::IntegratorFilter integratorFilter;

void Integrator::init()
{
	integratorFilter.input_pos = (float4 *)getInputData(Positions);
	integratorFilter.input_transformations = (float *)getInputData(MinTransformations);
}

void Integrator::execute()
{


//	device::IntegratorFilter<<<>>>
}


Integrator::Integrator()
{
	// TODO Auto-generated constructor stub

}

Integrator::~Integrator()
{
	// TODO Auto-generated destructor stub
}


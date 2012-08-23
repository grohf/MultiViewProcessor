/*
 * SyncFreenectSource.cpp
 *
 *  Created on: Aug 23, 2012
 *      Author: avo
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <libfreenect_sync.h>
#include <libfreenect.h>
#include <libfreenect-registration.h>

#include "SyncFreenectSource.h"
#include <vector_types.h>


//TODO: get deviceMemory stuff out here
#include <helper_cuda.h>

namespace device
{
	namespace constant
	{
	__constant__ double ref_pix_size;
	__constant__ double ref_dist;
	}

	struct SyncFreenectLoader
	{
		uint8_t *rgb;
		uint16_t *depth;

		float4 *xyzi;
		uchar4 *rgba;

		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZ(int cx,int cy, int wz, float &wx, float &wy) const
		{

			double factor = (2*constant::ref_pix_size * wz) / constant::ref_dist;

			wx = (float) ((cx - 320) * factor);
			wy = (float) ((cy - 240) * factor);

		}

		__device__ __forceinline__ void
		operator () () const
		{
	//			if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0)
	//				printf("oO!! \n");

			int x = blockIdx.x*blockDim.x + threadIdx.x;
			int y = blockIdx.y*blockDim.y + threadIdx.y;
			int z = blockIdx.z;

			int wz = depth[y*640+x];
			float wx,wy;

			convertCXCYWZtoWXWYWZ(x,y,wz,wx,wy);

			unsigned char r = rgb[3*(y*640+x)+0];
			unsigned char g = rgb[3*(y*640+x)+1];
			unsigned char b = rgb[3*(y*640+x)+2];

			float4 f4 = make_float4(wx,wy,wz,((r+r+r+b+g+g+g+g)>>3));
			xyzi[z*640*480+y*640+x] = f4;

		}

	};
	__global__ void loadSyncFreenectFrame(const SyncFreenectLoader sfl){ sfl(); }
}

device::SyncFreenectLoader loader;

void
SyncFreenectSource::init()
{
	checkCudaErrors(cudaMalloc((void**)&d_rgb,640*480*3*sizeof(uint8_t)));
	checkCudaErrors(cudaMalloc((void**)&d_depth,640*480*sizeof(uint16_t)));

	setRegInfo();

	loader.rgb = d_rgb;
	loader.depth = d_depth;

	loader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	loader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);

	block = dim3(32,24);
	grid = dim3(block.x/640,block.y/320);
}

void
SyncFreenectSource::loadFrame()
{
	uint16_t *h_depth = 0;
	uint8_t *h_rgb = 0;
	uint32_t ts;

	freenect_sync_get_depth((void**)&h_depth, &ts, 0, FREENECT_DEPTH_REGISTERED);
	freenect_sync_get_video((void**)&h_rgb, &ts, 0, FREENECT_VIDEO_RGB);

	checkCudaErrors(cudaMemcpy(d_rgb,h_rgb,640*480*3*sizeof(uint8_t),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_depth,h_depth,640*480*sizeof(uint16_t),cudaMemcpyHostToDevice));

	device::loadSyncFreenectFrame<<<grid,block>>>(loader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

SyncFreenectSource::SyncFreenectSource()
{

	DeviceDataParams pointXYZIParams;
	pointXYZIParams.elements = 640*480;
	pointXYZIParams.element_size = sizeof(float4);
	pointXYZIParams.dataType = Point4D;
	pointXYZIParams.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(pointXYZIParams),PointXYZI);

	DeviceDataParams pointRGBAParams;
	pointRGBAParams.elements = 640*480;
	pointRGBAParams.element_size = sizeof(uchar4);
	pointRGBAParams.dataType = Point4D;
	pointRGBAParams.elementType = UCHAR4;

	addTargetData(addDeviceDataRequest(pointRGBAParams),PointRGBA);
	printf("end \n");
}

void
SyncFreenectSource::setRegInfo()
{
	freenect_context *f_ctx;
	freenect_device *f_dev;


	if (freenect_init(&f_ctx, NULL) < 0) {
		printf("freenect_init() failed\n");
		return;
	}

	freenect_set_log_level(f_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

	int nr_devices = freenect_num_devices (f_ctx);
	printf ("Number of devices found: %d\n", nr_devices);

	int user_device_number = 0;


	if (nr_devices < 1) {
		freenect_shutdown(f_ctx);
		return;
	}

	if (freenect_open_device(f_ctx, &f_dev, user_device_number) < 0) {
		printf("Could not open device\n");
		freenect_shutdown(f_ctx);
		return;
	}

	freenect_registration reg = freenect_copy_registration(f_dev);

	printf("pix_size: %f | dist: %f \n",reg.zero_plane_info.reference_pixel_size,reg.zero_plane_info.reference_distance);

	cudaMemcpyToSymbol(device::constant::ref_pix_size,&reg.zero_plane_info.reference_pixel_size,sizeof(double));
	cudaMemcpyToSymbol(device::constant::ref_dist,&reg.zero_plane_info.reference_distance,sizeof(double));

	freenect_close_device(f_dev);
}

SyncFreenectSource::~SyncFreenectSource()
{
	// TODO Auto-generated destructor stub
}


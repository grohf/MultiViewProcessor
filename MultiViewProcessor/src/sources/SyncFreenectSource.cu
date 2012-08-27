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
#include <helper_image.h>

#include "../sink/pcd_io.h"

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

			double factor = (2*constant::ref_pix_size * wz)  / constant::ref_dist;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf("factor: %f ref_pix_size: %f ref_dist: %f \n",factor,constant::ref_pix_size,constant::ref_dist);

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

			uchar4 uc4 = make_uchar4(r,g,b,128);
			rgba[z*640*480+y*640+x] = uc4;

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

//	printf("pointer before: %d \n",d_test_xyzi);
//
//	checkCudaErrors(cudaMalloc((void**)&d_test_xyzi,640*480*sizeof(float4)));
//	checkCudaErrors(cudaMalloc((void**)&d_test_rgba,640*480*sizeof(uchar4)));
//
//	printf("pointer after: %d \n",d_test_xyzi);

//	setRegInfo();

	loader.rgb = d_rgb;
	loader.depth = d_depth;

	loader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	loader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);

//	printf("pointer xyzi: %d \n",loader.xyzi);

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y);

	cudaMemcpyToSymbol(device::constant::ref_pix_size,&ref_pix_size,sizeof(double));
	cudaMemcpyToSymbol(device::constant::ref_dist,&ref_dist,sizeof(double));

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

//	printf("pointer rgb: %d \n",loader.rgb);
//	printf("pointer xyzi: %d \n",loader.xyzi);

	device::loadSyncFreenectFrame<<<grid,block>>>(loader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* TEST */

//	size_t uc4s = 640*480*sizeof(uchar4);
////	uchar4 *h_uc4_rgb = (uchar4 *)malloc(uc4s);
////	checkCudaErrors(cudaMemcpy(h_uc4_rgb,loader.rgba,640*480*sizeof(uchar4),cudaMemcpyDeviceToHost));
////
	char path[50];
////	sprintf(path,"/home/avo/pcds/src_rgb%d.ppm",0);
////	sdkSavePPM4ub(path,(unsigned char*)h_uc4_rgb,640,480);
////
	float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
	checkCudaErrors(cudaMemcpy(h_f4_depth,loader.xyzi,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
//
//	uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
//	for(int i=0;i<640*480;i++)
//	{
//		unsigned char g = h_f4_depth[i].z/20;
//		h_uc4_depth[i] = make_uchar4(g,g,g,128);
//	}
//
//	sprintf(path,"/home/avo/pcds/src_depth%d.ppm",0);
//	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);
//
	sprintf(path,"/home/avo/pcds/src_wc_points%d.pcd",0);
	host::io::PCDIOController pcdIOCtrl;
	pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);



	printf("loaded! \n");
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

	setRegInfo();
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

//	printf("pix_size: %f | dist: %f \n",reg.zero_plane_info.reference_pixel_size,reg.zero_plane_info.reference_distance);

//	cudaMemcpyToSymbol(device::constant::ref_pix_size,&(reg.zero_plane_info.reference_pixel_size),sizeof(double));
//	cudaMemcpyToSymbol(device::constant::ref_dist,&(reg.zero_plane_info.reference_distance),sizeof(double));

	ref_pix_size = reg.zero_plane_info.reference_pixel_size;
	ref_dist = reg.zero_plane_info.reference_distance;

	printf("pix_size: %f | dist: %f \n",ref_pix_size,ref_dist);

	freenect_close_device(f_dev);
}

SyncFreenectSource::~SyncFreenectSource()
{
	freenect_sync_stop();
	printf("sync freenect stopped \n");
}


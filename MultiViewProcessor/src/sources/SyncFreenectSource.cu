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

#include "point_info.hpp"
#include "SyncFreenectSource.h"
#include <vector_types.h>

#include <thrust/device_vector.h>

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

//	struct SensorInfoGenerator
//	{
//		SensorInfo *sinfo;
//
//	};

	struct SyncFreenectLoader
	{

		uint8_t *rgb;
		uint16_t *depth;

		float4 *xyzi;
		uchar4 *rgba;
		float *intensity;

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

			int wz = depth[z*640*480+y*640+x];
			float wx,wy;

			convertCXCYWZtoWXWYWZ(x,y,wz,wx,wy);

			unsigned char r = rgb[3*(z*640*480+y*640+x)+0];
			unsigned char g = rgb[3*(z*640*480+y*640+x)+1];
			unsigned char b = rgb[3*(z*640*480+y*640+x)+2];

			float4 f4 = make_float4(wx,wy,wz,0);
			if(wz>0) setValid(f4.w);
			xyzi[z*640*480+y*640+x] = f4;

			intensity[z*640*480+y*640+x] = ((r+r+r+b+g+g+g+g)>>3);

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
//	checkCudaErrors(cudaMalloc((void**)&d_rgb,640*480*3*sizeof(uint8_t)));
//	checkCudaErrors(cudaMalloc((void**)&d_depth,640*480*sizeof(uint16_t)));

//	printf("pointer before: %d \n",d_test_xyzi);
//
//	checkCudaErrors(cudaMalloc((void**)&d_test_xyzi,640*480*sizeof(float4)));
//	checkCudaErrors(cudaMalloc((void**)&d_test_rgba,640*480*sizeof(uchar4)));
//
//	printf("pointer after: %d \n",d_test_xyzi);

//	setRegInfo();

//	loader.rgb = d_rgb;
//	loader.depth = d_depth;


	loader.depth = (uint16_t *)getTargetDataPointer(ImageDepth);
	loader.rgb = (uint8_t *)getTargetDataPointer(ImageRGB);

	loader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	loader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);
	loader.intensity = (float *)getTargetDataPointer(PointIntensity);

//	printf("pointer xyzi: %d \n",loader.xyzi);

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);

	cudaMemcpyToSymbol(device::constant::ref_pix_size,&ref_pix_size,sizeof(double));
	cudaMemcpyToSymbol(device::constant::ref_dist,&ref_dist,sizeof(double));

	SensorInfo sinfo;
	sinfo.pix_size = 2*ref_pix_size;
	sinfo.dist = ref_dist;

	SensorInfo *d_sinfoList = (SensorInfo *)getTargetDataPointer(SensorInfoList);
	checkCudaErrors(cudaMemcpy(d_sinfoList,&sinfo,1*sizeof(SensorInfo),cudaMemcpyHostToDevice));

}

void
SyncFreenectSource::loadFrame()
{

	for(int v=0;v<n_view;v++)
	{
		uint16_t *h_depth = 0;
		uint8_t *h_rgb = 0;
		uint32_t ts;

		freenect_sync_get_depth((void**)&h_depth, &ts, v, FREENECT_DEPTH_REGISTERED);
		freenect_sync_get_video((void**)&h_rgb, &ts, v, FREENECT_VIDEO_RGB);

		checkCudaErrors(cudaMemcpy(loader.rgb + v*640*480*3,h_rgb,640*480*3*sizeof(uint8_t),cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(loader.depth + v*640*480,h_depth,640*480*sizeof(uint16_t),cudaMemcpyHostToDevice));

	}

//	printf("pointer rgb: %d \n",loader.rgb);
//	printf("pointer xyzi: %d \n",loader.xyzi);

	device::loadSyncFreenectFrame<<<grid,block>>>(loader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* TEST */


//	char path[50];
//
//	for(int i=0;i<n_view;i++)
//	{
//		float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
//		checkCudaErrors(cudaMemcpy(h_f4_depth,loader.xyzi+i*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
//
//		sprintf(path,"/home/avo/pcds/src_wc_points_%d.pcd",i);
//		host::io::PCDIOController pcdIOCtrl;
//		pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
//	}


	printf("loaded! \n");
}

SyncFreenectSource::SyncFreenectSource(unsigned int n_view_)
{
	n_view = n_view_;

	DeviceDataParams imageDepthParams;
	imageDepthParams.elements = 640*480*n_view;
	imageDepthParams.element_size = sizeof(uint16_t);
	imageDepthParams.dataType = Point1D;
	imageDepthParams.elementType = UINT1;
	addTargetData(addDeviceDataRequest(imageDepthParams),ImageDepth);

	DeviceDataParams imageRGBParams;
	imageRGBParams.elements = 640*480*n_view;
	imageRGBParams.element_size = 3*sizeof(uint8_t);
	imageRGBParams.dataType = Point3D;
	imageRGBParams.elementType = UINT3;
	addTargetData(addDeviceDataRequest(imageRGBParams),ImageRGB);


	DeviceDataParams pointXYZIParams;
	pointXYZIParams.elements = 640*480*n_view;
	pointXYZIParams.element_size = sizeof(float4);
	pointXYZIParams.dataType = Point4D;
	pointXYZIParams.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(pointXYZIParams),PointXYZI);

	DeviceDataParams pointRGBAParams;
	pointRGBAParams.elements = 640*480*n_view;
	pointRGBAParams.element_size = sizeof(uchar4);
	pointRGBAParams.dataType = Point4D;
	pointRGBAParams.elementType = UCHAR4;

	addTargetData(addDeviceDataRequest(pointRGBAParams),PointRGBA);

	DeviceDataParams intensityParams;
	intensityParams.elements = 640*480*n_view;
	intensityParams.element_size = sizeof(float);
	intensityParams.dataType = Point1D;
	intensityParams.elementType = FLOAT1;
	addTargetData(addDeviceDataRequest(intensityParams),PointIntensity);

	DeviceDataParams sensorInfoListParams;
	sensorInfoListParams.elements = n_view;
	sensorInfoListParams.element_size = sizeof(SensorInfo);
	sensorInfoListParams.dataType = ListItem;
	sensorInfoListParams.elementType = SensorInfoItem;

	addTargetData(addDeviceDataRequest(sensorInfoListParams),SensorInfoList);

	setRegInfo();

	printf("discarding first 3 image sets... \n");
	for(int i=0;i<3;i++)
	{
		for(int v=0;v<n_view;v++)
		{
			uint16_t *h_depth = 0;
			uint8_t *h_rgb = 0;
			uint32_t ts;

			freenect_sync_get_depth((void**)&h_depth, &ts, v, FREENECT_DEPTH_REGISTERED);
			freenect_sync_get_video((void**)&h_rgb, &ts, v, FREENECT_VIDEO_RGB);
		}
	}
	printf("done! \n");
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


void
SyncFreenectSource::TestSyncnectSource(thrust::host_vector<float4>& views)
{

//	uint16_t *h_depth = 0;
//	uint8_t *h_rgb = 0;
//	uint32_t ts;
//
//	freenect_sync_get_depth((void**)&h_depth, &ts, 0, FREENECT_DEPTH_REGISTERED);
//	freenect_sync_get_video((void**)&h_rgb, &ts, 0, FREENECT_VIDEO_RGB);
//
//	thrust::host_vector<uint16_t> h_depth2(640*480*n_view);
//	thrust::host_vector<uint8_t> h_rgb2(640*480*3*n_view);
//	for(int i=0;i<h_depth2.size();i++)
//	{
//		h_depth2[i] = h_depth[i];
//	}
//	for(int i=0;i<h_rgb2.size();i++)
//	{
//		h_rgb2[i] = h_rgb[i];
//	}
//
//	thrust::device_vector<uint16_t> d_depth2 = h_depth2;
//	thrust::device_vector<uint8_t> d_rgb2 = h_rgb2;
//
////	thrust::device_vector<uint16_t> d_depth2(640*480*n_view);
////	thrust::device_vector<uint8_t> d_rgb2(640*480*n_view);
////
////	loader.depth = thrust::raw_pointer_cast(d_depth2.data());
////	loader.rgb = thrust::raw_pointer_cast(d_rgb2.data());
////
////	for(int v=0;v<n_view;v++)
////	{
////		uint16_t *h_depth = 0;
////		uint8_t *h_rgb = 0;
////		uint32_t ts;
////
////		freenect_sync_get_depth((void**)&h_depth, &ts, v, FREENECT_DEPTH_REGISTERED);
////		freenect_sync_get_video((void**)&h_rgb, &ts, v, FREENECT_VIDEO_RGB);
////
////		checkCudaErrors(cudaMemcpy(loader.rgb + v*640*480*3,h_rgb,640*480*3*sizeof(uint8_t),cudaMemcpyHostToDevice));
////		checkCudaErrors(cudaMemcpy(loader.depth + v*640*480,h_depth,640*480*sizeof(uint16_t),cudaMemcpyHostToDevice));
////
////	}
//
//	thrust::device_vector<float4> d_xyzi(640*480*n_view);
//	thrust::device_vector<uchar4> d_rgba(640*480*n_view);
//	thrust::device_vector<float> d_intensity(640*480*n_view);
//
//	loader.xyzi = thrust::raw_pointer_cast(d_xyzi.data());
//	loader.rgba = thrust::raw_pointer_cast(d_rgba.data());
//	loader.intensity = thrust::raw_pointer_cast(d_intensity.data());
//
////	printf("pointer xyzi: %d \n",loader.xyzi);
//
//	block = dim3(32,24);
//	grid = dim3(640/block.x,480/block.y,n_view);
//
//	cudaMemcpyToSymbol(device::constant::ref_pix_size,&ref_pix_size,sizeof(double));
//	cudaMemcpyToSymbol(device::constant::ref_dist,&ref_dist,sizeof(double));
//
//	SensorInfo sinfo;
//	sinfo.pix_size = 2*ref_pix_size;
//	sinfo.dist = ref_dist;
//
//
////	thrust::device_vector<SensorInfo> d_sinfo2(1*sizeof(SensorInfo));
////	SensorInfo *d_sinfoList = thrust::raw_pointer_cast(d_sinfo2.data());
////	checkCudaErrors(cudaMemcpy(d_sinfoList,&sinfo,1*sizeof(SensorInfo),cudaMemcpyHostToDevice));
//
//	device::loadSyncFreenectFrame<<<grid,block>>>(loader);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());


	uint16_t *h_depth = 0;
	uint8_t *h_rgb = 0;
	uint32_t ts;

	freenect_sync_get_depth((void**)&h_depth, &ts, 0, FREENECT_DEPTH_REGISTERED);
	freenect_sync_get_video((void**)&h_rgb, &ts, 0, FREENECT_VIDEO_RGB);

	for(int i=0;i<views.size()/2;i++)
	{
		int y = i/640;
		int x = i - y*640;
		float wz = h_depth[i];

//			if(wz > 0) printf("%f \n",wz);

		float wx = (x-320.) * ((0.2084 * wz)/(120.));
		float wy = (y-320.) * ((0.2084 * wz)/(120.));

		views[i].x = wx;
		views[i].y = wy;
		views[i].z = wz;
		views[i].w = 0;
	}


	char path[50];
	host::io::PCDIOController pcdIOCtrl;

	sprintf(path,"/home/avo/pcds/src_points_%d.pcd",0);
	pcdIOCtrl.writeASCIIPCD(path,(float *)views.data(),640*480);

}

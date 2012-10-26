/*
 * SynthRGBDBenchmarkSource.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: avo
 */

#include "SynthRGBDBenchmarkSource.h"
#include "lodepng.h"
#include "point_info.hpp"


#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//TODO: get deviceMemory stuff out here
#include <helper_cuda.h>
#include <helper_image.h>

#include "../sink/pcd_io.h"

namespace device
{
	struct SynthWCLoader
	{

		uint8_t *rgb;
		float *depth;

		float4 *xyzi;
		uchar4 *rgba;
		float *intensity;

		float fx,cx;
		float fy,cy;


		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZ(int ix,int iy, int wz, float &wx, float &wy) const
		{

			double factor = wz/1.f;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf("factor: %f ref_pix_size: %f ref_dist: %f \n",factor,constant::ref_pix_size,constant::ref_dist);

			wx = (float) (((ix - cx) * factor)/fx);
			wy = (float) (((iy - cy) * factor)/fy);

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
	__global__ void loadSynthFrame(const SynthWCLoader sfl){ sfl(); }
}

device::SynthWCLoader synthRGBDLoader;

SynthRGBDBenchmarkSource::SynthRGBDBenchmarkSource(unsigned int n_view_) : n_view(n_view_)
{
	DeviceDataParams imageDepthParams;
	imageDepthParams.elements = 640*480*n_view;
	imageDepthParams.element_size = sizeof(float);
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

}

void SynthRGBDBenchmarkSource::init()
{
	synthRGBDLoader.depth = (float *)getTargetDataPointer(ImageDepth);
	synthRGBDLoader.rgb = (uint8_t *)getTargetDataPointer(ImageRGB);

	synthRGBDLoader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	synthRGBDLoader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);
	synthRGBDLoader.intensity = (float *)getTargetDataPointer(PointIntensity);

	// Camera Freiburg 3 RGB
	synthRGBDLoader.fx = 535.4f;
	synthRGBDLoader.fy = 539.2f;
	synthRGBDLoader.cx = 320.1f;
	synthRGBDLoader.cy = 247.6f;

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);

}

void SynthRGBDBenchmarkSource::loadFrame()
{
//	uint16_t *h_depth = 0;
//	uint8_t *h_rgb = 0;
	printf("loadFrameSynthRGBD... \n");

	thrust::host_vector<float> h_depth(640*480*n_view);
	thrust::host_vector<float> h_rgb(640*480*3*n_view);

	const char* filename = "test2.png";
	std::vector<unsigned char> image; //the raw pixels
	unsigned width, height;

	//decode
	printf("load... \n");
	unsigned error = lodepng::decode(image, width, height, filename,LCT_GREY,16);
	printf("loaded! w: %d | h: %d \n",width,height);

	float max = 0.f;
	for(int i=0;i<640*480;i++)
	{
		h_depth[i] = (float)(((image[i*2]<<8) + image[i*2+1])/5.f);
		if(h_depth[i]>max)max = h_depth[i];
	}
	printf("max: %f \n",max);


	char path[50];
	uchar4 *h_uc4_depth = (uchar4 *)malloc(640*480*sizeof(uchar4));
	for(int i=0;i<640*480;i++)
	{
		unsigned char g = (h_depth[i]/10000.f)*255.f;
		h_uc4_depth[i] = make_uchar4(g,g,g,128);
	}
	sprintf(path,"/home/avo/pcds/synth_depth%d.ppm",0);
	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);


	checkCudaErrors(cudaMemcpy(synthRGBDLoader.rgb,h_rgb.data(),640*480*3*sizeof(uint8_t),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(synthRGBDLoader.depth,h_depth.data(),640*480*sizeof(float),cudaMemcpyHostToDevice));


	device::loadSynthFrame<<<grid,block>>>(synthRGBDLoader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	for(int i=0;i<n_view;i++)
	{
		float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
		checkCudaErrors(cudaMemcpy(h_f4_depth,synthRGBDLoader.xyzi+i*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

		sprintf(path,"/home/avo/pcds/synth_wc_points_%d.pcd",i);
		host::io::PCDIOController pcdIOCtrl;
		pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
	}

	printf("data loaded... \n");
}



SynthRGBDBenchmarkSource::~SynthRGBDBenchmarkSource()
{
	// TODO Auto-generated destructor stub
}


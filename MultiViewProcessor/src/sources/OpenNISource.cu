/*
 * OpenNISource.cpp
 *
 *  Created on: Jan 12, 2013
 *      Author: avo
 */

#include "OpenNISource.h"
#include "point_info.hpp"

#include <helper_cuda.h>
#include <helper_image.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>

#include <iostream>
#include <fstream>
#include <cstring>

#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>



#include "../sink/pcd_io.h"
//#include "../debug/EigenCheckClass.h"

namespace device
{

	struct OpenNILoader
	{

		uint8_t *rgb;
		uint16_t *depth;

		float4 *xyzi;
		uchar4 *rgba;
		float *intensity;

		SensorInfoV2 *sensorInfos;

		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZOI(int ix,int iy, int wz, float &wx, float &wy, int view) const
		{

			double factor = wz/1.f;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf("factor: %f ref_pix_size: %f ref_dist: %f \n",factor,constant::ref_pix_size,constant::ref_dist);

			wx = (float) (((ix - sensorInfos[view].cx) * factor)/sensorInfos[view].fx);
			wy = (float) (((iy - sensorInfos[view].cy) * factor)/sensorInfos[view].fy);

		}

		__device__ __forceinline__ void
		operator () () const
		{
			int x = blockIdx.x*blockDim.x + threadIdx.x;
			int y = blockIdx.y*blockDim.y + threadIdx.y;
			int z = blockIdx.z;

			int wz = depth[z*640*480+y*640+x];
			float wx,wy;

//			if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.z==0 && blockIdx.x==0 && blockIdx.y==0)
//			{
//				for(int v=0;v<2;v++)
//				{
//					printf("(%d)->",v);
//					for(int j=0;j<12;j++)
//						printf("%f |",transformationmatrices[v*12+j]);
//					printf("\n");
//
//					SensorInfoV2 sensor = sensorInfos[v];
//					printf("(%d)sensor info: %f %f %f %f \n",v,sensor.fx,sensor.fy,sensor.cx,sensor.cy);
//				}
//
//			}

			convertCXCYWZtoWXWYWZOI(x,y,wz,wx,wy,blockIdx.z);
			float tx,ty,tz;

			tx = wx;
			ty = wy;
			tz = wz;

			unsigned char r = rgb[3*(z*640*480+y*640+x)+0];
			unsigned char g = rgb[3*(z*640*480+y*640+x)+1];
			unsigned char b = rgb[3*(z*640*480+y*640+x)+2];

			float4 f4 = make_float4(tx,ty,tz,0);
			if(wz>0) setValid(f4.w);
			xyzi[z*640*480+y*640+x] = f4;

			intensity[z*640*480+y*640+x] = ((r+r+r+b+g+g+g+g)>>3);

			uchar4 uc4 = make_uchar4(r,g,b,128);
			rgba[z*640*480+y*640+x] = uc4;

		}
	};
	__global__ void loadTestOpenNI(const OpenNILoader to){to();}
}

device::OpenNILoader openNILoader;
//device::OpenNIWCLoader openNIWCLoader;

using namespace xn;
#define NUM_OF_SENSORS 10
#define USE_RGB_IMAGE_STREAMS 1

XnFPSData xnFPS;
xn::Context g_Context;

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define CHECK_RC(rc, what)                                      \
if (rc != XN_STATUS_OK)                                         \
{                                                               \
    printf("%s failed: %s\n", what, xnGetStatusString(rc));     \
    return rc;                                                  \
}

#define CHECK_RC_CONTINUE(rc, what)                             \
if (rc != XN_STATUS_OK)                                         \
{                                                               \
 printf("%s failed: %s\n", what, xnGetStatusString(rc));        \
}
//---------------------------------------------------------------------------

struct DepthRgbSensors
{
        char name[80];
        xn::ProductionNode device;
        xn::DepthGenerator depth;
        xn::DepthMetaData depthMD;
#if defined(USE_RGB_IMAGE_STREAMS)
        xn::ImageGenerator image;
        xn::ImageMetaData imageMD;
#endif
};

DepthRgbSensors sensors[NUM_OF_SENSORS];

void CloseDevice()
{
        //Do additional cleanup if needed
        g_Context.Release();
}

void UpdateCommon(DepthRgbSensors &sensor)
{
//        XnStatus rc = XN_STATUS_OK;
//        rc = g_Context.WaitAnyUpdateAll();
//        //This function will wait until all nodes will have new data:
//        //rc = g_Context.WaitAndUpdateAll();
//        CHECK_RC_CONTINUE(rc, "WaitAnyUpdateAll() failed");

//		if (sensor.image.IsValid() && sensor.depth.IsValid())
//		{
//				sensor.image.GetMetaData(sensor.imageMD);
//				if(sensor.depth.GetAlternativeViewPointCap().IsViewPointSupported(sensor.image))
//				{
//					sensor.depth.GetAlternativeViewPointCap().SetViewPoint(sensor.image);
//				}
//		}

        if (sensor.depth.IsValid())
        {
                sensor.depth.GetMetaData(sensor.depthMD);
        }
#if defined(USE_RGB_IMAGE_STREAMS)
        if (sensor.image.IsValid())
        {
                sensor.image.GetMetaData(sensor.imageMD);
        }
#endif
}


void
OpenNISource::init()
{
	openNILoader.depth = (uint16_t *)getTargetDataPointer(ImageDepth);
	openNILoader.rgb = (uint8_t *)getTargetDataPointer(ImageRGB);

	openNILoader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	openNILoader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);
	openNILoader.intensity = (float *)getTargetDataPointer(PointIntensity);

	SensorInfoV2 s[n_view];
	for(int v=0;v<n_view;v++)
	{
//		s[v].fx = 535.4f;
//		s[v].fy = 539.2f;
//		s[v].cx = 320.1f;
//		s[v].cy = 247.6f;

		s[v].fx = 525.0;
		s[v].fy = 525.0;
		s[v].cx = 319.5;
		s[v].cy = 239.5;
	}

	openNILoader.sensorInfos = (SensorInfoV2 *)getTargetDataPointer(SensorInfoList);
//	SensorInfoV2 *d_sinfoList = (SensorInfoV2 *)getTargetDataPointer(SensorInfoList);
	checkCudaErrors(cudaMemcpy(openNILoader.sensorInfos,&s,n_view*sizeof(SensorInfoV2),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(d_sinfoList+1,&s,1*sizeof(SensorInfoV2),cudaMemcpyHostToDevice));


	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);

	XnStatus nRetVal = XN_STATUS_OK;

   // Getting Sensors information and configure all sensors
   nRetVal = g_Context.Init();

   xn::NodeInfoList devicesList;
   int devicesListCount = 0;
   nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEVICE, NULL, devicesList);
   for (xn::NodeInfoList::Iterator it = devicesList.Begin(); it != devicesList.End(); ++it)
   {
	   devicesListCount++;
   }
//   CHECK_RC(nRetVal, "Enumerate");
   int i=0;
   for (xn::NodeInfoList::Iterator it = devicesList.Begin(); it != devicesList.End(); ++it, ++i)
   {
	   // Create the device node
	   xn::NodeInfo deviceInfo = *it;
		   nRetVal = g_Context.CreateProductionTree(deviceInfo, sensors[i].device);
//		   CHECK_RC(nRetVal, "Create Device");

	   // Create a query to depend on this node
	   Query query;
	   query.AddNeededNode(deviceInfo.GetInstanceName());

	   // Copy the device name
	   xnOSMemCopy(sensors[i].name,deviceInfo.GetInstanceName(),
				   xnOSStrLen(deviceInfo.GetInstanceName()));
	   // Now create a depth generator over this device
	   nRetVal = g_Context.CreateAnyProductionTree(XN_NODE_TYPE_DEPTH, &query, sensors[i].depth);
//	   CHECK_RC(nRetVal, "Create Depth");
#if defined(USE_RGB_IMAGE_STREAMS)
		   // now create a image generator over this device
		   nRetVal = g_Context.CreateAnyProductionTree(XN_NODE_TYPE_IMAGE, &query, sensors[i].image);
//		   CHECK_RC(nRetVal, "Create Image");

		   sensors[i].depth.GetAlternativeViewPointCap().SetViewPoint(sensors[i].image);
//           sensors[i].depth.GetFrameSyncCap().FrameSyncWith(sensors[i].image);

		   XnUInt64 zpd;
		   XnDouble zpps;
		   sensors[i].depth.GetIntProperty("ZPD",zpd);
		   sensors[i].depth.GetRealProperty("ZPPS",zpps);
		   printf("sensor %d: zpd: %d zpps: %f \n",i,zpd,zpps);
#endif
   }

   g_Context.StartGeneratingAll();
   nRetVal = xnFPSInit(&xnFPS, 180);
//   CHECK_RC(nRetVal, "FPS Init");

}

unsigned int count = 0;

void
OpenNISource::loadFrame()
{
	g_Context.WaitAndUpdateAll();
//	   g_Context.WaitNoneUpdateAll();
//	   g_Context.WaitAnyUpdateAll();

	for(int i = 0 ; i < n_view ; ++i)
	{
	   printf("Sensor: [%d] ",i);
	   UpdateCommon(sensors[i]);
	   xnFPSMarkFrame(&xnFPS);
	   // Print Depth central pixel
//	   const DepthMetaData *dmd = &sensors[i].depthMD;
//	   const ImageMetaData *imd = &sensors[i].imageMD;

	   const uint16_t* pDepthMap = sensors[i].depthMD.Data();
	   const uint8_t *imageMap = sensors[i].imageMD.Data();
//	   printf("Frame:%d Depth: %u. FPS: %f\n",
//					   dmd->FrameID(), sensors[i].depthMD(dmd->XRes() / 2, dmd->YRes() / 2),
//					   xnFPSCalc(&xnFPS));

	   checkCudaErrors(cudaMemcpy(openNILoader.depth+i*640*480,pDepthMap,640*480*sizeof(uint16_t),cudaMemcpyHostToDevice));
	   checkCudaErrors(cudaMemcpy(openNILoader.rgb+i*640*480*3,imageMap,3*640*480*sizeof(uint8_t),cudaMemcpyHostToDevice));

//			   // Print Image first pixel
//			   const ImageMetaData *imd = &sensors[i].imageMD;
//			   const XnUInt8 *imageMap = sensors[i].imageMD.Data();
//	//                   printf("Image frame [%d] first pixel is: R[%u],G[%u],B[%u]. FPS: %f\n",
//	//                           imd->FrameID(), imageMap[0], imageMap[1], imageMap[2], xnFPSCalc(&xnFPS));

//			   uchar4 *h_uc4 = (uchar4 *)malloc(640*480*sizeof(uchar4));
//			   for(int j=0;j<640*480;j++)
//			   {
//				   unsigned char grey = (pDepthMap[j]/10000.f)*255.0f;
//				   h_uc4[j].x = grey;
//				   h_uc4[j].y = grey;
//				   h_uc4[j].z = grey;
//				   h_uc4[j].w = 127.f;
//			   }
//			char path[50];
//			sprintf(path,"/home/avo/pcds/openNITestData/openNIsrc_%d_%d.ppm",count,i);
//			sdkSavePPM4ub(path,(unsigned char *)h_uc4,640,480);
//			printf("123\n ");
	}
	count ++;

	device::loadTestOpenNI<<<grid,block>>>(openNILoader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	char path[50];
	if(outputlevel>0)
	{
		for(int i=0;i<n_view;i++)
		{
			float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
			checkCudaErrors(cudaMemcpy(h_f4_depth,openNILoader.xyzi+i*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

//			for(int p=0;p<640*480;p++)
//			{
//				float4 t = h_f4_depth[p];
//				h_f4_depth[p].x = h_transformationMatrices[i*12+0]*t.x + h_transformationMatrices[i*12+1]*t.y + h_transformationMatrices[i*12+2]*t.z + h_transformationMatrices[i*12+9];
//				h_f4_depth[p].y = h_transformationMatrices[i*12+3]*t.x + h_transformationMatrices[i*12+4]*t.y + h_transformationMatrices[i*12+5]*t.z + h_transformationMatrices[i*12+10];
//				h_f4_depth[p].z = h_transformationMatrices[i*12+6]*t.x + h_transformationMatrices[i*12+7]*t.y + h_transformationMatrices[i*12+8]*t.z + h_transformationMatrices[i*12+11];
//			}


			sprintf(path,"/home/avo/pcds/openNI_wc_points_%d.pcd",i);
			host::io::PCDIOController pcdIOCtrl;
			pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
		}
	}

		printf("data loaded... \n");
}


OpenNISource::OpenNISource(unsigned int n_view_,int output) : n_view(n_view_),outputlevel(output)
{
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
	sensorInfoListParams.element_size = sizeof(SensorInfoV2);
	sensorInfoListParams.dataType = ListItem;
	sensorInfoListParams.elementType = SensorInfoItem;
	addTargetData(addDeviceDataRequest(sensorInfoListParams),SensorInfoList);
}

OpenNISource::~OpenNISource()
{
	// TODO Auto-generated destructor stub
}


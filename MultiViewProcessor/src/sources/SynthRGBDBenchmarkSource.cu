/*
 * SynthRGBDBenchmarkSource.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: avo
 */

#include "SynthRGBDBenchmarkSource.h"
#include "lodepng.h"
#include "point_info.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <fstream>
#include <cstring>

#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

//TODO: get deviceMemory stuff out here
#include <helper_cuda.h>
#include <helper_image.h>

#include "../sink/pcd_io.h"
#include "../debug/EigenCheckClass.h"

namespace device
{
	struct SynthWCLoader
	{

		uint8_t *rgb;
		float *depth;

		float4 *xyzi;
		uchar4 *rgba;
		float *intensity;

//		float fx,cx;
//		float fy,cy;

		SensorInfoV2 *sensorInfos;

		float *transformationmatrices;

		bool transformToWorld;

		__device__ __forceinline__ void
		convertCXCYWZtoWXWYWZ(int ix,int iy, int wz, float &wx, float &wy, int view) const
		{

			double factor = wz/1.f;

//			if(blockIdx.x==10&&blockIdx.y==10&&threadIdx.x==10&&threadIdx.y==10)
//				printf("factor: %f ref_pix_size: %f ref_dist: %f \n",factor,constant::ref_pix_size,constant::ref_dist);

			wx = (float) (((ix - sensorInfos[view].cx) * factor)/sensorInfos[view].fx);
			wy = (float) (((iy - sensorInfos[view].cy) * factor)/sensorInfos[view].fy);

		}

		__device__ __forceinline__ void
		transform(float& tx, float& ty, float& tz, float wx, float wy, float wz, int view) const
		{
//			wx -= transformationmatrices[view*12+9];
//			wy -= transformationmatrices[view*12+10];
//			wz -= transformationmatrices[view*12+11];

			tx = transformationmatrices[view*12+0] * wx + transformationmatrices[view*12+1] * wy + transformationmatrices[view*12+2] * wz + transformationmatrices[view*12+9];
			ty = transformationmatrices[view*12+3] * wx + transformationmatrices[view*12+4] * wy + transformationmatrices[view*12+5] * wz + transformationmatrices[view*12+10];
			tz = transformationmatrices[view*12+6] * wx + transformationmatrices[view*12+7] * wy + transformationmatrices[view*12+8] * wz + transformationmatrices[view*12+11];
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

			convertCXCYWZtoWXWYWZ(x,y,wz,wx,wy,blockIdx.z);
			float tx,ty,tz;
			if(transformToWorld)
				transform(tx,ty,tz,wx,wy,wz,z);
			else
			{
				tx = wx;
				ty = wy;
				tz = wz;
			}
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
	__global__ void loadSynthFrame(const SynthWCLoader sfl){ sfl(); }
}

device::SynthWCLoader synthRGBDLoader;

SynthRGBDBenchmarkSource::SynthRGBDBenchmarkSource(unsigned int n_view_,char *baseDir,bool transform,bool output) : n_view(n_view_), baseDir(baseDir),transform(transform), output(output)
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
	sensorInfoListParams.element_size = sizeof(SensorInfoV2);
	sensorInfoListParams.dataType = ListItem;
	sensorInfoListParams.elementType = SensorInfoItem;
	addTargetData(addDeviceDataRequest(sensorInfoListParams),SensorInfoList);

	DeviceDataParams transformationMatricesListParams;
	transformationMatricesListParams.elements = n_view;
	transformationMatricesListParams.element_size = 12*sizeof(float);
	transformationMatricesListParams.dataType = Matrix;
	transformationMatricesListParams.elementType = FLOAT1;
	addTargetData(addDeviceDataRequest(transformationMatricesListParams),GroundTruthTransformations);
}

void SynthRGBDBenchmarkSource::init()
{
	synthRGBDLoader.depth = (float *)getTargetDataPointer(ImageDepth);
	synthRGBDLoader.rgb = (uint8_t *)getTargetDataPointer(ImageRGB);

	synthRGBDLoader.xyzi = (float4 *)getTargetDataPointer(PointXYZI);
	synthRGBDLoader.rgba = (uchar4 *)getTargetDataPointer(PointRGBA);
	synthRGBDLoader.intensity = (float *)getTargetDataPointer(PointIntensity);

	SensorInfoV2 s[n_view];
	for(int v=0;v<n_view;v++)
	{
		s[v].fx = 535.4f;
		s[v].fy = 539.2f;
		s[v].cx = 320.1f;
		s[v].cy = 247.6f;
	}

	synthRGBDLoader.sensorInfos = (SensorInfoV2 *)getTargetDataPointer(SensorInfoList);
//	SensorInfoV2 *d_sinfoList = (SensorInfoV2 *)getTargetDataPointer(SensorInfoList);
	checkCudaErrors(cudaMemcpy(synthRGBDLoader.sensorInfos,&s,n_view*sizeof(SensorInfoV2),cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(d_sinfoList+1,&s,1*sizeof(SensorInfoV2),cudaMemcpyHostToDevice));


	// Camera Freiburg 3 RGB
//	synthRGBDLoader.fx = 535.4f;
//	synthRGBDLoader.fy = 539.2f;
//	synthRGBDLoader.cx = 320.1f;
//	synthRGBDLoader.cy = 247.6f;

	synthRGBDLoader.transformToWorld = transform;

	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);



}

const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* const DELIMITER = " ";

void SynthRGBDBenchmarkSource::loadFrame()
{
//	uint16_t *h_depth = 0;
//	uint8_t *h_rgb = 0;
	printf("loadFrameSynthRGBD... \n");

	thrust::host_vector<float> h_depth(640*480*n_view);
	thrust::host_vector<uint8_t> h_rgb(640*480*3*n_view);

	std::vector<unsigned char> image; //the raw pixels
	unsigned width, height;

	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<unsigned int> dist(0, 50);

	unsigned int depth_lines[n_view];

	srand ( time(NULL) );

	depth_lines[0] = (unsigned int) (rand() % 100);

	for(int i=1;i<n_view;i++)
	{
		depth_lines[i]= depth_lines[i-1] + (unsigned int) (rand() % 80);
	}

//	depth_lines[0] = 0;
//	depth_lines[1] = 20;

	EigenCheckClass::setFrameValues(depth_lines[0],depth_lines[1]);

	std::string base(baseDir);
	printf("base: %s \n",base.data());

	std::ifstream fin_gt,fin_depth,fin_rgb;
	fin_gt.open( (base+std::string("/groundtruth.txt")).data() );
	fin_depth.open((base+std::string("/depth.txt")).data());
	fin_rgb.open((base+std::string("/rgb.txt")).data());
	if (!fin_gt.good() || !fin_depth.good() || !fin_rgb.good())
	{
		printf("file not found \n");
		return;
	}

	//get min_sensor_ts
	double min_sensor_ts;

	bool cond = true;
	while (!fin_gt.eof() && cond)
	{
		char buf[MAX_CHARS_PER_LINE];
		fin_gt.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE] = {0};
		token[0] = strtok(buf, DELIMITER);
		if(strcmp(token[0],"#"))
		{
			min_sensor_ts = atof(token[0]);
			printf("min_sensor_ts: %f \n",min_sensor_ts);
//			oldSensorLineBuffer = buf;
			cond = false;
		}
	}

	unsigned int l = 0;
	unsigned int linecount = 0;
	double depth_ts[n_view];
	cond = true;

	//DEPTH
	while (!fin_depth.eof() && cond)
	{
		char buf[MAX_CHARS_PER_LINE];
		fin_depth.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE] = {0};
		token[0] = strtok(buf, DELIMITER);
//		printf("%s! > %d \n",token[0],strcmp(token[0],"#"));

		if(strcmp(token[0],"#") && (atof(token[0]) > min_sensor_ts))
		{
			if(linecount==depth_lines[l])
			{
				depth_ts[l] = atof(token[0]);
				token[1] = strtok(0, DELIMITER);

				std::string spath = (base+std::string(token[1]));
				image.clear();
				unsigned error = lodepng::decode(image, width, height, spath.data(),LCT_GREY,16);
//				printf("error: %d \n",error);
				for(int i=0;i<640*480;i++)
				{
					h_depth[l*640*480+i] = (float)(((image[i*2]<<8) + image[i*2+1])/5.f);
				}

//				printf("(%d) loaded %s ->  w: %d | h: %d -> %f \n",l,spath.data(),width,height,h_depth[l*640*480+640*200+300]);

				if(++l==n_view)
					cond = false;
			}

			linecount++;
		}
	}

	//RGB
	cond = true;
	l = 0;
	double 	ts_tmp = 0,
			ts_buff = 0;
	std::string pathBuff;
	while (!fin_rgb.eof() && cond)
	{
		char buf[MAX_CHARS_PER_LINE];
		fin_rgb.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE] = {0};
		token[0] = strtok(buf, DELIMITER);

		if(strcmp(token[0],"#"))// && (  (ts_tmp=atof(token[0]))>min_sensor_ts))
		{
//			printf("ts_tmp: %f \n",ts_tmp);

			ts_tmp=atof(token[0]);
			token[1] = strtok(0, DELIMITER);
			if(ts_buff>0)
			{
				while(ts_buff<=depth_ts[l] && ts_tmp >= depth_ts[l])
				{

					if(depth_ts[l]-ts_buff < ts_tmp-depth_ts[l])
					{
						printf("low: %s \n",pathBuff.data());

						image.clear();
						unsigned error = lodepng::decode(image, width, height, (base+pathBuff).data(),LCT_RGB,8);
						for(int i=0;i<640*480;i++)
						{
//							h_rgb[l*640*480+i] = (float)(((image[i*2]<<8) + image[i*2+1])/5.f);
							h_rgb[(l*640*480+i)*3+0] = image[i*3+0];
							h_rgb[(l*640*480+i)*3+1] = image[i*3+1];
							h_rgb[(l*640*480+i)*3+2] = image[i*3+2];
						}
					}
					else
					{
						printf("high: %s \n",token[1]);
						image.clear();
						unsigned error = lodepng::decode(image, width, height, (base+std::string(token[1])).data(),LCT_RGB,8);
						for(int i=0;i<640*480;i++)
						{
//							h_rgb[l*640*480+i] = (float)(((image[i*2]<<8) + image[i*2+1])/5.f);
							h_rgb[(l*640*480+i)*3+0] = image[i*3+0];
							h_rgb[(l*640*480+i)*3+1] = image[i*3+1];
							h_rgb[(l*640*480+i)*3+2] = image[i*3+2];
						}
					}
					l++;
				}
			}
			ts_buff = ts_tmp;
			pathBuff = std::string(token[1]);

			if(l==n_view)
				cond = false;
		}
	}

	printf("quats \n");
	//Quaternions
	thrust::host_vector<float> trans_quaternions(n_view*7);
	thrust::host_vector<float> h_transformationMatrices(n_view*12);

	cond = true;
	l = 0;

	std::vector<float> trans_quat_current(7);
	std::vector<float> trans_quat_buffer(7);
	while (!fin_gt.eof() && cond)
	{
		char buf[MAX_CHARS_PER_LINE];
		fin_gt.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE] = {0};
		token[0] = strtok(buf, DELIMITER);

		if(strcmp(token[0],"#"))
		{

			ts_tmp=atof(token[0]);

			for(int tq=1;tq<8;tq++)
			{
				token[tq] = strtok(0, DELIMITER);
				trans_quat_current[tq-1]=atof(token[tq]);
			}

			if(ts_buff>0)
			{
				while(ts_buff<=depth_ts[l] && ts_tmp >= depth_ts[l])
				{

					if(depth_ts[l]-ts_buff < ts_tmp-depth_ts[l])
					{
						printf("low: %f \n",ts_buff);
						for(int k=0;k<7;k++)
							trans_quaternions[l*7+k] = trans_quat_buffer[k];
					}
					else
					{
						printf("high: %f \n",ts_tmp);
						for(int k=0;k<7;k++)
							trans_quaternions[l*7+k] = trans_quat_current[k];
					}
					l++;
				}
			}
			ts_buff = ts_tmp;
			trans_quat_buffer = trans_quat_current;

			if(l==n_view)
				cond = false;
		}
	}


//	const char* filename = "d1341841874.245967.png";
////	std::vector<unsigned char> image; //the raw pixels
////	unsigned width, height;
//
//	//decode
//	printf("load... \n");
//	unsigned error = lodepng::decode(image, width, height, filename,LCT_GREY,16);
//	printf("loaded! w: %d | h: %d \n",width,height);
//
//	float max = 0.f;
//	for(int i=0;i<640*480;i++)
//	{
//		h_depth[i] = (float)(((image[i*2]<<8) + image[i*2+1])/5.f);
//		if(h_depth[i]>max)max = h_depth[i];
//	}
//	printf("max: %f \n",max);




	char path[50];
	if(output)
	{
		uchar4 *h_uc4_depth = (uchar4 *)malloc(640*480*sizeof(uchar4));
		uchar4 *h_uc4_rgb = (uchar4 *)malloc(640*480*sizeof(uchar4));
		for(int v=0;v<n_view;v++)
		{
			for(int i=0;i<640*480;i++)
			{
				unsigned char g = (h_depth[v*640*480+i]/10000.f)*255.f;
				h_uc4_depth[i] = make_uchar4(g,g,g,128);

				h_uc4_rgb[i] = make_uchar4(h_rgb[(v*640*480+i)*3+0],h_rgb[(v*640*480+i)*3+1],h_rgb[(v*640*480+i)*3+2],128);
			}
			sprintf(path,"/home/avo/pcds/synth_depth%d.ppm",v);
			sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);

			sprintf(path,"/home/avo/pcds/synth_rgb%d.ppm",v);
			sdkSavePPM4ub(path,(unsigned char*)h_uc4_rgb,640,480);

	//		sprintf(path,"/home/avo/pcds/synth_depth%d.ppm",v);
	//		sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);
		}
	}

	checkCudaErrors(cudaMemcpy(synthRGBDLoader.rgb,h_rgb.data(),n_view*640*480*3*sizeof(uint8_t),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(synthRGBDLoader.depth,h_depth.data(),n_view*640*480*sizeof(float),cudaMemcpyHostToDevice));


	EigenCheckClass::getTransformationFromQuaternion(n_view,trans_quaternions,h_transformationMatrices,true);

	for(int v=0;v<n_view;v++)
	{
		printf("%d: ",v);
		for(int t=0;t<12;t++)
		{
			printf("%f ",h_transformationMatrices[v*12+t]);

		}
		printf("\n");
	}

	thrust::device_vector<float> d_transformationMatrices = h_transformationMatrices;
	synthRGBDLoader.transformationmatrices = thrust::raw_pointer_cast(d_transformationMatrices.data());

//	getTargetDataPointer(GroundTruthTransformations) =

	checkCudaErrors(cudaMemcpy(getTargetDataPointer(GroundTruthTransformations),synthRGBDLoader.transformationmatrices,n_view*12*sizeof(float),cudaMemcpyDeviceToDevice));


	device::loadSynthFrame<<<grid,block>>>(synthRGBDLoader);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if(output)
	{
		for(int i=0;i<n_view;i++)
		{
			float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
			checkCudaErrors(cudaMemcpy(h_f4_depth,synthRGBDLoader.xyzi+i*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

//			for(int p=0;p<640*480;p++)
//			{
//				float4 t = h_f4_depth[p];
//				h_f4_depth[p].x = h_transformationMatrices[i*12+0]*t.x + h_transformationMatrices[i*12+1]*t.y + h_transformationMatrices[i*12+2]*t.z + h_transformationMatrices[i*12+9];
//				h_f4_depth[p].y = h_transformationMatrices[i*12+3]*t.x + h_transformationMatrices[i*12+4]*t.y + h_transformationMatrices[i*12+5]*t.z + h_transformationMatrices[i*12+10];
//				h_f4_depth[p].z = h_transformationMatrices[i*12+6]*t.x + h_transformationMatrices[i*12+7]*t.y + h_transformationMatrices[i*12+8]*t.z + h_transformationMatrices[i*12+11];
//			}

			if(transform)
				sprintf(path,"/home/avo/pcds/synth_wc_points_transformed_%d.pcd",i);
			else
				sprintf(path,"/home/avo/pcds/synth_wc_points_%d.pcd",i);
			host::io::PCDIOController pcdIOCtrl;
			pcdIOCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
		}
	}

	printf("data loaded... \n");
}



SynthRGBDBenchmarkSource::~SynthRGBDBenchmarkSource()
{
	// TODO Auto-generated destructor stub
}


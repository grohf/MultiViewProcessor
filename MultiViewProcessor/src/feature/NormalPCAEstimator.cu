/*
 * NormalPCAEstimator.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#include "NormalPCAEstimator.h"
#include "utils.hpp"
#include <helper_cuda.h>
#include <helper_image.h>
#include "../sink/pcd_io.h"


namespace device
{
	struct NormalEstimator
	{

		float4 *input;
		float4 *output;

		enum
		{
			kxr = 9,
			kyr = kxr,

			kx = 32+kxr*2,
			ky = 24+kyr*2,

			kl = 2*kxr+1
		};


		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float4 shm[kx*ky];

			int sx,sy,off;

			/* ---------- LOAD SHM ----------- */
			const int oy = blockIdx.y*blockDim.y-kyr;
			const int ox = blockIdx.x*blockDim.x-kxr;


			for(off=threadIdx.y*blockDim.x+threadIdx.x;off<kx*ky;off+=blockDim.x*blockDim.y){
				sy = off/kx;
				sx = off - sy*kx;

				sy = oy + sy;
				sx = ox + sx;

				if(sx < 0) 		sx	=	0;
				if(sx > 639) 	sx 	= 639;
				if(sy < 0)		sy 	= 	0;
				if(sy > 479)	sy 	= 479;

				shm[off]=input[blockIdx.z*640*480+sy*640+sx];
			}
			__syncthreads();

			off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
			float3 mid = make_float3(shm[off].x,shm[off].y,shm[off].z);

//			if(blockIdx.x==10 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y==10)
//				printf("soso \n");



			if(mid.z==0)
			{
				  sx = threadIdx.x + blockIdx.x * blockDim.x;
				  sy = threadIdx.y + blockIdx.y * blockDim.y;
				  float4 f4 = make_float4(0,0,0,-2);
//				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
				  output[blockIdx.z*640*480+sy*640+sx] = f4;
				  return;
			}

			if(mid.z>4000)
			{
				  sx = threadIdx.x + blockIdx.x * blockDim.x;
				  sy = threadIdx.y + blockIdx.y * blockDim.y;
				  float4 f4 = make_float4(0,0,0,-3);
//				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
				  output[blockIdx.z*640*480+sy*640+sx] = f4;
				  return;
			}


			float3 mean = make_float3(0.0f,0.0f,0.0f);

			unsigned int count = 0;
			for(sy=0;sy<kl;sy++)
			{
				for(sx=0;sx<kl;sx++)
				{
					off = (threadIdx.y+sy)*kx+threadIdx.x+sx;

					if(sqrtf( (mid.x-shm[off].x)*(mid.x-shm[off].x)+(mid.y-shm[off].y)*(mid.y-shm[off].y)+(mid.z-shm[off].z)*(mid.z-shm[off].z) ) < 20)
					{
						mean.x += shm[off].x;
						mean.y += shm[off].y;
						mean.z += shm[off].z;
						count++;
					}

				}
			}

//				if(count < (kl*kl)/2)
//				{
//			      sx = threadIdx.x + blockIdx.x * blockDim.x;
//			      sy = threadIdx.y + blockIdx.y * blockDim.y;
//			      float4 f4 = make_float4(0,0,0,-7);
//			      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
//			      return;
//
//				}

			if(count==0)
			{
				  sx = threadIdx.x + blockIdx.x * blockDim.x;
				  sy = threadIdx.y + blockIdx.y * blockDim.y;
				  float4 f4 = make_float4(0,0,0,-5);
//				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
				  output[blockIdx.z*640*480+sy*640+sx] = f4;
				  return;
			}
			if(count==0) printf("AHHHHHHHHH \n");
			mean *= 1.0f/count;

			float cov[] = {0,0,0,0,0,0};

			for(sy=0;sy<kl;sy++)
			{
				for(sx=0;sx<kl;sx++)
				{

					off = (threadIdx.y+sy)*kx+threadIdx.x+sx;

					if(sqrtf( (mid.x-shm[off].x)*(mid.x-shm[off].x)+(mid.y-shm[off].y)*(mid.y-shm[off].y)+(mid.z-shm[off].z)*(mid.z-shm[off].z) ) < 20)
					{
						float3 v;
						v.x = shm[off].x;
						v.y = shm[off].y;
						v.z = shm[off].z;


						float3 d = v - mean;

						cov[0] += d.x * d.x;               //cov (0, 0
						cov[1] += d.x * d.y;               //cov (0, 1)
						cov[2] += d.x * d.z;               //cov (0, 2)
						cov[3] += d.y * d.y;               //cov (1, 1)
						cov[4] += d.y * d.z;               //cov (1, 2)
						cov[5] += d.z * d.z;               //cov (2, 2)
					}
				}
			}

//		bool br = false;
//		for(int i=0;i<5;i++)
//		{
//			if(cov[i]==0)
//				br = true;
//
//		}
//
//		if(br)
//		{
//			sx = threadIdx.x + blockIdx.x * blockDim.x;
//			sy = threadIdx.y + blockIdx.y * blockDim.y;
//			float4 f4 = make_float4(0,0,0,-7);
//
//			output[blockIdx.z*640*480+sy*640+sx] = f4;
//			return;
//		}

//		float mult = 0.001f;
//		cov[0] = cov[0] / 1000.0f;
//		cov[1] = cov[1] * mult;
//		cov[2] = cov[2] * mult;
//		cov[3] = cov[3] * mult;
//		cov[4] = cov[4] * mult;
//		cov[5] = cov[5] * mult;


//		cov[0]=268.049866;
//		cov[1]=-9.961166;
//		cov[2]=377.211151;
//		cov[3]=1330.253662;
//		cov[4]=783.158630;
//		cov[5]=1229.697510;

//		if(blockIdx.x==10 && blockIdx.y==2)
//			printf("problem! cov[0]:%f | cov[1]:%f | cov[2]:%f | cov[3]:%f | cov[4]:%f | cov[5]:%f \n",cov[0],cov[1],cov[2],cov[3],cov[4],cov[5]);

		sx = threadIdx.x + blockIdx.x * blockDim.x;
		sy = threadIdx.y + blockIdx.y * blockDim.y;

//		cov[0]=268.049866;
//		cov[1]=-9.961166;
//		cov[2]=377.211151;
//		cov[3]=1330.253662;
//		cov[4]=783.158630;
//		cov[5]=1229.697510;

		typedef Eigen33::Mat33 Mat33;
		Eigen33 eigen33(cov);

		Mat33 tmp;
		Mat33 vec_tmp;
		Mat33 evecs;
		float3 evals;


		eigen33.compute(tmp, vec_tmp, evecs, evals);


//		      float3 n = normalized (evecs[0]);

		  float3 n = evecs[0];


		  if(n.x==1 || n.x==-1)
		  {
//		    	  surf3Dwrite<float4>(make_float4(0,0,0,-4),surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
			  output[blockIdx.z*640*480+sy*640+sx] = make_float4(0,0,0,-4);
			  return;
		  }
		  if( dot(n,mid) >= 0.0f)
		  {
//		    	  n = make_float3(0,0,0);
//		    	  float4 f4 = make_float4(0,0,0,1);
//		    	  surf3Dwrite<float4>(f4,surf::surfRef,sx*sizeof(float4),sy,blockIdx.z*2);
//		    	  return;

			  n.x *= -1.0f;
			  n.y *= -1.0f;
			  n.z *= -1.0f;

		  }

		  float eig_sum = evals.x + evals.y + evals.z;
		  float curvature = (eig_sum == 0) ? 0 : fabsf( evals.x / eig_sum );
		  float4 f4 = make_float4(n.x,n.y,n.z,curvature);

//		      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
		  output[blockIdx.z*640*480+sy*640+sx] = f4;


//		  off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
//		  output[blockIdx.z*640*480+sy*640+sx] = shm[off];
		}

	};

	__global__ void estimateNormalKernel(const NormalEstimator ne) {ne (); }
}
device::NormalEstimator normalEstimator;

void
NormalPCAEstimator::init()
{
	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,n_view);


	normalEstimator.input = (float4 *)getInputDataPointer(WorldCoordinates);
	normalEstimator.output = (float4 *)getTargetDataPointer(Normals);

}

void
NormalPCAEstimator::execute()
{

	device::estimateNormalKernel<<<grid,block>>>(normalEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* Test */

	char path[50];
	for(int v=0;v<n_view;v++)
	{
		float4 *h_f4_normals = (float4 *)malloc(640*480*sizeof(float4));
		checkCudaErrors(cudaMemcpy(h_f4_normals,normalEstimator.output+v*640*480,640*480*sizeof(float4),cudaMemcpyDeviceToHost));

		uchar4 *h_uc4 = (uchar4 *)malloc(640*480*sizeof(uchar4));
		for(int i=0;i<640*480;i++)
		{
			if(h_f4_normals[i].w >= 0)
			{
				h_uc4[i].x = (h_f4_normals[i].x+1.0f)*127.f;
				h_uc4[i].y = (h_f4_normals[i].y+1.0f)*127.f;
				h_uc4[i].z = (h_f4_normals[i].z+1.0f)*127.f;
				h_uc4[i].w = 127.f;
			}
			else
			{
				h_uc4[i].x = 0;
				h_uc4[i].y = 0;
				h_uc4[i].z = 0;
				h_uc4[i].w = 127.f;
			}

		}

		sprintf(path,"/home/avo/pcds/normals_pca_%d.ppm",v);
		sdkSavePPM4ub(path,(unsigned char *)h_uc4,640,480);
	}

//	char path[50];
//	float4 *h_f4_normals = (float4 *)malloc(640*480*sizeof(float4));
//	checkCudaErrors(cudaMemcpy(h_f4_normals,normalEstimator.output,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
//
//	sprintf(path,"/home/avo/pcds/normals_pca%d.pcd",2);
//	host::io::PCDIOController ioCtrl;
//	ioCtrl.writeASCIIPCDNormals(path,(float *)h_f4_normals,640*480);
//
//	float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
//	checkCudaErrors(cudaMemcpy(h_f4_depth,normalEstimator.input,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
//
//	sprintf(path,"/home/avo/pcds/points_pca%d.pcd",2);
//	ioCtrl.writeASCIIPCD(path,(float *)h_f4_depth,640*480);
//

	printf("normals done \n");
}

NormalPCAEstimator::NormalPCAEstimator(unsigned int n_view_)
{
	n_view = n_view_;

	DeviceDataParams params;
	params.elements = 640*480*n_view;
	params.element_size = sizeof(float4);
	params.dataType = Point4D;
	params.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(params),Normals);
}

NormalPCAEstimator::~NormalPCAEstimator()
{

}


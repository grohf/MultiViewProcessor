/*
 * EigenCheckClass.cpp
 *
 *  Created on: Sep 23, 2012
 *      Author: avo
 */

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <cstring>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <libfreenect_sync.h>
#include <libfreenect.h>

#include "../sink/pcd_io.h"
#include "EigenCheckClass.h"

EigenCheckClass::EigenCheckClass()
{
	// TODO Auto-generated constructor stub

}

EigenCheckClass::~EigenCheckClass()
{
	// TODO Auto-generated destructor stub
}

void
EigenCheckClass::testSVD()
{
//	int n = 3;
//	Eigen::MatrixXf src = Eigen::MatrixXf::Random (3,n);
//	for(int y=0;y<5;y++)
//	{
//		for(int x=0;x<5;x++)
//		{
//			src.block<3,1>(0,y*3+x) = Vector3f(x,y,5+x);
//		}
//	}
//	cout << "Here is the matrix src :\n" << src << endl;

//	Eigen::Matrix<float, 3, 3> MR = Eigen::Matrix<float, 3, 3>::Zero(3,3);
//	MR(0,1)=-1;
//	MR(1,0)=1;
//	MR(2,2)=1;
//
//	std::cout << "Here is the matrix MR :\n" << MR << std::endl;
//
//	std::cout << " det: " << MR.determinant() << std::endl;

//	MR.determinant();
//
//	Vector3f mt(2,1,0);
//
//	MatrixXf target = MatrixXf::Zero (3,n);
//	for(int i=0;i<n;i++)
//	{
//		target.block<3,1>(0,i) = MR * src.block<3,1>(0,i) + mt;
//	}
////	cout << "Here is the matrix target :\n" << target << endl;
//
//	Vector3f cent_src(0,0,0);
//	Vector3f cent_target(0,0,0);
//	for(int i=0;i<n;i++)
//	{
//		cent_src += src.block<3,1>(0,i);
//		cent_target += target.block<3,1>(0,i);
//	}
//
//	cent_src /= n;
//	cent_target /= n;
//
////	cout << "Here is the vector cent_src :\n" << cent_src << endl;
//
//	for(int i=0;i<n;i++)
//	{
//		src.block<3,1>(0,i) -= cent_src;
//		target.block<3,1>(0,i) -= cent_target;
//	}
//
////	cout << "Here is the demeaned matrix src :\n" << src << endl;
////	cout << "Here is the demeaned matrix target :\n" << target << endl;
//
//	Eigen::Matrix<float, 3, 3> H = (src.cast<float>() * target.cast<float>().transpose());
////	cout << "Here is the matrix H :\n" << H << endl;
//
//	Eigen::Matrix<float, 3, 3> HP = H;
//	Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd (HP, Eigen::ComputeFullU | Eigen::ComputeFullV);
//	Eigen::Matrix<float, 3, 3> u = svd.matrixU ();
//	Eigen::Matrix<float, 3, 3> v = svd.matrixV ();
//
////	Eigen::Matrix<float, 3, 3> A = H;
//
////	Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3>> svd (A, Eigen::ComputeFullU | Eigen::ComputeFullV);
////	Eigen::Matrix<float, 3, 3> u = svd.matrixU ();
////	Eigen::Matrix<float, 3, 3> v = svd.matrixV ();
////
////	cout << "u:\n" << u << endl;
////	cout << "v:\n" << v << endl;
//
//	Eigen::Matrix<float, 3, 3> R = v*u.transpose();
//	cout << "Here is the matrix MR :\n" << MR << endl;
//	cout << "Here is the matrix R :\n" << R << endl;
//
//	Vector3f t = cent_target - R * cent_src;
//	cout << "Here is the vector mt :\n" << mt << endl;
//	cout << "Here is the vector t :\n" << t << endl;
//
//
////   MatrixXf A = MatrixXf::Zero (3,3);
//////   Vector3f p = Vector3f(1,2,3);
////   A.block<3,1>(0,0) = Vector3f(1,2,3);
////
////   cout << "Here is the matrix A:\n" << A << endl;
////   VectorXf b = VectorXf::Random(3);
////   cout << "Here is the right hand side b:\n" << b << endl;
////   cout << "The least-squares solution is:\n"
////        << A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b) << endl;
//
////   A.jacobiSvd(ComputeFullU | ComputeFullV);
//
////   Eigen::Matrix<float, 3, 3> HP = H;
////   Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd (HP, Eigen::ComputeFullU | Eigen::ComputeFullV);
////   Eigen::Matrix<float, 3, 3> u = svd.matrixU ();
////   Eigen::Matrix<float, 3, 3> v = svd.matrixV ();
////
////
////   cout << "u:\n" << u << endl;
////   cout << "v:\n" << v << endl;
}


void
EigenCheckClass::createTestOMRotation()
{
//	Eigen::Matrix<float, 3, 3> MR = Eigen::Matrix<float, 3, 3>::Zero(3,3);
//	MR(0,1) = -1;
//	MR(1,0) = 1;
//	MR(2,2) = 1;

//	std::cout << "Here is the matrix MR :\n" << MR << std::endl;
//	std::cout << " det: " << MR.determinant() << std::endl;

//	Eigen::Matrix<float,3,3> t = Eigen::AngleAxisf(0.4,Eigen::Vector3f(1,1,1));

	srand(time(NULL));
	Eigen::Vector3f angles = Eigen::Vector3f::Random(3) * (2 * M_PI);

	printf("angles: %f %f %f \n",angles(0),angles(1),angles(2));

	Eigen::Matrix3f MR2;
	MR2	= Eigen::AngleAxisf(angles(0),Eigen::Vector3f::UnitX())
		* Eigen::AngleAxisf(angles(1),Eigen::Vector3f::UnitY())
		* Eigen::AngleAxisf(angles(2),Eigen::Vector3f::UnitZ());

	std::cout << "RotationMatrix :\n" << MR2 << std::endl;
//	std::cout << " det: " << MR2.determinant() << std::endl;

	Eigen::Vector3f mt = Eigen::Vector3f::Random(3)*500.f;
	std::cout << "TranslationVector :\n" << mt << std::endl;


	int n = 16;
	Eigen::MatrixXf src = Eigen::MatrixXf::Random (3,n)*2000.f;
//	std::cout << "Here is the matrix src :\n" << src << std::endl;

	printf("\n");

	Eigen::MatrixXf target = Eigen::MatrixXf::Zero (3,n);
	for(int i=0;i<n;i++)
	{
		target.block<3,1>(0,i) = MR2 * src.block<3,1>(0,i) + mt;
	}

	pos_m = thrust::host_vector<float4>(n);
	pos_d = thrust::host_vector<float4>(n);
	float *data_m = src.data();
	float *data_d = target.data();
	for(size_t i = 0; i < pos_m.size(); i++)
	{
		pos_m[i].x = data_m[i*3+0];
		pos_m[i].y = data_m[i*3+1];
		pos_m[i].z = data_m[i*3+2];
		pos_m[i].w = 0;

		pos_d[i].x = data_d[i*3+0];
		pos_d[i].y = data_d[i*3+1];
		pos_d[i].z = data_d[i*3+2];
		pos_d[i].w = 0;
	}


//	std::cout << "Here is the matrix target :\n" << target << std::endl;

	Eigen::Vector3f cent_src(0,0,0);
	Eigen::Vector3f cent_target(0,0,0);
	for(int i=0;i<n;i++)
	{
		cent_src += src.block<3,1>(0,i);
		cent_target += target.block<3,1>(0,i);
	}

	cent_src /= n;
	cent_target /= n;

//	cout << "Here is the vector cent_src :\n" << cent_src << endl;

	for(int i=0;i<n;i++)
	{
		src.block<3,1>(0,i) -= cent_src;
		target.block<3,1>(0,i) -= cent_target;
	}

//	std::cout << "Here is the demeaned matrix src :\n" << src << std::endl;
//	std::cout << "Here is the demeaned matrix target :\n" << target << std::endl;

	Eigen::Matrix<float, 3, 3> HM = (src.cast<float>() * target.cast<float>().transpose());
	Eigen::Matrix3f HMT = HM.transpose();

	H = thrust::host_vector<float>(9);
	float *data_H = HMT.data();
	for(size_t i = 0; i < H.size(); i++)
	{

		H[i] = data_H[i];
	}

	std::cout << "Here is the CorrelationMatrix HM :\n" << HM << std::endl;
}

void
EigenCheckClass::createRotatedViews(thrust::host_vector<float4>& views, thrust::host_vector<float4>& views_synth, thrust::host_vector<float4>& normals, thrust::host_vector<float4>& normals_synth, thrust::host_vector<float>& transforamtionmatrix)
{
	srand(time(NULL));
		Eigen::Vector3f angles = Eigen::Vector3f::Random(3) * (2 * M_PI);

		printf("angles: %f %f %f \n",angles(0),angles(1),angles(2));

		Eigen::Matrix3f MR2;
		MR2	= Eigen::AngleAxisf(angles(0),Eigen::Vector3f::UnitX())
			* Eigen::AngleAxisf(angles(1),Eigen::Vector3f::UnitY())
			* Eigen::AngleAxisf(angles(2),Eigen::Vector3f::UnitZ());

		std::cout << "RotationMatrix :\n" << MR2 << std::endl;
	//	std::cout << " det: " << MR2.determinant() << std::endl;

		Eigen::Vector3f mt = Eigen::Vector3f::Random(3)*500.f;
//		std::cout << "TranslationVector :\n" << mt << std::endl;


		Eigen::MatrixXf src = Eigen::MatrixXf::Zero(3,views.size());
		Eigen::MatrixXf src_normals = Eigen::MatrixXf::Zero(3,views.size());

		float *data = src.data();
		float *data_normals = src_normals.data();
		for(int i=0;i<views.size();i++)
		{
			data[i*3+0] = views[i].x;
			data[i*3+1] = views[i].y;
			data[i*3+2] = views[i].z;

			data_normals[i*3+0] = normals[i].x;
			data_normals[i*3+1] = normals[i].y;
			data_normals[i*3+2] = normals[i].z;
		}

//		Eigen::Vector3f t = src.block<3,1>(0,50000);
//		std::cout << "Here is the t :\n" << t << std::endl;

		Eigen::MatrixXf target = Eigen::MatrixXf::Zero (3,views.size());
		Eigen::MatrixXf target_normals = Eigen::MatrixXf::Zero (3,views.size());
		for(int i=0;i<views.size();i++)
		{
			target.block<3,1>(0,i) = MR2 * src.block<3,1>(0,i) + mt;
			target_normals.block<3,1>(0,i) = MR2 * src_normals.block<3,1>(0,i);
		}

		data = target.data();
		data_normals = target_normals.data();
		for(int i=0;i<views.size();i++)
		{
			views_synth[i].x = data[i*3+0];
			views_synth[i].y = data[i*3+1];
			views_synth[i].z = data[i*3+2];
			views_synth[i].w = 0;

			normals_synth[i].x = data_normals[i*3+0];
			normals_synth[i].y = data_normals[i*3+1];
			normals_synth[i].z = data_normals[i*3+2];
			normals_synth[i].w = 0;
		}


		data = MR2.data();
		for(int i=0;i<9;i++)
		{
			transforamtionmatrix[i] = data[i];
		}

		Eigen::Vector3f mtr = -1 * ( MR2.transpose() * mt);
		std::cout << "ReTranslationVector :\n" << mtr << std::endl;

		data = mtr.data();
		for(int i=0;i<3;i++)
		{
			transforamtionmatrix[9+i] = data[i];
//			data[i] -= 5;
		}

		/* Test ReProjection */
		int count = 0;
		for(int i=0;i<views.size();i++)
		{
				float cxf = ((views[i].x*120.f)/(0.2084f*views[i].z))+320;
				float cyf = ((views[i].y*120.f)/(0.2084f*views[i].z))+240;

				int yi = i / 640;
				int xi = i - yi * 640;

				int cx = (int)cxf;
				int cy = (int)cyf;
				if(cx>=0 && cy>=0)
				{
					if((yi != cy || xi != cx))
						printf("(%d/%d) -> (%d/%d) <= (%f/%f) \n",xi,yi,cx,cy,cxf,cyf);

					count++;
				}
		}
		printf("count: %d \n",count);

//	 Test
//		char path[50];
//		host::io::PCDIOController pcdIOCtrl;
//
//		sprintf(path,"/home/avo/pcds/src_real_points_%d.pcd",0);
//		pcdIOCtrl.writeASCIIPCD(path,(float *)views.data(),640*480);
//
//
//
//
//		sprintf(path,"/home/avo/pcds/src_synth_points_%d.pcd",0);
//		pcdIOCtrl.writeASCIIPCD(path,(float *)views_synth.data(),640*480);
//
//		sprintf(path,"/home/avo/pcds/src_synth_normals_%d.pcd",0);
//		pcdIOCtrl.writeASCIIPCDNormals(path,(float *)normals_synth.data(),640*480);
//
//		//Test rerotate!
//
//
//
//		Eigen::MatrixXf rerotated = Eigen::MatrixXf::Zero (3,views.size());
//		for(int i=0;i<views.size();i++)
//		{
//			rerotated.block<3,1>(0,i) = MR2.transpose() * target.block<3,1>(0,i) + mtr;
//		}
//
//		thrust::host_vector<float4> synth_retransformed(640*480);
//
//		data = rerotated.data();
//		for(int i=0;i<synth_retransformed.size();i++)
//		{
//			synth_retransformed[i].x = data[i*3+0];
//			synth_retransformed[i].y = data[i*3+1];
//			synth_retransformed[i].z = data[i*3+2];
//		}
//
//		sprintf(path,"/home/avo/pcds/src_retransformed_points_%d.pcd",0);
//		pcdIOCtrl.writeASCIIPCD(path,(float *)synth_retransformed.data(),640*480);


//		freenect_sync_stop();
}

const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* const DELIMITER = " ";

void
EigenCheckClass::createRefrenceTransformationMatrices(char *dirPath, unsigned int n_views, unsigned int *lns_depth, char **path_depth,  char **path_rgb, float *tansformationMatrices)
{

	std::ifstream fin_gt,fin_depth,fin_rgb;
	fin_gt.open("/home/avo/Desktop/rgbd_dataset_freiburg3_teddy/groundtruth.txt");
	fin_depth.open("/home/avo/Desktop/rgbd_dataset_freiburg3_teddy/depth.txt");
	fin_rgb.open("/home/avo/Desktop/rgbd_dataset_freiburg3_teddy/rgb.txt");
	if (!fin_gt.good() || !fin_depth.good() || !fin_rgb.good())
	{
		printf("file not found \n");
		return;
	}

	//get min_sensor_ts
	double min_sensor_ts;

	char *oldSensorLineBuffer;
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
			oldSensorLineBuffer = buf;
			cond = false;

//			for (int n = 1; n < MAX_TOKENS_PER_LINE; n++)
//			{
//				token[n] = strtok(0, DELIMITER);
//				if (!token[n]) break;
//
//				double t = atof(token[n]);
//				printf("%f \n",t);
//			}
		}
	}

	cond = true;
	int l = 0;

	unsigned int linecount = 0;
//	char *depth_png;
//	double depth_ts;
//	path_depth = (char **)malloc(n_views*sizeof(char *));

	char depth_path_tmp[n_views][50];
	double depth_ts[n_views];

	while (!fin_depth.eof() && cond)
	{
		char buf[MAX_CHARS_PER_LINE];
		fin_depth.getline(buf,MAX_CHARS_PER_LINE);

		const char* token[MAX_TOKENS_PER_LINE] = {0};
		token[0] = strtok(buf, DELIMITER);
//		printf("%s! > %d \n",token[0],strcmp(token[0],"#"));

		if(strcmp(token[0],"#") && (atof(token[0]) > min_sensor_ts))
		{
			if(linecount==lns_depth[l])
			{
				depth_ts[l] = atof(token[0]);
				char tmp_char[50];
//				depth_path_tmp[l] = strtok(0, DELIMITER);

				strcpy(depth_path_tmp[l],strtok(0, DELIMITER));

				printf("%d: %s -> %f \n",l,depth_path_tmp[l],depth_ts[l]);

				if(++l==n_views)
					cond = false;
			}

			linecount++;
		}
	}


//	path_depth = (char **)&depth_path_tmp;

	for(int i=0;i<n_views;i++)
	{
		printf("%d: %s -> %f \n",i,depth_path_tmp[i],depth_ts[i]);
	}

//	printf("%f > %s \n",depth_ts,depth_png);



//	int c = 0;
//	while (!fin_gt.eof() && (c++ < 20) )
//	{
//		char buf[512];
//		fin_gt.getline(buf,512);
//		int n = 0;
//
//		const char* token[MAX_TOKENS_PER_LINE] = {0};
//
//		token[0] = strtok(buf, DELIMITER);
//		if (token[0])
//		{
//		  for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
//		  {
//			token[n] = strtok(0, DELIMITER);
//			if (!token[n]) break;
//		  }
//		}
//
//		// process (print) the tokens
//		for (int i = 0; i < n; i++) // n = #of tokens
//		  std::cout << "Token[" << i << "] = " << token[i] << std::endl;
//		std::cout << std::endl;
//	}


	/*
	Eigen::Quaternion<float> q1(0.7889,-0.5183,0.1970,-0.2651);
	Eigen::Quaternion<float> q2 = q1.inverse();

	printf("quads: %f %f %f %f \n",q1.x(),q1.y(),q1.z(),q1.w());
	printf("quads2: %f %f %f %f \n",q2.x(),q2.y(),q2.z(),q2.w());

	Eigen::Quaternion<float> q3 = q2*q1;
	 */


//	std::cout << "q1 :\n" << q1.matrix() << std::endl;
//	std::cout << "q1.inv :\n" << q1.matrix().inverse() << std::endl;
//	std::cout << "q2 :\n" << q2.matrix() << std::endl;
//	std::cout << "q3 :\n" << q3.matrix() << std::endl;
//
//
////	std::cout << "Here is the quad :\n" << q.toRotationMatrix() << std::endl;
//	std::cout << "Here is the q3.det :\n" << q3.matrix().determinant() << std::endl;


}

void
EigenCheckClass::getTransformationFromQuaternion(unsigned int n_view,thrust::host_vector<float>& in_quaternions,thrust::host_vector<float>& tranformmatrices,bool anchorFist)
{
	Eigen::Matrix3f reRota;
	Eigen::Vector3f reTrans;
//	Eigen::Matrix4f reTransform = Eigen::Matrix4f::Identity();
	for(int v=0;v<n_view;v++)
	{
		printf("%d: ",v);
		for(int tq=0;tq<7;tq++)
		{
			printf("%f |",in_quaternions[v*7+tq]);
		}
		printf("\n");


		Eigen::Quaternion<float> q(in_quaternions[v*7+6],in_quaternions[v*7+3],in_quaternions[v*7+4],in_quaternions[v*7+5]);
//		std::cout << "q1 :\n" << q1.matrix() << std::endl;
//		std::cout << "q1_inv :\n" << q1.inverse().matrix() << std::endl;

		Eigen::Matrix3f rota = q.matrix();
		Eigen::Vector3f v_out;
		v_out << in_quaternions[v*7+0],in_quaternions[v*7+1],in_quaternions[v*7+2];
		if(anchorFist)
		{
			if(v==0)
			{
				reRota = rota.inverse();
				reTrans = -(reRota * v_out);
			}

			rota = reRota * rota;
			v_out = reRota * v_out + reTrans;

		}

		Eigen::Matrix3f r_out = rota.inverse();
		std::cout << "r_out :\n" << r_out << std::endl;
		std::cout << "v_out :\n" << v_out << std::endl;

		for(int t=0;t<9;t++)
		{
			tranformmatrices[v*12+t] = (r_out.data())[t];
		}
		tranformmatrices[v*12+9] 	= v_out[0];
		tranformmatrices[v*12+10] 	= v_out[1];
		tranformmatrices[v*12+11] 	= v_out[2];
	}

}

void
EigenCheckClass::checkCorrelationQuality(thrust::host_vector<float4> points_view1,thrust::host_vector<float4> points_view2,thrust::host_vector<float> transformation,float threshhold)
{
	Eigen::Matrix3f rota1;
	Eigen::Matrix3f rota2;
	Eigen::Vector3f trans1;
	Eigen::Vector3f trans2;

	for(int j=0;j<3;j++)
	{
		trans1(j) = transformation[9+j]*1000.f;
		trans2(j) = transformation[12+9+j]*1000.f;
		for(int i=0;i<3;i++)
		{
			rota1(j,i) = transformation[j*3+i];
			rota2(j,i) = transformation[12+j*3+i];
		}
	}


	std::cout << "rota1 :\n" << rota1 << std::endl;
	std::cout << "trans1 :\n" << trans1 << std::endl;


	std::cout << "rota2 :\n" << rota2 << std::endl;
	std::cout << "trans2 :\n" << trans2 << std::endl;

	int count = 0;
	for(int i=0;i<points_view1.size();i++)
	{
		float4 p1 = points_view1[i];
		Eigen::Vector3f v1(p1.x,p1.y,p1.z);

		float4 p2 = points_view2[i];
		Eigen::Vector3f v2(p2.x,p2.y,p2.z);

		Eigen::Vector3f diff = ((rota1*v1+trans1) - (rota2*v2+trans2));

		if(diff.norm() <= threshhold)
			count++;

		printf("%f \n",diff.norm());
	}

	printf("count: %d",count);

//	float4 p1 = points_view1[0];
//	Eigen::Vector3f v1(p1.x,p1.y,p1.z);
//
//	printf("%f %f %f | ",p1.x,p1.y,p1.z);
//	std::cout << "v1 :\n" << v1 << std::endl;
}

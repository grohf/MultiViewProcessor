/*
 * EigenCheckClass.cpp
 *
 *  Created on: Sep 23, 2012
 *      Author: avo
 */

#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
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

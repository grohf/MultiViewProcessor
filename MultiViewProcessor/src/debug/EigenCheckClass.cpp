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

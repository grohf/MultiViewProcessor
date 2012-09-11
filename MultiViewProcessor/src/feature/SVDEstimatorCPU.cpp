///*
// * SVDEstimatorCPU.cpp
// *
// *  Created on: Sep 6, 2012
// *      Author: avo
// */
//
//
//#include <iostream>
//
//#include <Eigen/Dense>
//
//
//
//using namespace std;
//using namespace Eigen;
//
//#include "SVDEstimatorCPU.h"
//
//
//
//SVDEstimator_CPU::SVDEstimator_CPU()
//{
//	// TODO Auto-generated constructor stub
//
//}
//
//SVDEstimator_CPU::~SVDEstimator_CPU()
//{
//	// TODO Auto-generated destructor stub
//}
//
//void SVDEstimator_CPU::init()
//{
//
//}
//
//
//void SVDEstimator_CPU::execute()
//{
//	int n = 3;
//	MatrixXf src = MatrixXf::Random (3,n);
////	for(int y=0;y<5;y++)
////	{
////		for(int x=0;x<5;x++)
////		{
////			src.block<3,1>(0,y*3+x) = Vector3f(x,y,5+x);
////		}
////	}
////	cout << "Here is the matrix src :\n" << src << endl;
//
//	Eigen::Matrix<float, 3, 3> MR = Eigen::Matrix<float, 3, 3>::Zero(3,3);
//	MR(0,1)=-1;
//	MR(1,0)=1;
//	MR(2,2)=1;
////	cout << "Here is the matrix MR :\n" << MR << endl;
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
//}

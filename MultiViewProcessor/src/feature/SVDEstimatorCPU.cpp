/*
 * SVDEstimatorCPU.cpp
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */


#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#include "SVDEstimatorCPU.h"



SVDEstimator_CPU::SVDEstimator_CPU()
{
	// TODO Auto-generated constructor stub

}

SVDEstimator_CPU::~SVDEstimator_CPU()
{
	// TODO Auto-generated destructor stub
}

void SVDEstimator_CPU::init()
{

}


void SVDEstimator_CPU::execute()
{
   MatrixXf A = MatrixXf::Random(3, 3);
   cout << "Here is the matrix A:\n" << A << endl;
//   VectorXf b = VectorXf::Random(3);
//   cout << "Here is the right hand side b:\n" << b << endl;
//   cout << "The least-squares solution is:\n"
//        << A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b) << endl;

   A.jacobiSvd(ComputeFullU | ComputeFullV);

   Eigen::Matrix<float, 3, 3> H = A;
   Eigen::JacobiSVD<Eigen::Matrix<float, 3, 3> > svd (H, Eigen::ComputeFullU | Eigen::ComputeFullV);
   Eigen::Matrix<float, 3, 3> u = svd.matrixU ();
   Eigen::Matrix<float, 3, 3> v = svd.matrixV ();


   cout << "u:\n" << u << endl;
   cout << "v:\n" << v << endl;
}

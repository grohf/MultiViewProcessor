/*
 * EigenCheckClass.h
 *
 *  Created on: Sep 23, 2012
 *      Author: avo
 */

#ifndef EIGENCHECKCLASS_H_
#define EIGENCHECKCLASS_H_

#include <thrust/host_vector.h>

class EigenCheckClass {

public:
	thrust::host_vector<float4> pos_m;
	thrust::host_vector<float4> pos_d;

	thrust::host_vector<float> H;
	thrust::host_vector<float> T;

public:
	EigenCheckClass();
	virtual ~EigenCheckClass();

	void testSVD();
	void createTestOMRotation();
};

#endif /* EIGENCHECKCLASS_H_ */

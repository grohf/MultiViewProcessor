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

	void createRotatedViews(thrust::host_vector<float4>& views, thrust::host_vector<float4>& views_synth, thrust::host_vector<float4>& normals, thrust::host_vector<float4>& normals_synth, thrust::host_vector<float>& transforamtionmatrix);

};

#endif /* EIGENCHECKCLASS_H_ */

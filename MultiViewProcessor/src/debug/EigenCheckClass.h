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

	void createRefrenceTransformationMatrices(char *dirPath, unsigned int n_views, unsigned int *lns_depth, char **path_depth, char **path_rgb, float *tansformationMatrices);

	static void getTransformationFromQuaternion(unsigned int n_view,thrust::host_vector<float>& in_qaternions,thrust::host_vector<float>& tranformmatrices,bool anchorFist=true);

	static void checkCorrelationQuality(thrust::host_vector<float4> points_view1,thrust::host_vector<float4> points_view2,thrust::host_vector<float> transforamtion,float threshhold);
};

#endif /* EIGENCHECKCLASS_H_ */

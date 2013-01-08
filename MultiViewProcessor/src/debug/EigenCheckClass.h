/*
 * EigenCheckClass.h
 *
 *  Created on: Sep 23, 2012
 *      Author: avo
 */

#ifndef EIGENCHECKCLASS_H_
#define EIGENCHECKCLASS_H_

#include <thrust/host_vector.h>
#include <stdio.h>

class EigenCheckClass {

public:
//	thrust::host_vector<float4> pos_m;
//	thrust::host_vector<float4> pos_d;
//
//	thrust::host_vector<float> H;
//	thrust::host_vector<float> T;
	static thrust::host_vector<float> GT_transformation;

	static char line[100];

	static unsigned int sframe;
	static unsigned int tframe;
	static unsigned int binsPerFeature;
	static float singleCoreespQ;
	static float setCorrespQ;
	static float transformationQuality;
	static float minTransformError;
	static float angle;
	static unsigned int scene;


public:
	EigenCheckClass();
	virtual ~EigenCheckClass();

	void testSVD();
	void createTestOMRotation();

	void createRotatedViews(thrust::host_vector<float4>& views, thrust::host_vector<float4>& views_synth, thrust::host_vector<float4>& normals, thrust::host_vector<float4>& normals_synth, thrust::host_vector<float>& transforamtionmatrix);

	void createRefrenceTransformationMatrices(char *dirPath, unsigned int n_views, unsigned int *lns_depth, char **path_depth, char **path_rgb, float *tansformationMatrices);

	static void getTransformationFromQuaternion(unsigned int n_view,thrust::host_vector<float>& in_qaternions,thrust::host_vector<float>& tranformmatrices,bool anchorFist=true);

	static void checkCorrelationQuality(thrust::host_vector<float4> points_view1,thrust::host_vector<float4> points_view2,thrust::host_vector<float> transforamtion,float threshhold);

	static void checkCorrespondancesQualityGT(thrust::host_vector<float4> pos,thrust::host_vector<unsigned int> src_idx,unsigned int s_length,thrust::host_vector<unsigned int> target_idx,unsigned int corresp_list_length,float threshold);

	static void checkCorrespondancesSetQualityGT(thrust::host_vector<float4> pos,thrust::host_vector<unsigned int> csetIdx, thrust::host_vector<unsigned int> cset, unsigned int sets , unsigned int setLength, unsigned int offset, unsigned int modus=0);
	static void checkCorrespondancesSetCombinationQualityGT(thrust::host_vector<float4> pos,thrust::host_vector<unsigned int> csetIdx, thrust::host_vector<unsigned int> cset, unsigned int sets, unsigned int n_set_combi , unsigned int setLength, unsigned int offset, unsigned int modus=0);
	static void checkTransformationQuality(thrust::host_vector<float> eTransformation);

	static void checkGlobalFineRegistration(unsigned int n_view,thrust::host_vector<float4> pos);

	static void setGroundTruthTransformation(thrust::host_vector<float> tm);

	static void flushLine(char *fname);
	static void setFrameValues(unsigned int sframe,unsigned int tframe);
	static void setBinsPerFeatureCount(unsigned int binCount);
	static void setMinTransformError(float valid);
	static void setScene(unsigned int s);
};

#endif /* EIGENCHECKCLASS_H_ */

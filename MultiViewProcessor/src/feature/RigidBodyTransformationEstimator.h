/*
 * RigidBodyTransformationEstimator.h
 *
 *  Created on: Sep 6, 2012
 *      Author: avo
 */

#ifndef RIGIDBODYTRANSFORMATIONESTIMATOR_H_
#define RIGIDBODYTRANSFORMATIONESTIMATOR_H_

#include <curand.h>
#include "feature.hpp"

class RigidBodyTransformationEstimator : public Feature {

	enum Input
	{
		IdxList,
		HistogramMap,
		InfoList,
		Coordiantes,
	};

	enum Target
	{
		ProbList,
		ProbIdxList,
		RndIndices,
		RndSrcIndices,
		RndTargetIndices,
		SIndices,
		CorrelationMatrices,
	};

	dim3 grid;
	dim3 block;

	unsigned int s;
	unsigned int k;
	unsigned int rn;

	dim3 deMeanGrid;
	dim3 deMeanBlock;

	curandGenerator_t gen ;
	float *d_rnd_data;

public:
	RigidBodyTransformationEstimator(unsigned int rn_, unsigned int s_,unsigned int k_)
	{
		s = s_;
		k = k_;
		rn = rn_;

		DeviceDataParams probListParams;
		probListParams.elements = s*k;
		probListParams.element_size = sizeof(float);
		probListParams.elementType = FLOAT1;
		probListParams.dataType = Point;
		addTargetData(addDeviceDataRequest(probListParams),ProbList);

		DeviceDataParams probIdxListParams;
		probIdxListParams.elements = s*k;
		probIdxListParams.element_size = sizeof(unsigned int);
		probIdxListParams.elementType = UINT1;
		probIdxListParams.dataType = Indice;
		addTargetData(addDeviceDataRequest(probIdxListParams),ProbIdxList);

		DeviceDataParams rndParams;
		rndParams.elements = s;
		rndParams.element_size = sizeof(float);
		rndParams.elementType = FLOAT1;
		rndParams.dataType = Indice;
		addTargetData(addDeviceDataRequest(rndParams),RndIndices);

		DeviceDataParams rndParams2;
		rndParams2.elements = rn*s*k;
		rndParams2.element_size = sizeof(float);
		rndParams2.elementType = FLOAT1;
		rndParams2.dataType = Indice;
		addTargetData(addDeviceDataRequest(rndParams2),RndSrcIndices);
		addTargetData(addDeviceDataRequest(rndParams2),RndTargetIndices);

		DeviceDataParams sIdxParams;
		sIdxParams.elements = s;
		sIdxParams.element_size = sizeof(unsigned int);
		sIdxParams.elementType = UINT1;
		sIdxParams.dataType = Indice;
		addTargetData(addDeviceDataRequest(sIdxParams),SIndices);

		DeviceDataParams correlationmatrixesParams;
		correlationmatrixesParams.elements = rn;
		correlationmatrixesParams.element_size = 9 * sizeof(float);
		correlationmatrixesParams.elementType = FLOAT1;
		correlationmatrixesParams.dataType = Matrix;
		addTargetData(addDeviceDataRequest(correlationmatrixesParams),CorrelationMatrices);



	}
	virtual ~RigidBodyTransformationEstimator() { }

	void init();
	void execute();

	void setPersistanceIndexList(DeviceDataInfoPtr pidxList)
	{
		addInputData(pidxList,IdxList);
	}

	void setPersistanceHistogramMap(DeviceDataInfoPtr phMap)
	{
		addInputData(phMap,HistogramMap);
	}

	void setPersistenceInfoList(DeviceDataInfoPtr pInfoList)
	{
		addInputData(pInfoList,InfoList);
	}

	void setCoordinatesMap(DeviceDataInfoPtr pCoords)
	{
		addInputData(pCoords,Coordiantes);
	}


};

#endif /* RIGIDBODYTRANSFORMATIONESTIMATOR_H_ */

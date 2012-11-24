/*
 * RigidBodyTransformationAdvancedEstimatior.h
 *
 *  Created on: Nov 20, 2012
 *      Author: avo
 */

#ifndef RIGIDBODYTRANSFORMATIONADVANCEDESTIMATIOR_H_
#define RIGIDBODYTRANSFORMATIONADVANCEDESTIMATIOR_H_

#include "feature.hpp"

class RigidBodyTransformationAdvancedEstimatior : public Feature {

	enum Input
	{
		IdxList,
		HistogramMap,
		InfoList,
		Coordiantes,
	};

	enum Target
	{
		TransformationMatrices,
		TransformationMetaDataList,
	};

	unsigned int s;
	unsigned int k;
	unsigned int rn;

	unsigned int n_view;

public:
	RigidBodyTransformationAdvancedEstimatior(unsigned int n_view, unsigned int rn, unsigned int s,unsigned int k);
	virtual ~RigidBodyTransformationAdvancedEstimatior();

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

	DeviceDataInfoPtr getTransformationMatrices()
	{
		return getTargetData(TransformationMatrices);
	}

	DeviceDataInfoPtr getTransformationInfoList()
	{
		return getTargetData(TransformationMetaDataList);
	}

	void TestRBTAFct();

};

#endif /* RIGIDBODYTRANSFORMATIONADVANCEDESTIMATIOR_H_ */

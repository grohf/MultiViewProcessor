/*
 * TranformationValidator.h
 *
 *  Created on: Sep 18, 2012
 *      Author: avo
 */

#ifndef TRANFORMATIONVALIDATOR_H_
#define TRANFORMATIONVALIDATOR_H_

#include "feature.hpp"


class TranformationValidator : public Feature {

	enum Input
	{
		WorldCoords,
		Normals,
		SensorInfoList,
		TransforamtionMatrices,
		TransformationInfoList,
	};

	enum Target
	{
		TransformationMatricesToWorldCorrdinates,
		ErrorList,
		ErrorListIndices,
	};

	dim3 grid;
	dim3 block;

	unsigned int n_views;
	unsigned int n_rsac;

public:
	TranformationValidator(unsigned int n_views,unsigned int n_rsac);
//	: n_views(n_views), n_rsac(n_rsac)
//	{
//		DeviceDataParams errorListParams;
//		errorListParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
//		errorListParams.element_size = sizeof(float);
//		errorListParams.elementType = FLOAT1;
//		errorListParams.dataType = Point;
//		addTargetData(addDeviceDataRequest(errorListParams),ErrorList);
//
//		DeviceDataParams errorListIdxParams;
//		errorListIdxParams.elements = ((n_views-1)*n_views)/2 * n_rsac;
//		errorListIdxParams.element_size = sizeof(unsigned int);
//		errorListIdxParams.elementType = UINT1;
//		errorListIdxParams.dataType = Indice;
//		addTargetData(addDeviceDataRequest(errorListIdxParams),ErrorListIndices);
//
//	}

	virtual ~TranformationValidator() { }

	void init();
	void execute();


	void setWorldCooordinates(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,WorldCoords);
	}

	void setNormals(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,Normals);
	}

	void setSensorInfoList(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,SensorInfoList);
	}

	void setTransformationmatrices(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,TransforamtionMatrices);
	}

	void setTranformationInfoList(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,TransformationInfoList);
	}

};

#endif /* TRANFORMATIONVALIDATOR_H_ */

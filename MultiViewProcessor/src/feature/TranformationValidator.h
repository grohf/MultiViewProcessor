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
		TransforamtionMatrices,
		TransformationInfoList,
	};

	dim3 grid;
	dim3 block;

public:
	TranformationValidator() { }
	virtual ~TranformationValidator() { }

	void init();
	void execute();


	void setWorldCooordinates(DeviceDataInfoPtr ddi)
	{
		addInputData(ddi,WorldCoords);
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

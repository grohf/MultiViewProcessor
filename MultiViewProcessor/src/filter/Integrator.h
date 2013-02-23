/*
 * Integrator.h
 *
 *  Created on: Feb 23, 2013
 *      Author: avo
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include "filter.hpp"

class Integrator : public Filter {

	enum Input
	{
		Positions,
		MinTransformations,
	};

	enum Target
	{
		TransformPositions,
	};

	unsigned int n_view;

public:


	Integrator();
	virtual ~Integrator();

	void init();
	void execute();

	void setPosition(DeviceDataInfoPtr ddiPtr)
	{
		addFilterInput(ddiPtr,Positions);
	}

	void setMinTransformations(DeviceDataInfoPtr ddiPtr)
	{
		addInputData(ddiPtr,MinTransformations);
	}

//	DeviceDataInfoPtr getTransformedPositions()
//	{
//		return getTargetData(TransformPositions);
//	}

};

#endif /* INTEGRATOR_H_ */

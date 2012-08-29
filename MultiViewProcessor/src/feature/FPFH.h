/*
 * FPFH.h
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#ifndef FPFH_H_
#define FPFH_H_

#include "feature.hpp"

class FPFH: public Feature {

	float radius;

	enum Input
	{
		PointCoordinates,
		PointNormal,
		IndicesBlock
	};

	enum Target
	{
		SPFHistogram1,
		SPFHistogram2,

		FPFHistogram1,
		FPFHistogram2,

		MeanHistogram1,
		MeanHistogram2,

		DivHistogram1,

		PFPFHistogram,
		PFPFHIndices,
	};

	dim3 block;
	dim3 grid;

public:
	FPFH() {

		DeviceDataParams params;
		params.elements = 640*480;
		params.element_size = 8 * sizeof(float);
		params.elementType = FLOAT1;
		params.dataType = Histogramm;

		addTargetData(addDeviceDataRequest(params),SPFHistogram1);
		addTargetData(addDeviceDataRequest(params),SPFHistogram2);

		addTargetData(addDeviceDataRequest(params),FPFHistogram1);
		addTargetData(addDeviceDataRequest(params),FPFHistogram2);

		DeviceDataParams meanHistoparams;
		meanHistoparams.elements = 8;
		meanHistoparams.element_size = sizeof(float);
		meanHistoparams.elementType = FLOAT1;
		meanHistoparams.dataType = Histogramm;

		addTargetData(addDeviceDataRequest(meanHistoparams),MeanHistogram1);
		addTargetData(addDeviceDataRequest(meanHistoparams),MeanHistogram2);

		DeviceDataParams divHistogramParams;
		divHistogramParams.elements = 640*480;
		divHistogramParams.element_size = sizeof(float);
		divHistogramParams.elementType = FLOAT1;
		divHistogramParams.dataType = Histogramm;

		addTargetData(addDeviceDataRequest(divHistogramParams),DivHistogram1);


	}
	~FPFH(){

	}

	void execute();
	void init();

//	void setRadius(float r)
//	{
//		radius = r;
//	}

	void setIndices(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,IndicesBlock);
	}

	void setNormals(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,PointNormal);
	}

	void setPointCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,PointCoordinates);
	}

	DeviceDataInfoPtr getFPFH()
	{
		return getTargetData(PFPFHistogram);
	}
};

#endif /* FPFH_H_ */


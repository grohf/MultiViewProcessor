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
		DivHistogram2,

		Sigma1,
		Sigma2,


		TestMap,

//		PFPFHistogram,
		PFPFHIndices,
		PFPFHInfoList,
	};

	dim3 block;
	dim3 grid;

	unsigned int *d_infoList;

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
		addTargetData(addDeviceDataRequest(divHistogramParams),DivHistogram2);

		DeviceDataParams sigmaHistoparams;
		sigmaHistoparams.elements = 1;
		sigmaHistoparams.element_size = sizeof(float);
		sigmaHistoparams.elementType = FLOAT1;
		sigmaHistoparams.dataType = Sigma;
		addTargetData(addDeviceDataRequest(sigmaHistoparams),Sigma1);
		addTargetData(addDeviceDataRequest(sigmaHistoparams),Sigma2);


		DeviceDataParams PersitanceMapParams;
		PersitanceMapParams.elements = 640*480;
		PersitanceMapParams.element_size = sizeof(unsigned int);
		PersitanceMapParams.elementType = UINT1;
		PersitanceMapParams.dataType = Indice;
		addTargetData(addDeviceDataRequest(PersitanceMapParams),PFPFHIndices);


		DeviceDataParams TestMapParams;
		TestMapParams.elements = 640*480;
		TestMapParams.element_size = sizeof(uchar4);
		TestMapParams.elementType = UCHAR4;
		TestMapParams.dataType = Point4D;
		addTargetData(addDeviceDataRequest(TestMapParams),TestMap);


//		DeviceDataParams PFPFHparams;
//		PFPFHparams.elements = 640*480*8;
//		PFPFHparams.element_size = 8 * sizeof(float);
//		PFPFHparams.elementType = FLOAT1;
//		PFPFHparams.dataType = Histogramm;
//		addTargetData(addDeviceDataRequest(PFPFHparams),PFPFHistogram);

		DeviceDataParams PersistanceInfoListParams;
		PersistanceInfoListParams.elements = 1+8;
		PersistanceInfoListParams.element_size = sizeof(unsigned int);
		PersistanceInfoListParams.elementType = UINT1;
		PersistanceInfoListParams.dataType = ListItem;
		addTargetData(addDeviceDataRequest(PersistanceInfoListParams),PFPFHInfoList);

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
		return getTargetData(FPFHistogram2);
	}

	DeviceDataInfoPtr getPersistanceIndexList()
	{
		return getTargetData(PFPFHIndices);
	}
};

#endif /* FPFH_H_ */


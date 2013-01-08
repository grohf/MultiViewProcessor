/*
 * DFPFHEstimator.h
 *
 *  Created on: Sep 25, 2012
 *      Author: avo
 */

#ifndef DFPFHESTIMATOR_H_
#define DFPFHESTIMATOR_H_

#include "feature.hpp"

class DFPFHEstimator : public Feature {

	enum Input
	{
		PointCoordinates,
		PointNormal,
//		IndicesBlock,
	};

	enum Target
	{

		DFPFHistogram,

		PDFPFHIdxList,
		PDFPFHIdxLength,
	};

	unsigned int n_view;
	unsigned int outputlevel;
public:
	DFPFHEstimator(unsigned int n_view,unsigned int outputlevel=0);
	virtual ~DFPFHEstimator();

	void init();
	void execute();

	void TestDFPFHE();

//	void setIndices(DeviceDataInfoPtr ddip)
//	{
//		addInputData(ddip,IndicesBlock);
//	}

	void setNormals(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,PointNormal);
	}

	void setPointCoordinates(DeviceDataInfoPtr ddip)
	{
		addInputData(ddip,PointCoordinates);
	}

	DeviceDataInfoPtr getDFPFH()
	{
		return getTargetData(DFPFHistogram);
	}

	DeviceDataInfoPtr getPersistanceIndexList()
	{
		return getTargetData(PDFPFHIdxList);
	}

	DeviceDataInfoPtr getPersistenceInfoList()
	{
		return getTargetData(PDFPFHIdxLength);
	}

};


#endif /* DFPFHESTIMATOR_H_ */

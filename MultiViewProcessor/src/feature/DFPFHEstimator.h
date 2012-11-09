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
		SDPFHistogram1,
		SDPFHistogram2,

		DFPFHistogram1,
		DFPFHistogram2,

		MeanHistogram1,
		MeanHistogram2,

		DivHistogram1,
		DivHistogram2,

		Sigma1,
		Sigma2,

		PDFPFHIndices,
		PDFPFHInfoList,
	};

	unsigned int n_view;

public:
	DFPFHEstimator(unsigned int n_view);
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
		return getTargetData(DFPFHistogram2);
	}

	DeviceDataInfoPtr getPersistanceIndexList()
	{
		return getTargetData(PDFPFHIndices);
	}

	DeviceDataInfoPtr getPersistenceInfoList()
	{
		return getTargetData(PDFPFHInfoList);
	}

};


#endif /* DFPFHESTIMATOR_H_ */

/*
 * SyncFreenectSource.h
 *
 *  Created on: Aug 23, 2012
 *      Author: avo
 */

#ifndef SYNCFREENECTSOURCE_H_
#define SYNCFREENECTSOURCE_H_

#include <thrust/host_vector.h>

#include <source.h>
#include <vector_types.h>
#include <stdint.h>

class SyncFreenectSource : public Source
{
public:


	enum TargetIdx
	{
//		PointXYZI = 0,
//		PointRGBA = 1,
//		SensorInfoList = 2,

		PointXYZI,
		PointRGBA,
		PointIntensity,
		SensorInfoList,

		ImageDepth,
		ImageRGB,
	};

	SyncFreenectSource(unsigned int n_view_);
	virtual ~SyncFreenectSource();

	void init();
	void loadFrame();


	void TestSyncnectSource(thrust::host_vector<float4>& views);
//	void close();

private:
	void setRegInfo();

private:
	dim3 grid,block;

	uint8_t *d_rgb;
	uint16_t *d_depth;

	float4 *d_test_xyzi;
	uchar4 *d_test_rgba;

	double ref_pix_size;
	double ref_dist;

	unsigned int n_view;
};


#endif /* SYNCFREENECTSOURCE_H_ */

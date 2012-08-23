/*
 * SyncFreenectSource.h
 *
 *  Created on: Aug 23, 2012
 *      Author: avo
 */

#ifndef SYNCFREENECTSOURCE_H_
#define SYNCFREENECTSOURCE_H_

#include <source.h>
#include <vector_types.h>
#include <stdint.h>

class SyncFreenectSource : public Source
{
public:

	enum TargetIdx
	{
		PointXYZI = 0,
		PointRGBA = 1,
	};

	SyncFreenectSource();
	virtual ~SyncFreenectSource();

	void init();
	void loadFrame();

private:
	void setRegInfo();

private:
	dim3 grid,block;

	uint8_t *d_rgb;
	uint16_t *d_depth;
};


#endif /* SYNCFREENECTSOURCE_H_ */

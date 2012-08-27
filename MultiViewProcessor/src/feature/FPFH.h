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
public:
	FPFH();
	virtual ~FPFH();

	void execute();
	void init();
};

#endif /* FPFH_H_ */


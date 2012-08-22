/*
 * TestFilter.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef TESTFILTER_H_
#define TESTFILTER_H_

#include "filter.hpp"

class TestFilter : public Filter {
public:

	void init();
	void execute();
};

#endif /* TESTFILTER_H_ */

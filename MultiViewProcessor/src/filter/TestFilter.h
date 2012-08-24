/*
 * TestFilter.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef TESTFILTER_H_
#define TESTFILTER_H_

#include "filter.hpp"
#include <boost/shared_ptr.hpp>

class TestFilter : public Filter {


public:

//	boost::shared_ptr<TestFilter> Ptr;
//
//	TestFilter()
//	{
//		Ptr = boost::shared_ptr<TestFilter>(this);
//	}

	void init();
	void execute();
};

#endif /* TESTFILTER_H_ */

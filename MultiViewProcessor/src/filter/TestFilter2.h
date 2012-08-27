/*
 * TestFilter2.h
 *
 *  Created on: Aug 25, 2012
 *      Author: avo
 */

#ifndef TESTFILTER2_H_
#define TESTFILTER2_H_

#include "filter.hpp"

class TestFilter2 : public Filter {

public:
	enum Target
	{
		IndiceBlock,
		PointXYZ,
	};

	class Input
	{
	public:
		enum
		{
			Test1,
			Test2,
			Test3
		};
	};

	TestFilter2();
	virtual ~TestFilter2();

	void init();
	void execute();
};

#endif /* TESTFILTER2_H_ */

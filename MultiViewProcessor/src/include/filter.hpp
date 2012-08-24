#ifndef __FILTER_HPP__
#define __FILTER_HPP__

#include "enhancer.hpp"
#include <device_data.h>


class Filter : public Enhancer
{
public:

//	virtual void init() = 0;
//	virtual void execute() = 0;

protected:

};

typedef boost::shared_ptr<Filter> FilterPtr;

#endif /* __FILTER_HPP__ */

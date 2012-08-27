#ifndef __SINK_HPP__
#define __SINK_HPP__

#include "processor.h"

class Sink
{
	Processor processor;

public:
	void setProcessor(Processor processor_)
	{
		processor = processor_;
	}

	virtual void start() = 0;

	virtual ~Sink();
};


#endif /* __SINK_HPP__ */

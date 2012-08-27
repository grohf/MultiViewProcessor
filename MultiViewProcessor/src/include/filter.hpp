#ifndef __FILTER_HPP__
#define __FILTER_HPP__

#include "enhancer.hpp"
#include <device_data.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <map>

typedef boost::function1<void ,DeviceDataParams&> DeviceDataParamFunction;

class Filter : public Enhancer
{
std::map<unsigned int,DeviceDataParamFunction> paramChangeMap;

public:

	virtual void init() = 0;
	virtual void execute() = 0;

	virtual void paramsChange()
	{

	}

	void addFilterInput(DeviceDataInfoPtr ddi,unsigned int idx=0)
	{
		addInputData(ddi,idx);
		DeviceDataParams params = ddi->getDeviceDataParams();
		std::map<unsigned int,DeviceDataParamFunction>::iterator it = paramChangeMap.find(idx);
		if(it != paramChangeMap.end())
		{
			DeviceDataParamFunction f = it->second;
			f(params);
		}
		addTargetData(addDeviceDataRequest(params),idx);
	}

protected:

	void addParamChanger(DeviceDataParamFunction f,unsigned int idx=0)
	{
		paramChangeMap[idx] = f;
	}

};

typedef boost::shared_ptr<Filter> FilterPtr;


#endif /* __FILTER_HPP__ */

/*
 * module.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef MODULE_H_
#define MODULE_H_

#include <device_data.h>
#include <boost/shared_ptr.hpp>

#include <vector_types.h>
#include <boost/container/map.hpp>
#include <vector>
#include <map>

namespace device
{
	struct BaseKernel
	{
		enum
		{
			WARP_SIZE = 32,
		};
	};
}

class Module {
public:

	virtual ~Module() {}

	virtual void init() = 0;
	virtual void execute() = 0;
};

typedef boost::shared_ptr<Module> ModulePtr;

template<typename T = unsigned int>
class InputLister
{
public:
	void addInputData(DeviceDataInfoPtr dInfoPtr, T id)
	{
		map[id] = dInfoPtr;
	}
protected:
	DeviceDataInfoPtr getInputData(T id)
	{
		return map[id];
	}

	void* getInputDataPointer(T id)
	{
		return map[id]->getDeviceDataPtr().get();
	}

private:
//	std::vector<DeviceDataInfoPtr> list;
	std::map<T,DeviceDataInfoPtr> map;

};

template<typename T = unsigned int>
class TargetLister
{
public:
	DeviceDataInfoPtr getTargetData(T id)
	{
		return map[id];
	}

protected:
	void addTargetData(DeviceDataInfoPtr dInfoPtr, T id)
	{
		map[id] = dInfoPtr;
	}

	void* getTargetDataPointer(T id)
	{
		return map[id]->getDeviceDataPtr().get();
	}


private:
//	std::vector<DeviceDataInfoPtr> list;
	std::map<T,DeviceDataInfoPtr> map;
};

#endif /* MODULE_H_ */

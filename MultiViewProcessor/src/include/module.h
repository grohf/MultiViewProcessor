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

#include <boost/container/map.hpp>
#include <vector>
#include <map>

class Module {
public:

	virtual ~Module() {}

	virtual void init() = 0;
	virtual void execute() = 0;
};

typedef boost::shared_ptr<Module> ModulePtr;

template<typename T = unsigned int>
class SourceLister
{
public:
	void addSourceData(DeviceDataInfoPtr dInfoPtr, T id)
	{
		map[id] = dInfoPtr;
	}
protected:
	DeviceDataInfoPtr getSourceData(T id)
	{
		return map[id];
	}

	DeviceDataPtr getSourceDataPointer(T id)
	{
		return map[id]->getDeviceDataPtr();
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
//		if(list.size()<idx+1)
//			list.resize(idx+1);

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

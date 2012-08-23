/*
 * module.h
 *
 *  Created on: Aug 22, 2012
 *      Author: avo
 */

#ifndef MODULE_H_
#define MODULE_H_

#include <device_data.h>
#include <vector>

class Module {
public:
	virtual ~Module() {}

	virtual void init() = 0;
	virtual void execute() = 0;
};

//template<typename T>
//class Lister
//{
//public:
//	void addData(T data,unsigned int idx)
//	{
//		list[idx] = data;
//	}
//
//
//private:
//	std::vector<T> list;
//};

class SourceLister
{
public:
	void addSourceData(DeviceDataInfoPtr dInfoPtr, unsigned int idx)
	{
		list[idx] = dInfoPtr;
	}
protected:
	DeviceDataInfoPtr getSourceData(unsigned int idx)
	{
		return list[idx];
	}

	DeviceDataPtr getSourceDataPointer(unsigned int idx)
	{
		return list[idx]->ptr;
	}

private:
	std::vector<DeviceDataInfoPtr> list;

};

class TargetLister
{
public:
	DeviceDataInfoPtr getTargetData(unsigned int idx)
	{
		return list[idx];
	}

protected:
	void addTargetData(DeviceDataInfoPtr dInfoPtr, unsigned int idx)
	{
		if(list.size()<idx+1)
			list.resize(idx+1);

		list[idx] = dInfoPtr;
	}

	DeviceDataPtr getTargetDataPointer(unsigned int idx)
	{
		return list[idx]->ptr;
	}


private:
	std::vector<DeviceDataInfoPtr> list;
};

#endif /* MODULE_H_ */

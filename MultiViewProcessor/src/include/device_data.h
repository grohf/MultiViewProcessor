#ifndef __DEVICE_DATA_H__
#define __DEVICE_DATA_H__

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/container/vector.hpp>

enum DeviceDataType
{
	Point = 0x00,
	Point1D = 0x01,
	Point2D = 0x02,
	Point3D = 0x03,
	Point4D = 0x04,

	Indice = 0x11,
	BlockIndice = 0x12,
	WarpIndice = 0x13,

	Histogramm 	= 0x20,
	Sigma		= 0x30,
};

enum ElementType
{
	UCHAR1 	= 0x01,
	UCHAR2 	= 0x02,
	UCHAR3 	= 0x03,
	UCHAR4 	= 0x04,

	UINT1 	= 0x21,
	UINT2 	= 0x22,
	UINT3 	= 0x23,
	UINT4 	= 0x24,

	FLOAT1 	= 0x31,
	FLOAT2 	= 0x32,
	FLOAT3 	= 0x33,
	FLOAT4 	= 0x34,

	BINS16 = 0x41,

};

//template<typename T>
//void test(T value);

typedef struct
{
	unsigned int elements;
	unsigned int element_size;
	unsigned int capacity;

	DeviceDataType dataType;
	ElementType elementType;
}DeviceDataParams;


typedef boost::shared_ptr<void> DeviceDataPtr;

class Processor;
class DeviceDataInfo
{
	friend class Processor;

	DeviceDataPtr ptr;
	DeviceDataParams params;

	void setDeviceDataPtr(DeviceDataPtr d_ptr)
	{
		ptr = d_ptr;
	}

public:
	DeviceDataInfo(DeviceDataParams params_)
	{
		params = params_;
	}

	DeviceDataPtr getDeviceDataPtr() const
	{
		return ptr;
	}

	DeviceDataParams getDeviceDataParams() const
	{
		return params;
	}

};

typedef boost::shared_ptr<DeviceDataInfo> DeviceDataInfoPtr;


class DeviceDataRequester
{
public:

	~DeviceDataRequester() {}

	std::vector<DeviceDataInfoPtr> getRequestedDeviceDataInfoPtrList()
	{
		return list;
	}

	DeviceDataInfoPtr addDeviceDataRequest(DeviceDataParams params)
	{

		DeviceDataInfo *ddi = new DeviceDataInfo(params);
		DeviceDataInfoPtr ptr(ddi);

		list.push_back(ptr);

		return ptr;
	}



private:
	std::vector<DeviceDataInfoPtr> list;
};


#endif /* __DEVICE_DATA_H__ */

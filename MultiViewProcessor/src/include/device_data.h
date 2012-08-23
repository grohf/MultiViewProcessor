#ifndef __DEVICE_DATA_H__
#define __DEVICE_DATA_H__

#include <vector>

enum DeviceDataType
{
	Point = 0x00,
	Point1D = 0x01,
	Point2D = 0x02,
	Point3D = 0x03,
	Point4D = 0x04,

	Indice = 0x11,
	BlockIndice = 0x12,
	WarpIndice = 0x13
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
};

typedef struct
{
	unsigned int elements;
	unsigned int element_size;
	unsigned int capacity;

	DeviceDataType dataType;
	ElementType elementType;
}DeviceDataParams;


typedef void* DeviceDataPtr;

typedef struct
{
	DeviceDataPtr ptr;
	DeviceDataParams params;
}DeviceDataInfo;

typedef DeviceDataInfo* DeviceDataInfoPtr;


class DeviceDataRequester
{
public:

	virtual ~DeviceDataRequester() {}

	virtual std::vector<DeviceDataInfoPtr> getRequestedDeviceDataInfoPtrList()
	{
		return list;
	}

	virtual DeviceDataInfoPtr addDeviceDataRequest(DeviceDataParams params)
	{
		DeviceDataInfo ddi;
		ddi.params = params;

		DeviceDataInfoPtr ptr = &ddi;
		list.push_back(ptr);

		return ptr;
	}


private:
	std::vector<DeviceDataInfoPtr> list;
};


#endif /* __DEVICE_DATA_H__ */

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

typedef struct
{
	unsigned int elements;
	unsigned int element_size;
	unsigned int capacity;

	DeviceDataType type;
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

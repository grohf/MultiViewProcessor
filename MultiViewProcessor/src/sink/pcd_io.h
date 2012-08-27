#ifndef __PCD_IO_H__
#define __PCD_IO_H__

namespace host
{
	namespace io
	{
		class PCDIOController
		{
		public:
			/* Exporter to PCL */
			int writeASCIIPCD(char *filename,float *data,unsigned int size);
			int writeASCIIPCDNormals(char *filename,float *data,unsigned int size);

//			int readASCIIPCDXYZRGB(char *filename_depth,char *filename_rgb,uint16_t *depth, uint8_t *rgb);
		};
	}
}


#endif


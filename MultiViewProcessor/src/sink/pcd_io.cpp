
#include "pcd_io.h"

#include <stdio.h>
#include <fstream>
#include <iosfwd>

//#include <iostream>
//#include <stdlib.h>
//#include <string>
//
//#include <cstring>

int host::io::PCDIOController::writeASCIIPCD(char *filename,float *data,unsigned int size)
{

	std::ofstream fs;
	fs.precision(8);
	fs.imbue(std::locale::classic());
	fs.open(filename);

	if(!fs.is_open() || fs.fail() )
	{
		printf("Can't open file! \n");
		return -1;
	}



//	std::ostringstream stream;
//	stream.precision (8);
//	stream.imbue (std::locale::classic ());

	fs << "# .PCD v0.7 - Point Cloud Data file format"
			"\nVERSION 0.7"
			"\nFIELDS x y z"
			"\nSIZE 4 4 4"
			"\nTYPE F F F"
			"\nCOUNT 1 1 1"
			"\nWIDTH 640"
			"\nHEIGHT 480"
			"\nVIEWPOINT 0 0 0 1 0 0 0"
			"\nPOINTS 307200"
			"\nDATA ascii"
			"\n";

	printf("start writeASCIIPCD... \n");
	for(int i=0;i<640*480;i++){
		if(data[i*4+0]==0.0f && data[i*4+1]==0.0f && data[i*4+2]==0.0f)
			fs << "nan nan nan" << std::endl;
		else
			fs << data[i*4+0]/1000.0f << " " << data[i*4+1]/1000.0f << " " << data[i*4+2]/1000.0f << std::endl;
	}

	printf("saved \n");


//	fs << oss;

	return 0;
}

int host::io::PCDIOController::writeASCIIPCDNormals(char *filename,float *data,unsigned int size)
{
	std::ofstream fs;
	fs.precision(8);
	fs.imbue(std::locale::classic());
	fs.open(filename);

	if(!fs.is_open() || fs.fail() )
	{
		printf("Can't open file! \n");
		return -1;
	}

	printf("start writeASCIIPCDNormals... \n");

	fs << "# .PCD v0.7 - Point Cloud Data file format"
			"\nVERSION 0.7"
			"\nFIELDS normal_x normal_y normal_z curvature"
			"\nSIZE 4 4 4 4"
			"\nTYPE F F F F"
			"\nCOUNT 1 1 1 1"
			"\nWIDTH 640"
			"\nHEIGHT 480"
			"\nVIEWPOINT 0 0 0 1 0 0 0"
			"\nPOINTS 307200"
			"\nDATA ascii"
			"\n";

	for(int i=0;i<640*480;i++){
//		if(data[i*4+3]<0.0f)
//			fs << "nan nan nan nan" << std::endl;
//		else
			fs << data[i*4+0] << " " << data[i*4+1] << " " << data[i*4+2] << " " << data[i*4+3] << std::endl;
	}

	printf("saved \n");


//	fs << oss;

	return 0;

}

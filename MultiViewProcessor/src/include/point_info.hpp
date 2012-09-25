/*
 * point_info.hpp
 *
 *  Created on: Sep 24, 2012
 *      Author: avo
 */

#ifndef POINT_INFO_HPP_
#define POINT_INFO_HPP_

#define VALID 			0
#define RECONSTRUCTED 	1
//#define


namespace device
{
	enum PointInfoIdx
	{
		Valid = 0,
		Reconstructed = 1,
		ReconstructionLevel = 2,
		Segmentation = 8,
	};

	__device__ __host__ __forceinline__ bool is_bit_set(unsigned int val, unsigned int idx)
	{
		return (val & (1 << idx)) != 0;
	}

	__device__ __host__ __forceinline__ void set_bit(unsigned int &val, unsigned int idx)
	{
		val |= (1<<idx);
	}



	__device__ __host__ __forceinline__ bool isValid(unsigned int val)
	{
		return is_bit_set(val,Valid);
	}

	__device__ __host__ __forceinline__ void setValid(unsigned int& val)
	{
		set_bit(val,Valid);
	}



	__device__ __host__ __forceinline__ bool isValid(float val)
	{
		return is_bit_set(val,Valid);
	}

	__device__ __host__ __forceinline__ void setValid(float& val)
	{
		val = ((unsigned int)val)|(1<<Valid);
	}
	__device__ __host__ __forceinline__ void unsetValid(float& val)
	{
		val = ((unsigned int)val) & ~(1<<Valid);
	}


	/* RECONSTRUICTION */
	__device__ __host__ __forceinline__ bool isReconstructed(float val)
	{
		return is_bit_set(val,Reconstructed);
	}

	__device__ __host__ __forceinline__ void setReconstructed(float& val)
	{
		val = ((unsigned int)val)|(1<<Reconstructed);
	}

	__device__ __host__ __forceinline__ void setReconstructed(float& val,unsigned int level)
	{
		unsigned int tmp = ((unsigned int)val)|(1<<Reconstructed);
		tmp &= ~(63<<ReconstructionLevel);
		val = tmp | (level<<ReconstructionLevel);
	}

	__device__ __host__ __forceinline__ int getReconstructionLevel(float val)
	{
		if( !is_bit_set(val,Reconstructed) )
			return -1;
		return ( ((unsigned int)val) & (63<<ReconstructionLevel))>>ReconstructionLevel;
	}

	__device__ __host__ __forceinline__ void unsetReconstructed(float& val)
	{
		val = ((unsigned int)val) & ~(1<<Reconstructed);
	}



	enum SegmentationPointInfo
	{
		TBD 		= 0,
		Foreground  = 1,
		Background	= 2,
	};

	__device__ __host__ __forceinline__ void setSegmentationPointInfo(float& val,SegmentationPointInfo spi)
	{
		unsigned int tmp = (unsigned int) val;
		tmp &= ~(3<<Segmentation);
		val = tmp | (spi<<Segmentation);
	}

	__device__ __host__ __forceinline__ bool isSegmentationPointInfo(float& val,SegmentationPointInfo spi)
	{
		unsigned int tmp = (unsigned int) val;
		return ( ((tmp & (3<<Segmentation))>>Segmentation) == spi );
	}

	__device__ __host__ __forceinline__ void setForeground(float& val)
	{
		setSegmentationPointInfo(val,Foreground);
	}

	__device__ __host__ __forceinline__ bool isForeground(float& val)
	{
		return isSegmentationPointInfo(val,Foreground);
	}

	__device__ __host__ __forceinline__ void setBackground(float& val)
	{
		setSegmentationPointInfo(val,Background);
	}

	__device__ __host__ __forceinline__ bool isBackground(float& val)
	{
		return isSegmentationPointInfo(val,Background);
	}


	__device__ __host__ __forceinline__ bool isSegmented(float& val)
	{
		return ! (isSegmentationPointInfo(val,TBD));
	}

	__device__ __host__ __forceinline__ void unsetSegmented(float& val)
	{
		setSegmentationPointInfo(val,TBD);
	}

}






#endif /* POINT_INFO_HPP_ */

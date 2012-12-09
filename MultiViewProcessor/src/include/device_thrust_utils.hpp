/*
 * device_thrust_utils.hpp
 *
 *  Created on: Dec 9, 2012
 *      Author: avo
 */



#ifndef DEVICE_THRUST_UTILS_HPP_
#define DEVICE_THRUST_UTILS_HPP_

#include <thrust/device_vector.h>
#include <thrust/remove.h>

namespace device
{

	template <typename T>
	struct is_not_valid_transform
	{
		template <typename Tuple>  __device__
		bool operator()(const Tuple& tuple) const
		{
			// unpack the tuple into x and y coordinates
			const T x = thrust::get<0>(tuple);

			if (x == -1)
				return true;
			else
				return false;
		}
	};


	template <typename T>
	struct is_not_valid
	{
	    template <typename Tuple>  __device__
	    bool operator()(const Tuple& tuple) const
	    {
	        // unpack the tuple into x and y coordinates
	        const T x = thrust::get<0>(tuple);

	        if (x == -1)
	            return true;
	        else
	            return false;
	    }
	};

}


size_t remove_bad_transformations(thrust::device_vector<int>& d_validTransforms,thrust::device_vector<float>& d_transformMatrices, unsigned int errorListLength, unsigned int view_combinations)
{

//	thrust::device_vector<int> d_validTransforms = d_validTransforms;
	thrust::device_vector<int> d_validTransformationTrans = d_validTransforms;


	size_t size1 = 0;
	size_t size2 = 0;
	for(int v=0;v<view_combinations;v++)
	{

		size1 += thrust::remove_if(
		thrust::make_zip_iterator(
		thrust::make_tuple(
				d_validTransforms.data()+v*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+0*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+1*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+2*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+3*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+4*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+5*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+6*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+7*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+8*errorListLength)),

		thrust::make_zip_iterator(
		thrust::make_tuple(
				d_validTransforms.data()+(v+1)*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+1*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+2*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+3*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+4*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+5*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+6*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+7*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+8*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+9*errorListLength)),

				device::is_not_valid_transform<float>()) -

		thrust::make_zip_iterator(
		thrust::make_tuple(
				d_validTransforms.data()+v*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+0*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+1*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+2*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+3*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+4*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+5*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+6*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+7*errorListLength,
				d_transformMatrices.data()+v*errorListLength*12+8*errorListLength));

		size2 += thrust::remove_if(
			thrust::make_zip_iterator(
			thrust::make_tuple(
					d_validTransformationTrans.data()+v*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+9*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+10*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+11*errorListLength)),

			thrust::make_zip_iterator(
			thrust::make_tuple(
					d_validTransformationTrans.data()+(v+1)*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+10*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+11*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+12*errorListLength)),

					device::is_not_valid_transform<float>()) -

			thrust::make_zip_iterator(
			thrust::make_tuple(
					d_validTransformationTrans.data()+v*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+9*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+10*errorListLength,
					d_transformMatrices.data()+v*errorListLength*12+11*errorListLength));
	}

	return size1;
}


size_t remove2Test()
{
		thrust::device_vector<int> d_test(10);
		thrust::remove(d_test.data(),d_test.data()+10,-1);

		return 2;
}

#endif /* DEVICE_THRUST_UTILS_HPP_ */

/*
 * NormalPCAEstimator.cpp
 *
 *  Created on: Aug 27, 2012
 *      Author: avo
 */

#include "NormalPCAEstimator.h"
//#include "utils.hpp"
#include <helper_cuda.h>
#include <helper_image.h>

namespace device
{

template <class T>
__device__ __host__ __forceinline__ void swap ( T& a, T& b )
{
  T c(a); a=b; b=c;
}

template<typename T> struct numeric_limits;

template<> struct numeric_limits<float>
{
  __device__ __forceinline__ static float
  quiet_NaN() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
  __device__ __forceinline__ static float
  epsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };

  __device__ __forceinline__ static float
  min() { return 1.175494351e-38f/*FLT_MIN*/; };
  __device__ __forceinline__ static float
  max() { return 3.402823466e+38f/*FLT_MAX*/; };
};

//    template<> struct numeric_limits<short>
//    {
//      __device__ __forceinline__ static short
//      max() { return SHRT_MAX; };
//    };

}

namespace device
{

	__device__ __forceinline__ float
	dot(const float3& v1, const float3& v2)
	{
	  return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
	}

	__device__ __forceinline__ float3&
	operator+=(float3& vec, const float& v)
	{
	  vec.x += v;  vec.y += v;  vec.z += v; return vec;
	}

	__device__ __forceinline__ float3
	operator+(const float3& v1, const float3& v2)
	{
	  return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	__device__ __forceinline__ float3&
	operator*=(float3& vec, const float& v)
	{
	  vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
	}

	__device__ __forceinline__ float3
	operator-(const float3& v1, const float3& v2)
	{
	  return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	__device__ __forceinline__ float3
	operator*(const float3& v1, const float& v)
	{
	  return make_float3(v1.x * v, v1.y * v, v1.z * v);
	}

	__device__ __forceinline__ float
	norm(const float3& v)
	{
	  return sqrt(dot(v, v));
	}

	__device__ __forceinline__ float3
	normalized(const float3& v)
	{
	  return v * rsqrt(dot(v, v));
	}

	__device__ __host__ __forceinline__ float3
	cross(const float3& v1, const float3& v2)
	{
	  return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	__device__ __forceinline__ void computeRoots2(const float& b, const float& c, float3& roots)
	 {
	   roots.x = 0.f;
	   float d = b * b - 4.f * c;
	   if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		 d = 0.f;

	   float sd = sqrtf(d);

	   roots.z = 0.5f * (b + sd);
	   roots.y = 0.5f * (b - sd);
	 }

	 __device__ __forceinline__ void
	 computeRoots3(float c0, float c1, float c2, float3& roots)
	 {
	   if ( fabsf(c0) < numeric_limits<float>::epsilon())// one root is 0 -> quadratic equation
	   {
		 computeRoots2 (c2, c1, roots);
	   }
	   else
	   {
		 const float s_inv3 = 1.f/3.f;
		 const float s_sqrt3 = sqrtf(3.f);
		 // Construct the parameters used in classifying the roots of the equation
		 // and in solving the equation for the roots in closed form.
		 float c2_over_3 = c2 * s_inv3;
		 float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		 if (a_over_3 > 0.f)
		   a_over_3 = 0.f;

		 float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		 float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		 if (q > 0.f)
		   q = 0.f;

		 // Compute the eigenvalues by solving for the roots of the polynomial.
		 float rho = sqrtf(-a_over_3);
		 float theta = atan2f (sqrtf (-q), half_b)*s_inv3;
		 float cos_theta = __cosf (theta);
		 float sin_theta = __sinf (theta);
		 roots.x = c2_over_3 + 2.f * rho * cos_theta;
		 roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		 roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		 // Sort in increasing order.
		 if (roots.x >= roots.y)
		   swap(roots.x, roots.y);

		 if (roots.y >= roots.z)
		 {
		   swap(roots.y, roots.z);

		   if (roots.x >= roots.y)
			 swap (roots.x, roots.y);
		 }
		 if (roots.x <= 0) // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
		   computeRoots2 (c2, c1, roots);
	   }
	 }


struct Eigen33
{
public:
  template<int Rows>
  struct MiniMat
  {
    float3 data[Rows];
    __device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
    __device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
  };
  typedef MiniMat<3> Mat33;
  typedef MiniMat<4> Mat43;


  static __forceinline__ __device__ float3
  unitOrthogonal (const float3& src)
  {
    float3 perp;
    /* Let us compute the crossed product of *this with a vector
    * that is not too close to being colinear to *this.
    */

    /* unless the x and y coords are both close to zero, we can
    * simply take ( -y, x, 0 ) and normalize it.
    */
    if(!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
    {
      float invnm = rsqrtf(src.x*src.x + src.y*src.y);
      perp.x = -src.y * invnm;
      perp.y =  src.x * invnm;
      perp.z = 0.0f;
    }
    /* if both x and y are close to zero, then the vector is close
    * to the z-axis, so it's far from colinear to the x-axis for instance.
    * So we take the crossed product with (1,0,0) and normalize it.
    */
    else
    {
      float invnm = rsqrtf(src.z * src.z + src.y * src.y);
      perp.x = 0.0f;
      perp.y = -src.z * invnm;
      perp.z =  src.y * invnm;
    }

    return perp;
  }

  __device__ __forceinline__
  Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}
  __device__ __forceinline__ void
  compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
  {
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.

    float max01 = fmaxf( fabsf(mat_pkg[0]), fabsf(mat_pkg[1]) );
    float max23 = fmaxf( fabsf(mat_pkg[2]), fabsf(mat_pkg[3]) );
    float max45 = fmaxf( fabsf(mat_pkg[4]), fabsf(mat_pkg[5]) );
    float m0123 = fmaxf( max01, max23);
    float scale = fmaxf( max45, m0123);

    if (scale <= numeric_limits<float>::min())
      scale = 1.f;

    mat_pkg[0] /= scale;
    mat_pkg[1] /= scale;
    mat_pkg[2] /= scale;
    mat_pkg[3] /= scale;
    mat_pkg[4] /= scale;
    mat_pkg[5] /= scale;

    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // real-valued, because the matrix is symmetric.
    float c0 = m00() * m11() * m22()
        + 2.f * m01() * m02() * m12()
        - m00() * m12() * m12()
        - m11() * m02() * m02()
        - m22() * m01() * m01();
    float c1 = m00() * m11() -
        m01() * m01() +
        m00() * m22() -
        m02() * m02() +
        m11() * m22() -
        m12() * m12();
    float c2 = m00() + m11() + m22();

    computeRoots3(c0, c1, c2, evals);

    if(evals.z - evals.x <= numeric_limits<float>::epsilon())
    {
      evecs[0] = make_float3(1.f, 0.f, 0.f);
      evecs[1] = make_float3(0.f, 1.f, 0.f);
      evecs[2] = make_float3(0.f, 0.f, 1.f);
    }
    else if (evals.y - evals.x <= numeric_limits<float>::epsilon() )
    {
      // first and second equal
      tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
      tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

      vec_tmp[0] = cross(tmp[0], tmp[1]);
      vec_tmp[1] = cross(tmp[0], tmp[2]);
      vec_tmp[2] = cross(tmp[1], tmp[2]);

      float len1 = dot (vec_tmp[0], vec_tmp[0]);
      float len2 = dot (vec_tmp[1], vec_tmp[1]);
      float len3 = dot (vec_tmp[2], vec_tmp[2]);

      if (len1 >= len2 && len1 >= len3)
      {
        evecs[2] = vec_tmp[0] * rsqrtf (len1);
      }
      else if (len2 >= len1 && len2 >= len3)
      {
        evecs[2] = vec_tmp[1] * rsqrtf (len2);
      }
      else
      {
        evecs[2] = vec_tmp[2] * rsqrtf (len3);
      }

      evecs[1] = unitOrthogonal(evecs[2]);
      evecs[0] = cross(evecs[1], evecs[2]);
    }
    else if (evals.z - evals.y <= numeric_limits<float>::epsilon() )
    {
      // second and third equal
      tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
      tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

      vec_tmp[0] = cross(tmp[0], tmp[1]);
      vec_tmp[1] = cross(tmp[0], tmp[2]);
      vec_tmp[2] = cross(tmp[1], tmp[2]);

      float len1 = dot(vec_tmp[0], vec_tmp[0]);
      float len2 = dot(vec_tmp[1], vec_tmp[1]);
      float len3 = dot(vec_tmp[2], vec_tmp[2]);

      if (len1 >= len2 && len1 >= len3)
      {
        evecs[0] = vec_tmp[0] * rsqrtf(len1);
      }
      else if (len2 >= len1 && len2 >= len3)
      {
        evecs[0] = vec_tmp[1] * rsqrtf(len2);
      }
      else
      {
        evecs[0] = vec_tmp[2] * rsqrtf(len3);
      }

      evecs[1] = unitOrthogonal( evecs[0] );
      evecs[2] = cross(evecs[0], evecs[1]);
    }
    else
    {

      tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
      tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

      vec_tmp[0] = cross(tmp[0], tmp[1]);
      vec_tmp[1] = cross(tmp[0], tmp[2]);
      vec_tmp[2] = cross(tmp[1], tmp[2]);

      float len1 = dot(vec_tmp[0], vec_tmp[0]);
      float len2 = dot(vec_tmp[1], vec_tmp[1]);
      float len3 = dot(vec_tmp[2], vec_tmp[2]);

      float mmax[3];

      unsigned int min_el = 2;
      unsigned int max_el = 2;
      if (len1 >= len2 && len1 >= len3)
      {
        mmax[2] = len1;
        evecs[2] = vec_tmp[0] * rsqrtf (len1);
      }
      else if (len2 >= len1 && len2 >= len3)
      {
        mmax[2] = len2;
        evecs[2] = vec_tmp[1] * rsqrtf (len2);
      }
      else
      {
        mmax[2] = len3;
        evecs[2] = vec_tmp[2] * rsqrtf (len3);
      }

      tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
      tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

      vec_tmp[0] = cross(tmp[0], tmp[1]);
      vec_tmp[1] = cross(tmp[0], tmp[2]);
      vec_tmp[2] = cross(tmp[1], tmp[2]);

      len1 = dot(vec_tmp[0], vec_tmp[0]);
      len2 = dot(vec_tmp[1], vec_tmp[1]);
      len3 = dot(vec_tmp[2], vec_tmp[2]);

      if (len1 >= len2 && len1 >= len3)
      {
        mmax[1] = len1;
        evecs[1] = vec_tmp[0] * rsqrtf (len1);
        min_el = len1 <= mmax[min_el] ? 1 : min_el;
        max_el = len1  > mmax[max_el] ? 1 : max_el;
      }
      else if (len2 >= len1 && len2 >= len3)
      {
        mmax[1] = len2;
        evecs[1] = vec_tmp[1] * rsqrtf (len2);
        min_el = len2 <= mmax[min_el] ? 1 : min_el;
        max_el = len2  > mmax[max_el] ? 1 : max_el;
      }
      else
      {
        mmax[1] = len3;
        evecs[1] = vec_tmp[2] * rsqrtf (len3);
        min_el = len3 <= mmax[min_el] ? 1 : min_el;
        max_el = len3 >  mmax[max_el] ? 1 : max_el;
      }

      tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
      tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

      vec_tmp[0] = cross(tmp[0], tmp[1]);
      vec_tmp[1] = cross(tmp[0], tmp[2]);
      vec_tmp[2] = cross(tmp[1], tmp[2]);

      len1 = dot (vec_tmp[0], vec_tmp[0]);
      len2 = dot (vec_tmp[1], vec_tmp[1]);
      len3 = dot (vec_tmp[2], vec_tmp[2]);


      if (len1 >= len2 && len1 >= len3)
      {
        mmax[0] = len1;
        evecs[0] = vec_tmp[0] * rsqrtf (len1);
        min_el = len3 <= mmax[min_el] ? 0 : min_el;
        max_el = len3  > mmax[max_el] ? 0 : max_el;
      }
      else if (len2 >= len1 && len2 >= len3)
      {
        mmax[0] = len2;
        evecs[0] = vec_tmp[1] * rsqrtf (len2);
        min_el = len3 <= mmax[min_el] ? 0 : min_el;
        max_el = len3  > mmax[max_el] ? 0 : max_el;
      }
      else
      {
        mmax[0] = len3;
        evecs[0] = vec_tmp[2] * rsqrtf (len3);
        min_el = len3 <= mmax[min_el] ? 0 : min_el;
        max_el = len3  > mmax[max_el] ? 0 : max_el;
      }

      unsigned mid_el = 3 - min_el - max_el;
      evecs[min_el] = normalized( cross( evecs[(min_el+1) % 3], evecs[(min_el+2) % 3] ) );
      evecs[mid_el] = normalized( cross( evecs[(mid_el+1) % 3], evecs[(mid_el+2) % 3] ) );
    }
    // Rescale back to the original size.
    evals *= scale;
  }
private:
  volatile float* mat_pkg;

  __device__  __forceinline__ float m00() const { return mat_pkg[0]; }
  __device__  __forceinline__ float m01() const { return mat_pkg[1]; }
  __device__  __forceinline__ float m02() const { return mat_pkg[2]; }
  __device__  __forceinline__ float m10() const { return mat_pkg[1]; }
  __device__  __forceinline__ float m11() const { return mat_pkg[3]; }
  __device__  __forceinline__ float m12() const { return mat_pkg[4]; }
  __device__  __forceinline__ float m20() const { return mat_pkg[2]; }
  __device__  __forceinline__ float m21() const { return mat_pkg[4]; }
  __device__  __forceinline__ float m22() const { return mat_pkg[5]; }

  __device__  __forceinline__ float3 row0() const { return make_float3( m00(), m01(), m02() ); }
  __device__  __forceinline__ float3 row1() const { return make_float3( m10(), m11(), m12() ); }
  __device__  __forceinline__ float3 row2() const { return make_float3( m20(), m21(), m22() ); }

  __device__  __forceinline__ static bool isMuchSmallerThan (float x, float y)
  {
      // copied from <eigen>/include/Eigen/src/Core/NumTraits.h
      const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon();
      return x * x <= prec_sqr * y * y;
  }
};


	struct NormalEstimator
	{

		float4 *input;
		float4 *output;

		enum
		{
			kxr = 3,
			kyr = kxr,

			kx = 32+kxr*2,
			ky = 24+kyr*2,

			kl = 2*kxr+1
		};


		__device__ __forceinline__ void
		operator () () const
		{

			__shared__ float4 shm[kx*ky];

			int sx,sy,off;

			/* ---------- LOAD SHM ----------- */
			const int oy = blockIdx.y*blockDim.y-kyr;
			const int ox = blockIdx.x*blockDim.x-kxr;


			for(off=threadIdx.y*blockDim.x+threadIdx.x;off<kx*ky;off+=blockDim.x*blockDim.y){
				sy = off/kx;
				sx = off - sy*kx;

				sy = oy + sy;
				sx = ox + sx;

				if(sx < 0) 		sx	=	0;
				if(sx > 639) 	sx 	= 639;
				if(sy < 0)		sy 	= 	0;
				if(sy > 479)	sy 	= 479;

				shm[off]=input[blockIdx.z*640*480+sy*640+sx];
			}
			__syncthreads();

			off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
			float3 mid = make_float3(shm[off].x,shm[off].y,shm[off].z);

			if(blockIdx.x==10 && blockIdx.y==2 && threadIdx.x==10 && threadIdx.y==10)
				printf("soso \n");



//			if(mid.z==0)
//			{
//				  sx = threadIdx.x + blockIdx.x * blockDim.x;
//				  sy = threadIdx.y + blockIdx.y * blockDim.y;
//				  float4 f4 = make_float4(0,0,0,-2);
////				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
//				  output[blockIdx.z*640*480+sy*640+sx] = f4;
//				  return;
//			}
//
//			if(mid.z>4000)
//			{
//				  sx = threadIdx.x + blockIdx.x * blockDim.x;
//				  sy = threadIdx.y + blockIdx.y * blockDim.y;
//				  float4 f4 = make_float4(0,0,0,-3);
////				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
//				  output[blockIdx.z*640*480+sy*640+sx] = f4;
//				  return;
//			}


			float3 mean = make_float3(0.0f,0.0f,0.0f);

			unsigned int count = 0;
			for(sy=0;sy<kl;sy++)
			{
				for(sx=0;sx<kl;sx++)
				{
					off = (threadIdx.y+sy)*kx+threadIdx.x+sx;

					if(sqrtf( (mid.x-shm[off].x)*(mid.x-shm[off].x)+(mid.y-shm[off].y)*(mid.y-shm[off].y)+(mid.z-shm[off].z)*(mid.z-shm[off].z) ) < 20)
					{
						mean.x += shm[off].x;
						mean.y += shm[off].y;
						mean.z += shm[off].z;
						count++;
					}

				}
			}

//				if(count < (kl*kl)/2)
//				{
//			      sx = threadIdx.x + blockIdx.x * blockDim.x;
//			      sy = threadIdx.y + blockIdx.y * blockDim.y;
//			      float4 f4 = make_float4(0,0,0,-7);
//			      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
//			      return;
//
//				}

			if(count==0)
			{
				  sx = threadIdx.x + blockIdx.x * blockDim.x;
				  sy = threadIdx.y + blockIdx.y * blockDim.y;
				  float4 f4 = make_float4(0,0,0,-5);
//				      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
				  output[blockIdx.z*640*480+sy*640+sx] = f4;
				  return;
			}
			if(count==0) printf("AHHHHHHHHH \n");
			mean *= 1.0f/count;

			float cov[] = {0,0,0,0,0,0};

			for(sy=0;sy<kl;sy++)
			{
				for(sx=0;sx<kl;sx++)
				{

					off = (threadIdx.y+sy)*kx+threadIdx.x+sx;

					if(sqrtf( (mid.x-shm[off].x)*(mid.x-shm[off].x)+(mid.y-shm[off].y)*(mid.y-shm[off].y)+(mid.z-shm[off].z)*(mid.z-shm[off].z) ) < 20)
					{
						float3 v;
						v.x = shm[off].x;
						v.y = shm[off].y;
						v.z = shm[off].z;


						float3 d = v - mean;

						cov[0] += d.x * d.x;               //cov (0, 0
						cov[1] += d.x * d.y;               //cov (0, 1)
						cov[2] += d.x * d.z;               //cov (0, 2)
						cov[3] += d.y * d.y;               //cov (1, 1)
						cov[4] += d.y * d.z;               //cov (1, 2)
						cov[5] += d.z * d.z;               //cov (2, 2)
					}
				}
			}

//		bool br = false;
//		for(int i=0;i<5;i++)
//		{
//			if(cov[i]==0)
//				br = true;
//
//		}
//
//		if(br)
//		{
//			sx = threadIdx.x + blockIdx.x * blockDim.x;
//			sy = threadIdx.y + blockIdx.y * blockDim.y;
//			float4 f4 = make_float4(0,0,0,-7);
//
//			output[blockIdx.z*640*480+sy*640+sx] = f4;
//			return;
//		}

//		float mult = 0.001f;
//		cov[0] = cov[0] / 1000.0f;
//		cov[1] = cov[1] * mult;
//		cov[2] = cov[2] * mult;
//		cov[3] = cov[3] * mult;
//		cov[4] = cov[4] * mult;
//		cov[5] = cov[5] * mult;


//		if(blockIdx.x==10 && blockIdx.y==2)
			printf("problem! cov[0]:%f | cov[1]:%f | cov[2]:%f | cov[3]:%f | cov[4]:%f | cov[5]:%f \n",cov[0],cov[1],cov[2],cov[3],cov[4],cov[5]);

		sx = threadIdx.x + blockIdx.x * blockDim.x;
		sy = threadIdx.y + blockIdx.y * blockDim.y;

		cov[0]=268.049866;
		cov[1]=-9.961166;
		cov[2]=377.211151;
		cov[3]=1330.253662;
		cov[4]=783.158630;
		cov[5]=1229.697510;

		typedef Eigen33::Mat33 Mat33;
		Eigen33 eigen33(cov);

		Mat33 tmp;
		Mat33 vec_tmp;
		Mat33 evecs;
		float3 evals;


		eigen33.compute(tmp, vec_tmp, evecs, evals);
/*

//		      float3 n = normalized (evecs[0]);

		  float3 n = evecs[0];


		  if(n.x==1 || n.x==-1)
		  {
//		    	  surf3Dwrite<float4>(make_float4(0,0,0,-4),surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
			  output[blockIdx.z*640*480+sy*640+sx] = make_float4(0,0,0,-4);
			  return;
		  }
		  if( dot(n,mid) >= 0.0f)
		  {
//		    	  n = make_float3(0,0,0);
//		    	  float4 f4 = make_float4(0,0,0,1);
//		    	  surf3Dwrite<float4>(f4,surf::surfRef,sx*sizeof(float4),sy,blockIdx.z*2);
//		    	  return;

			  n.x *= -1.0f;
			  n.y *= -1.0f;
			  n.z *= -1.0f;

		  }

		  float eig_sum = evals.x + evals.y + evals.z;
		  float curvature = (eig_sum == 0) ? 0 : fabsf( evals.x / eig_sum );
		  float4 f4 = make_float4(n.x,n.y,n.z,curvature);

//		      surf3Dwrite<float4>(f4,surf::surfRefBuffer,sx*sizeof(float4),sy,blockIdx.z);
		  output[blockIdx.z*640*480+sy*640+sx] = f4;
*/

		  off = (threadIdx.y+kyr)*kx+threadIdx.x+kxr;
		  output[blockIdx.z*640*480+sy*640+sx] = shm[off];
		}

	};

	__global__ void estimateNormalKernel(const NormalEstimator ne) {ne (); }
}
device::NormalEstimator normalEstimator;

void
NormalPCAEstimator::init()
{
	block = dim3(32,24);
	grid = dim3(640/block.x,480/block.y,1);

//	block = dim3(1);
//	grid = dim3(640/block.x,2,1);
//	grid = dim3(1,1,1);

	normalEstimator.input = (float4 *)getInputDataPointer(WorldCoordinates);
	normalEstimator.output = (float4 *)getTargetDataPointer(Normals);

}

void
NormalPCAEstimator::execute()
{
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	device::estimateNormalKernel<<<grid,block>>>(normalEstimator);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	size_t uc4s = 640*480*sizeof(uchar4);
	char path[50];
	float4 *h_f4_depth = (float4 *)malloc(640*480*sizeof(float4));
	checkCudaErrors(cudaMemcpy(h_f4_depth,normalEstimator.output,640*480*sizeof(float4),cudaMemcpyDeviceToHost));
	uchar4 *h_uc4_depth = (uchar4 *)malloc(uc4s);
	for(int i=0;i<640*480;i++)
	{
		unsigned char g = h_f4_depth[i].z/20;
		h_uc4_depth[i] = make_uchar4(g,g,g,128);
	}

	sprintf(path,"/home/avo/pcds/src_normal_shm%d.ppm",0);
	sdkSavePPM4ub(path,(unsigned char*)h_uc4_depth,640,480);

	printf("normals done \n");
}

NormalPCAEstimator::NormalPCAEstimator()
{
	DeviceDataParams params;
	params.elements = 640*480;
	params.element_size = sizeof(float4);
	params.dataType = Point4D;
	params.elementType = FLOAT4;

	addTargetData(addDeviceDataRequest(params),Normals);
}

NormalPCAEstimator::~NormalPCAEstimator()
{

}


// this implementation is a modification from user G_fans' implementation
// https://devtalk.nvidia.com/default/topic/673965/are-there-any-cuda-libararies-for-3x3-matrix-amp-vector3-amp-quaternion-operations-/
// which is a modification from
// https://github.com/erwincoumans/experiments/blob/master/opencl/primitives/AdlPrimitives/Math/MathCL.h

#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include "../ext/helper_math.h"

#ifndef UNIFIED_MATH_CUDA_H
#define UNIFIED_MATH_CUDA_H

/*****************************************
Vector
/*****************************************/

__host__ __device__
inline float4 cross3(const float4 & a, const float4 & b)
{
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0f);
}

__device__ float4 getNormalizedVec(const float4 & v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3(const float4 & a, const float4 & b)
{
	float4 a1 = make_float4(a.x, a.y, a.z, 0.f);
	float4 b1 = make_float4(b.x, b.y, b.z, 0.f);
	return dot(a1, b1);
}

/*****************************************
Matrix3x4
/*****************************************/
typedef struct
{
	float3 col[3];
}matrix3;

__host__ __device__
inline matrix3 make_matrix3()
{
	matrix3 m;
	m.col[0] = make_float3(1, 0, 0);
	m.col[1] = make_float3(0, 1, 0);
	m.col[2] = make_float3(0, 0, 1);
	return m;
}

__host__ __device__
inline matrix3 make_scale_matrix3(float x)
{
	matrix3 m;
	m.col[0] = make_float3(x, 0, 0);
	m.col[1] = make_float3(0, x, 0);
	m.col[2] = make_float3(0, 0, x);
	return m;
}

__host__ __device__
inline matrix3 transpose(const matrix3 & m)
{
	matrix3 out;
	out.col[0].x = m.col[0].x; out.col[1].x = m.col[0].y; out.col[2].x = m.col[0].z; 
	out.col[0].y = m.col[1].x; out.col[1].y = m.col[1].y; out.col[2].y = m.col[1].z; 
	out.col[0].z = m.col[2].x; out.col[1].z = m.col[2].y; out.col[2].z = m.col[2].z; 
	return out;
}

__host__ __device__
inline float3 operator*(const matrix3 & m, const float3 & a)
{
	float3 out;
	out = m.col[0] * a;
	out += m.col[1] * a;
	out += m.col[2] * a;
	return out;
}

/*****************************************
Quaternion
/*****************************************/

typedef float4 quaternion;

__device__
inline quaternion mul(quaternion a, quaternion b)
{
	quaternion ans;
	ans = cross3(a, b);
	ans = make_float4(ans.x + a.w*b.x + b.w*a.x + b.w*a.y, ans.y + a.w*b.y + b.w*a.z, ans.z + a.w*b.z, ans.w + a.w*b.w + b.w*a.w);
	//        ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w*b.w - dot3(a, b);
	return ans;
}

__host__ __device__
inline quaternion inverse(const quaternion q)
{
	return make_float4(-q.x, -q.y, -q.z, q.w);
}

__device__
inline float4 rotate(const quaternion q, const float4 vec)
{
	quaternion qInv = inverse(q);
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = mul(mul(q, vcpy), qInv);
	return out;
}

__host__ __device__
inline matrix3 extract_rotation_matrix(quaternion quat)
{
	float4 quat2 = make_float4(quat.x*quat.x, quat.y*quat.y, quat.z*quat.z, 0.f);
	matrix3 out;

	out.col[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
	out.col[1].x = 2 * quat.x*quat.y - 2 * quat.w*quat.z;
	out.col[2].x = 2 * quat.x*quat.z + 2 * quat.w*quat.y;

	out.col[0].y = 2 * quat.x*quat.y + 2 * quat.w*quat.z;
	out.col[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
	out.col[2].y = 2 * quat.y*quat.z - 2 * quat.w*quat.x;

	out.col[0].z = 2 * quat.x*quat.z - 2 * quat.w*quat.y;
	out.col[1].z = 2 * quat.y*quat.z + 2 * quat.w*quat.x;
	out.col[2].z = 1 - 2 * quat2.x - 2 * quat2.y;

	return out;
}
#endif  // UNIFIED_MATH_CUDA_H
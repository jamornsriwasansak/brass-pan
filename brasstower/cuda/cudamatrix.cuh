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

inline __device__ float4 getNormalizedVec(const float4 & v)
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
Matrix3
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
inline matrix3 operator+(const matrix3 & a, const matrix3 & b)
{
	matrix3 out;
	out.col[0] = a.col[0] + b.col[0];
	out.col[1] = a.col[1] + b.col[1];
	out.col[2] = a.col[2] + b.col[2];
	return out;
}

__host__ __device__
inline float3 operator*(const matrix3 & m, const float3 & a)
{
	float3 out;
	out = m.col[0] * a.x;
	out += m.col[1] * a.y;
	out += m.col[2] * a.z;
	return out;
}

/*****************************************
Matrix3
/*****************************************/

typedef struct
{
	float4 col[4];
} matrix4;

/*****************************************
Quaternion
/*****************************************/

typedef float4 quaternion;

__device__
inline quaternion mul(quaternion a, quaternion b)
{
	quaternion ans;
	ans.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;  // i
	ans.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;  // j
	ans.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;  // k
	ans.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;  // 1
	return ans;
}

__host__ __device__
inline quaternion angleAxis(const float3 & direction, const float angle)
{
	quaternion result;
	float sinHalfAngle = sin(angle * 0.5f);
	float cosHalfAngle = cos(angle * 0.5f);

	result.x = direction.x * sinHalfAngle;
	result.y = direction.y * sinHalfAngle;
	result.z = direction.z * sinHalfAngle;
	result.w = cosHalfAngle;
	return result;
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
inline matrix3 extract_rotation_matrix3(quaternion quat)
{
	float4 quat2 = quat * quat;
	matrix3 out;

	out.col[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
	out.col[0].y = 2 * quat.x*quat.y + 2 * quat.w*quat.z;
	out.col[0].z = 2 * quat.x*quat.z - 2 * quat.w*quat.y;

	out.col[1].x = 2 * quat.x*quat.y - 2 * quat.w*quat.z;
	out.col[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
	out.col[1].z = 2 * quat.y*quat.z + 2 * quat.w*quat.x;

	out.col[2].x = 2 * quat.x*quat.z + 2 * quat.w*quat.y;
	out.col[2].y = 2 * quat.y*quat.z - 2 * quat.w*quat.x;
	out.col[2].z = 1 - 2 * quat2.x - 2 * quat2.y;

	return out;
}

__host__ __device__
inline matrix4 extract_rotation_matrix4(quaternion quat)
{
	float4 quat2 = quat * quat;
	matrix4 out;

	out.col[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
	out.col[0].y = 2 * quat.x*quat.y + 2 * quat.w*quat.z;
	out.col[0].z = 2 * quat.x*quat.z - 2 * quat.w*quat.y;
	out.col[0].w = 0;

	out.col[1].x = 2 * quat.x*quat.y - 2 * quat.w*quat.z;
	out.col[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
	out.col[1].z = 2 * quat.y*quat.z + 2 * quat.w*quat.x;
	out.col[1].w = 0;

	out.col[2].x = 2 * quat.x*quat.z + 2 * quat.w*quat.y;
	out.col[2].y = 2 * quat.y*quat.z - 2 * quat.w*quat.x;
	out.col[2].z = 1 - 2 * quat2.x - 2 * quat2.y;
	out.col[2].w = 0;

	out.col[3].x = 0;
	out.col[3].y = 0;
	out.col[3].z = 0;
	out.col[3].w = 1;

	return out;
}
#endif  // UNIFIED_MATH_CUDA_H
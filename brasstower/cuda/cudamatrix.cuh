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
inline float4 cross3(float4 a, float4 b)
{
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0f);
}

__device__ float4 getNormalizedVec(const float4 v)
{
	float invLen = 1.0f / sqrtf(dot(v, v));
	return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3(float4 a, float4 b)
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
	float4 row[3];
}matrix3x4;

__host__ __device__
inline void set_zero(matrix3x4& m)
{
	m.row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	m.row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__host__ __device__
inline matrix3x4 make_matrix3x4()
{
	matrix3x4 m;
	m.row[0] = make_float4(1, 0, 0, 0);
	m.row[1] = make_float4(0, 1, 0, 0);
	m.row[2] = make_float4(0, 0, 1, 0);
	return m;
}

__host__ __device__
inline matrix3x4 make_matrix3x4(float x)
{
	matrix3x4 m;
	m.row[0] = make_float4(x, x, x, x);
	m.row[1] = make_float4(x, x, x, x);
	m.row[2] = make_float4(x, x, x, x);
	return m;
}

__host__ __device__
inline void set_identity(matrix3x4 * m)
{
	m->row[0] = make_float4(1, 0, 0, 0);
	m->row[1] = make_float4(0, 1, 0, 0);
	m->row[2] = make_float4(0, 0, 1, 0);
}

__host__ __device__
inline matrix3x4 transpose(const matrix3x4 & m)
{
	matrix3x4 out;
	out.row[0] = make_float4(m.row[0].x, m.row[1].x, m.row[2].x, 0.f);
	out.row[1] = make_float4(m.row[0].y, m.row[1].y, m.row[2].y, 0.f);
	out.row[2] = make_float4(m.row[0].z, m.row[1].z, m.row[2].z, 0.f);
	return out;
}

__device__
inline matrix3x4 mul(matrix3x4& a, matrix3x4& b)
{
	matrix3x4 transB = transpose(b);
	matrix3x4 ans;
	//        why this doesn't run when 0ing in the for{}
	a.row[0].w = 0.f;
	a.row[1].w = 0.f;
	a.row[2].w = 0.f;
	for (int i = 0; i<3; i++)
	{
		//        a.m_row[i].w = 0.f;
		ans.row[i].x = dot3(a.row[i], transB.row[0]);
		ans.row[i].y = dot3(a.row[i], transB.row[1]);
		ans.row[i].z = dot3(a.row[i], transB.row[2]);
		ans.row[i].w = 0.f;
	}
	return ans;
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
inline matrix3x4 extract_rotation_matrix(quaternion quat)
{
	float4 quat2 = make_float4(quat.x*quat.x, quat.y*quat.y, quat.z*quat.z, 0.f);
	matrix3x4 out;

	out.row[0].x = 1 - 2 * quat2.y - 2 * quat2.z;
	out.row[0].y = 2 * quat.x*quat.y - 2 * quat.w*quat.z;
	out.row[0].z = 2 * quat.x*quat.z + 2 * quat.w*quat.y;
	out.row[0].w = 0.f;

	out.row[1].x = 2 * quat.x*quat.y + 2 * quat.w*quat.z;
	out.row[1].y = 1 - 2 * quat2.x - 2 * quat2.z;
	out.row[1].z = 2 * quat.y*quat.z - 2 * quat.w*quat.x;
	out.row[1].w = 0.f;

	out.row[2].x = 2 * quat.x*quat.z - 2 * quat.w*quat.y;
	out.row[2].y = 2 * quat.y*quat.z + 2 * quat.w*quat.x;
	out.row[2].z = 1 - 2 * quat2.x - 2 * quat2.y;
	out.row[2].w = 0.f;

	return out;
}

#endif  // UNIFIED_MATH_CUDA_H
#pragma once

#include "glm/glm.hpp"
#include <cuda_runtime.h>

float2 make_float2(const glm::vec2 & v)
{
	return make_float2(v.x, v.y);
}

float3 make_float3(const glm::vec3 & v)
{
	return make_float3(v.x, v.y, v.z);
}

float4 make_float4(const glm::vec4 & v)
{
	return make_float4(v.x, v.y, v.z, v.w);
}

int2 make_int2(const glm::ivec2 & v)
{
	return make_int2(v.x, v.y);
}

int3 make_int3(const glm::ivec3 & v)
{
	return make_int3(v.x, v.y, v.z);
}

int4 make_int4(const glm::ivec4 & v)
{
	return make_int4(v.x, v.y, v.z, v.w);
}

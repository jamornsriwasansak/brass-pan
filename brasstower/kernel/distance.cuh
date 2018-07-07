#include "kernel/cubrasstower.cuh"

__global__ void
distanceConstraints(float3 * deltaX,
					const float3 * __restrict__ positions,
					const float * __restrict__ invMasses,
					const int2 * __restrict__ distancePairs,
					const float2 * __restrict__ distanceParams,
				    const int * __restrict__ newIds,
					const int numPairs)
{
	int launchIndex = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (launchIndex >= numPairs) { return; }

	int2 originalPair = distancePairs[launchIndex];
	float2 distanceParam = distanceParams[launchIndex];
	float distance = distanceParam.x;
	float stiffness = distanceParam.y;

	int id1 = newIds[originalPair.x];
	int id2 = newIds[originalPair.y];

	float3 x1 = positions[id1];
	float3 x2 = positions[id2];
	float3 x12 = x1 - x2;
	float dist12 = length(x12);
	if (dist12 <= 1e-5f) { return; }

	float3 n = x12 / dist12;

	float w1 = invMasses[id1];
	float w2 = invMasses[id2];
	float sumW12 = w1 + w2;

	float3 k = n * ((dist12 - distance) / sumW12) * stiffness;
	float3 deltaX1 = -w1 * k;
	float3 deltaX2 = w2 * k;

	atomicAdd(deltaX, id1, -w1 * k);
	atomicAdd(deltaX, id2, w2 * k);
}
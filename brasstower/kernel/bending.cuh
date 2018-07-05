#include "kernel/cubrasstower.cuh"

__global__ void
bendingTripletsConstraints(float3 * deltaX,
						   const float3 * __restrict__ positions,
						   const float * __restrict__ invMasses,
						   const int3 * __restrict__ triplets,
						   const int * __restrict__ newIds,
						   const int numPairs)
{
	/*int launchIndex = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (launchIndex >= numPairs) { return; }

	int3 triplet = triplets[launchIndex];

	int id1 = newIds[triplet.x];
	int id2 = newIds[triplet.y];
	int id3 = newIds[triplet.z];

	float3 x1 = positions[id1];
	float3 x2 = positions[id2];
	float3 x3 = positions[id3];

	float3 v1 = x1 - x2;
	float3 v3 = x3 - x2;
	float dist1 = length(v1);
	float dist3 = length(v3);
	float3 n1 = v1 / dist1;
	float3 n3 = v3 / dist3;

	float d = dot(n1, n3);
	float k = 1.0f / sqrtf(1.0f - d * d);
	
	float3 gradient1 = (d * v1 - dist1 * n3) * k;
	float3 gradient3 = (d * v3 - dist3 * n1) * k;

	float constraint = acosf(d) - MATH_PI;

	float w1 = invMasses[id1];
	float w3 = invMasses[id3];

	float s = constraint / (w1 * dot(gradient1, gradient1) + w3 * dot(gradient3, gradient3));

	atomicAdd(deltaX, id1, -s * w1 * gradient1);
	atomicAdd(deltaX, id3, -s * w3 * gradient3);*/
}
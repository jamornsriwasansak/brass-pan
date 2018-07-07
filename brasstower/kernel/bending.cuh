#include "kernel/cubrasstower.cuh"

// buggy doesn't work
__global__ void
bendingConstraints(float3 * deltaX,
				   const float3 * __restrict__ positions,
				   const float * __restrict__ invMasses,
				   const int4 * __restrict__ bendings,
				   const int * __restrict__ newIds,
				   const int numBendings)
{
	int launchIndex = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (launchIndex >= numBendings) { return; }

	int4 ids = bendings[0];
	int id1 = newIds[ids.x];
	int id2 = newIds[ids.y];
	int id3 = newIds[ids.z];
	int id4 = newIds[ids.w];

	float3 P1 = positions[id1];
	float3 P2 = positions[id2] - P1;
	float3 P3 = positions[id3] - P1;
	float3 P4 = positions[id4] - P1;

	float3 cP2P3 = cross(P2, P3);
	float3 cP2P4 = cross(P2, P4);

	float lengthCP2P3 = length(cP2P3);
	if (lengthCP2P3 <= 1e-5f) { return; }

	float lengthCP2P4 = length(cP2P4);
	if (lengthCP2P4 <= 1e-5f) { return; }

	float invLengthCP2P3 = 1.0f / lengthCP2P3;
	float invLengthCP2P4 = 1.0f / length(cP2P4);

	float3 N1 = cP2P3 * invLengthCP2P3;
	float3 N2 = cP2P4 * invLengthCP2P4;

	float d = dot(N1, N2);
	d = min(max(d, -1.0f), 1.0f);

	float3 cP2N1 = cross(P2, N1);
	float3 cP2N2 = cross(P2, N2);
	float3 cP3N1 = cross(P3, N1);
	float3 cP3N2 = cross(P3, N2);
	float3 cP4N1 = cross(P4, N1);
	float3 cP4N2 = cross(P4, N2);

	float3 Q3 = (cP2N2 - cP2N1 * d) / lengthCP2P3;
	float3 Q4 = (cP2N1 - cP2N2 * d) / lengthCP2P4;
	float3 Q2 = -(cP3N2 - cP3N1 * d) / lengthCP2P3 - (cP4N1 - cP4N2 * d) / lengthCP2P4;
	float3 Q1 = -Q2 - Q3 - Q4;

	float w1 = invMasses[id1];
	float w2 = invMasses[id2];
	float w3 = invMasses[id3];
	float w4 = invMasses[id4];

	float denominator = w1 * length2(Q1) + w2 * length2(Q2) + w3 * length2(Q3) + w4 * length2(Q4);
	float numerator = sqrtf(1.0f - d * d) * (acos(d) - acos(-1.0f));
	if (denominator <= 1e-5f) return;
	float scaling = numerator / denominator;

	const float kBend = 0.05f;

	float3 deltaX1 = -w1 * Q1 * scaling * kBend;
	float3 deltaX2 = -w2 * Q2 * scaling * kBend;
	float3 deltaX3 = -w3 * Q3 * scaling * kBend;
	float3 deltaX4 = -w4 * Q4 * scaling * kBend;

	atomicAdd(deltaX, id1, deltaX1);
	atomicAdd(deltaX, id2, deltaX2);
	atomicAdd(deltaX, id3, deltaX3);
	atomicAdd(deltaX, id4, deltaX4);
}
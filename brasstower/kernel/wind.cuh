#include "kernel/cubrasstower.cuh"

__global__ void
applyWindForce(float3 * deltaX,
			   const float3 * __restrict__ positions,
			   const float3 * __restrict__ velocities,
			   const float * __restrict__ masses,
			   const int * __restrict__ newIds,
			   const int3 * __restrict__ faceIds,
			   const int numFaces,
			   const float deltaTime)
{
	int launchIndex = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (launchIndex >= numFaces) { return; }

	int3 ids = faceIds[launchIndex];
	int id1 = newIds[ids.x];
	int id2 = newIds[ids.y];
	int id3 = newIds[ids.z];

	float3 p1 = positions[id1];
	float3 p2 = positions[id2];
	float3 p3 = positions[id3];

	float3 a = p2 - p1;
	float3 b = p3 - p1;

	float3 v = (velocities[id1] + velocities[id2] + velocities[id3]) / 3.f;
	float lenV = length(v);
	if (lenV <= 1e-5f) return;
	float3 vhat = v / lenV;

	float3 faceNormal = cross(a, b);
	float lenFaceNormal = length(faceNormal);
	if (lenFaceNormal <= 1e-5f) return;
	faceNormal /= lenFaceNormal;
	faceNormal = (dot(faceNormal, v) > 0) ? faceNormal : -faceNormal;
	float area = 0.5f * lenFaceNormal;

	float3 dragForce = 10.0f * (lenV * lenV) * area * dot(faceNormal, vhat) * (-vhat);
	
	float3 uhat = cross(cross(faceNormal, vhat), vhat);
	float3 liftForce = 100.0f * (lenV * lenV) * area * dot(faceNormal, vhat) * uhat;

	float3 force = dragForce + liftForce;
	float3 move = force * deltaTime * deltaTime;

	atomicAdd(deltaX, id1, move);
	atomicAdd(deltaX, id2, move);
	atomicAdd(deltaX, id3, move);
}
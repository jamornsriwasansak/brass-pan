#include "kernel/cubrasstower.cuh"

__global__ void
planeStabilize(float3 * __restrict__ positions,
			   float3 * __restrict__ newPositions,
			   const int numParticles,
			   const float3 planeOrigin,
			   const float3 planeNormal,
			   const float radius)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	float3 origin2position = planeOrigin - positions[i];
	float distance = dot(origin2position, planeNormal) + radius;
	if (distance <= 0) { return; }

	positions[i] += distance * planeNormal;
	newPositions[i] += distance * planeNormal;
}

__global__ void
particlePlaneCollisionConstraint(float3 * __restrict__ newPositions,
								 float3 * __restrict__ positions,
								 const int numParticles,
								 const float3 planeOrigin,
								 const float3 planeNormal,
								 const float radius)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	float3 origin2position = planeOrigin - newPositions[i];
	float distance = dot(origin2position, planeNormal) + radius;
	if (distance <= 0) { return; }

	float3 position = positions[i];

	float3 diff = newPositions[i] - position;
	float diffNormal = dot(diff, planeNormal);
	float3 diffTangent = diff - diffNormal * planeNormal;
	float diffTangentLength = length(diffTangent);
	float diffLength = length(diff);

	float3 resolvedPosition = distance * planeNormal + newPositions[i];
	float3 deltaX = resolvedPosition - position;
	float3 tangentialDeltaX = deltaX - dot(deltaX, planeNormal) * planeNormal;

	//positions[i] += (2.0f * diffNormal + distance) * planeNormal * ENERGY_LOST_RATIO;

	// Adaptation of Unified Particle Physics for Real-Time Applications, eq.24 
	if (diffTangentLength < FRICTION_STATIC * diffNormal)
	{
		newPositions[i] = resolvedPosition - tangentialDeltaX;
	}
	else
	{
		newPositions[i] = resolvedPosition - tangentialDeltaX * min(FRICTION_DYNAMICS * -diffNormal / diffTangentLength, 1.0f);
	}
}

#include "kernel/cubrasstower.cuh"

__global__ void
immovableConstraints(float3 * __restrict__ newPositions,
					 const float3 * __restrict__ positions,
					 const int * __restrict__ immovableIds,
					 const int * __restrict__ newIds,
					 const int numImmovables)
{
	int launchIndex = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (launchIndex >= numImmovables) { return; }

	int id = newIds[immovableIds[launchIndex]];
	newPositions[id] = positions[id];
}
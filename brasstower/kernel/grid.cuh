#include "kernel/cubrasstower.cuh"

// GRID //
// from White Paper "Particles" by SIMON GREEN 

__device__
const int3 calcGridPos(const float3 position,
					   const  float3 origin,
					   const  float3 cellSize)
{
	return make_int3((position - origin) / cellSize);
}

__device__
const int positiveMod(const int dividend, const int divisor)
{
	return (dividend % divisor + divisor) % divisor;
}

__device__
const int calcGridAddress(const int3 gridPos, const int3 gridSize)
{
	return (positiveMod(gridPos.z, gridSize.z) * gridSize.y * gridSize.x)
		+ (positiveMod(gridPos.y, gridSize.y) * gridSize.x)
		+ (positiveMod(gridPos.x, gridSize.x));
}

__global__ void
updateGridId(int * __restrict__ gridIds,
			 int * __restrict__ particleIds,
			 const float3 * __restrict__ positions,
			 const float3 cellOrigin,
			 const float3 cellSize,
			 const int3 gridSize,
			 const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int3 gridPos = calcGridPos(positions[i], cellOrigin, cellSize);
	const int gridId = calcGridAddress(gridPos, gridSize);

	gridIds[i] = gridId;
	particleIds[i] = i;
}

__global__ void
findStartId(int * cellStart,
			const int * sortedGridIds,
			const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int cell = sortedGridIds[i];

	if (i > 0)
	{
		if (cell != sortedGridIds[i - 1])
			cellStart[cell] = i;
	}
	else
	{
		cellStart[cell] = i;
	}
}

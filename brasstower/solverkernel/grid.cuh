#include "solverkernel/cubrasstower.cuh"

// GRID //
// from White Paper "Particles" by SIMON GREEN 

__device__ int3 calcGridPos(float3 position,
							float3 origin,
							float3 cellSize)
{
	return make_int3((position - origin) / cellSize);
}

__device__ int positiveMod(int dividend, int divisor)
{
	return (dividend % divisor + divisor) % divisor;
}

__device__ int calcGridAddress(int3 gridPos, int3 gridSize)
{
	gridPos = make_int3(positiveMod(gridPos.x, gridSize.x), positiveMod(gridPos.y, gridSize.y), positiveMod(gridPos.z, gridSize.z));
	return (gridPos.z * gridSize.y * gridSize.x) + (gridPos.y * gridSize.x) + gridPos.x;
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
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	int3 gridPos = calcGridPos(positions[i], cellOrigin, cellSize);
	int gridId = calcGridAddress(gridPos, gridSize);

	gridIds[i] = gridId;
	particleIds[i] = i;
}

__global__ void
findStartId(int * cellStart,
			const int * sortedGridIds,
			const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	int cell = sortedGridIds[i];

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

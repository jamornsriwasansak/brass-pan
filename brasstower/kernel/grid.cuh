#include "kernel/cubrasstower.cuh"

// GRID //
// from White Paper "Particles" by SIMON GREEN 

__device__ __constant__ uint GridDim;
__device__ __constant__ float3 GridCellSize;

__device__
const int3 calcGridPos(const float3 position)
{
	return make_int3((position + 50000) / GridCellSize);
}

__device__
const int calcGridAddress(const int x, const int y, const int z)
{
	const uint cx = x & (GridDim - 1);
	const uint cy = y & (GridDim - 1);
	const uint cz = z & (GridDim - 1);
	return cx * (GridDim * GridDim) + cy * GridDim + cz;
}

__device__
const int calcGridAddress(const int3 gridPos)
{
	return calcGridAddress(gridPos.x, gridPos.y, gridPos.z);
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

	const int3 gridPos = calcGridPos(positions[i]);
	const int gridId = calcGridAddress(gridPos);

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

__global__ void
findStartEndId(int * __restrict__ cellStart,
			   int * __restrict__ cellEnd,
			   const int * __restrict__ sortedGridIds,
			   const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int u = sortedGridIds[i];

	if (i > 0)
	{
		int v = sortedGridIds[i - 1];
		if (u != v)
		{
			cellStart[u] = i;
			cellEnd[v] = i;
		}
	}
	else
	{
		cellStart[u] = i;
	}

	if (i == numParticles - 1)
	{
		cellEnd[u] = numParticles;
	}
}

// this is much faster than using cudaMemcpy or lanuching a new kernel
// when # num particles <<< # num grid cells
__global__ void
resetStartEndId(int * __restrict__ cellStart,
				int * __restrict__ cellEnd,
				const int * __restrict__ sortedGridIds,
				const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int u = sortedGridIds[i];

	if (i > 0)
	{
		int v = sortedGridIds[i - 1];
		if (u != v)
		{
			cellStart[u] = -1;
			cellEnd[v] = -1;
		}
	}
	else
	{
		cellStart[u] = -1;
	}

	if (i == numParticles - 1)
	{
		cellEnd[u] = -1;
	}
}

__global__ void
reorderParticlesData(float3 * __restrict__ sortedNewPositions,
					 float3 * __restrict__ sortedPositions,
					 float3 * __restrict__ sortedVelocities,
					 float * __restrict__ sortedMasses,
					 int * __restrict__ sortedPhases,
					 int * __restrict__ sortedOriginalIds,
					 const float3 * __restrict__ newPositions,
					 const float3 * __restrict__ positions,
					 const float3 * __restrict__ velocities,
					 const float * __restrict__ masses,
					 const int * __restrict__ phases,
					 const int * __restrict__ originalIds,
					 const int * __restrict__ indices,
					 const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	int oldIndex = indices[i];

	sortedNewPositions[i] = newPositions[oldIndex];
	sortedPositions[i] = positions[oldIndex];
	sortedVelocities[i] = velocities[oldIndex];
	sortedMasses[i] = masses[oldIndex];
	sortedPhases[i] = phases[oldIndex];
	sortedOriginalIds[i] = originalIds[oldIndex];
}
#include "solverkernel/cubrasstower.cuh"

// PROJECT CONSTRAINTS //

__global__ void
particleParticleCollisionConstraint(float3 * __restrict__ newPositionsNext,
									const float3 * __restrict__ newPositionsPrev,
									const float3 * __restrict__ positions,
									const float * __restrict__ invMasses,
									const int* __restrict__ phases,
									const int* __restrict__ sortedCellId,
									const int* __restrict__ sortedParticleId,
									const int* __restrict__ cellStart,
									const float3 cellOrigin,
									const float3 cellSize,
									const int3 gridSize,
									const int numParticles,
									const float radius)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	float3 xi = positions[i];
	float3 xiPrev = newPositionsPrev[i];
	float3 sumDeltaXi = make_float3(0.f);
	float3 sumFriction = make_float3(0.f);
	float invMass = invMasses[i];

	int3 centerGridPos = calcGridPos(newPositionsPrev[i], cellOrigin, cellSize);
	int3 start = centerGridPos - 1;
	int3 end = centerGridPos + 1;

	int constraintCount = 0;
	for (int z = start.z; z <= end.z; z++)
		for (int y = start.y; y <= end.y; y++)
			for (int x = start.x; x <= end.x; x++)
			{
				int3 gridPos = make_int3(x, y, z);
				int gridAddress = calcGridAddress(gridPos, gridSize);
				int bucketStart = cellStart[gridAddress];
				if (bucketStart == -1) { continue; }

				for (int k = 0; k + bucketStart < numParticles; k++)
				{
					int gridAddress2 = sortedCellId[bucketStart + k];
					int particleId2 = sortedParticleId[bucketStart + k];
					if (gridAddress2 != gridAddress) { break; }

					if (i != particleId2 && phases[i] != phases[particleId2])
					{
						float3 xjPrev = newPositionsPrev[particleId2];
						float3 diff = xiPrev - xjPrev;
						float dist2 = length2(diff);
						if (dist2 < radius * radius * 4.0f)
						{
							float dist = sqrtf(dist2);
							float invMass2 = invMasses[particleId2];
							float weight1 = invMass / (invMass + invMass2);

							float3 projectDir = diff * (2.0f * radius / dist - 1.0f);
							float3 deltaXi = weight1 * projectDir;
							float3 xiStar = deltaXi + xiPrev;
							sumDeltaXi += deltaXi;

							float deltaXiLength2 = length2(deltaXi);

							if (deltaXiLength2 > radius * radius * 0.001f * 0.001f)
							{
								constraintCount += 1;
								float weight2 = invMass2 / (invMass + invMass2);
								float3 xj = positions[particleId2];
								float3 deltaXj = -weight2 * projectDir;
								float3 xjStar = deltaXj + xjPrev;
								float3 term1 = (xiStar - xi) - (xjStar - xj);
								float3 n = diff / dist;
								float3 tangentialDeltaX = term1 - dot(term1, n) * n;

								float tangentialDeltaXLength2 = length2(tangentialDeltaX);

								if (tangentialDeltaXLength2 <= (FRICTION_STATIC * FRICTION_STATIC) * deltaXiLength2)
								{
									sumFriction -= weight1 * tangentialDeltaX;
								}
								else
								{
									sumFriction -= weight1 * tangentialDeltaX * min(FRICTION_DYNAMICS * sqrtf(deltaXiLength2 / tangentialDeltaXLength2), 1.0f);
								}
							}
						}
					}
				}
			}

	newPositionsNext[i] = (constraintCount == 0) ?
		xiPrev + sumDeltaXi :
		xiPrev + sumDeltaXi + sumFriction / constraintCount; // averaging constraints is very important here. otherwise the solver will explode.
}

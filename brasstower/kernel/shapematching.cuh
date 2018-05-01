#include "kernel/cubrasstower.cuh"

// Meshless Deformations Based on Shape Matching
// by Muller et al.

// one block per one shape

__global__ void
shapeMatchingAlphaOne(quaternion * __restrict__ rotations,
					  float3 * __restrict__ CMs,
					  float3 * __restrict__ positions,
					  const float3 * __restrict__ initialPositions,
					  const int2 * __restrict__ rigidBodyParticleIdRange)
{
	// typename stuffs
	typedef cub::BlockReduce<float3, NUM_MAX_PARTICLE_PER_RIGID_BODY> BlockReduceFloat3;
	__shared__ typename BlockReduceFloat3::TempStorage TempStorageFloat3;

	// init rigidBodyId, particleRange and numParticles
	int rigidBodyId = blockIdx.x;
	int2 particleRange = rigidBodyParticleIdRange[rigidBodyId];

	int numParticles = particleRange.y - particleRange.x;
	int particleId = particleRange.x + threadIdx.x;

	float3 position = positions[particleId];

	__shared__ float3 CM;
	__shared__ matrix3 extractedR;

	if (threadIdx.x < numParticles)
	{
		// find center of mass using block reduce
		float3 sumCM = BlockReduceFloat3(TempStorageFloat3).Sum(position);

		if (threadIdx.x == 0)
		{
			CM = sumCM / (float)numParticles;
			CMs[rigidBodyId] = CM;
		}
	}
	__syncthreads();

	float3 qi;
	if (threadIdx.x < numParticles)
	{
		// compute matrix Apq
		float3 initialPosition = initialPositions[particleId];
		float3 pi = position - CM;
		qi = initialPosition;// do not needed to subtract from initialCM since initialCM = float3(0);

							 // Matrix Ai refers to p * q
		float3 AiCol0 = pi * qi.x;
		float3 AiCol1 = pi * qi.y;
		float3 AiCol2 = pi * qi.z;

		// Matrix A refers to Apq
		float3 ACol0 = BlockReduceFloat3(TempStorageFloat3).Sum(AiCol0);
		float3 ACol1 = BlockReduceFloat3(TempStorageFloat3).Sum(AiCol1);
		float3 ACol2 = BlockReduceFloat3(TempStorageFloat3).Sum(AiCol2);

		if (threadIdx.x == 0)
		{
			// extract rotation matrix using method 
			// using A Robust Method to Extract the Rotational Part of Deformations
			// by Muller et al.

			quaternion q = rotations[rigidBodyId];
			matrix3 R = extract_rotation_matrix3(q);
			for (int i = 0; i < 20; i++)
			{
				matrix3 R = extract_rotation_matrix3(q);
				float3 omegaNumerator = (cross(R.col[0], ACol0) +
										 cross(R.col[1], ACol1) +
										 cross(R.col[2], ACol2));
				float omegaDenominator = 1.0f / fabs(dot(R.col[0], ACol0) +
													 dot(R.col[1], ACol1) +
													 dot(R.col[2], ACol2)) + 1e-9f;
				float3 omega = omegaNumerator * omegaDenominator;
				float w2 = length2(omega);
				if (w2 <= 1e-9f) { break; }
				float w = sqrtf(w2);

				q = mul(angleAxis(omega / w, w), q);
				q = normalize(q);
			}
			extractedR = extract_rotation_matrix3(q);
			rotations[rigidBodyId] = q;
		}
	}
	__syncthreads();

	if (threadIdx.x < numParticles)
	{
		float3 newPosition = extractedR * qi + CM;
		positions[particleId] = newPosition;
	}
}

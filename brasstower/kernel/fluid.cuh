#include "kernel/cubrasstower.cuh"

__device__ __constant__ float KernelConst1;
__device__ __constant__ float KernelConst2;
__device__ __constant__ float KernelConst3;
__device__ __constant__ float KernelConst4;
__device__ __constant__ float KernelRadius;
__device__ __constant__ float KernelSquaredRadius;
__device__ __constant__ float KernelHalfRadius;

void SetKernelRadius(float h)
{
	float const1 = 315.f / 64.f / 3.141592f / powf(h, 9.0f);
	checkCudaErrors(cudaMemcpyToSymbol(KernelConst1, &const1, sizeof(float)));
	float const2 = -45.f / 3.141592f / powf(h, 6.f);
	checkCudaErrors(cudaMemcpyToSymbol(KernelConst2, &const2, sizeof(float)));
	float const3 = 32.0f / 3.141592f / powf(h, 9.f);
	checkCudaErrors(cudaMemcpyToSymbol(KernelConst3, &const3, sizeof(float)));
	float const4 = powf(h, 6.0f) / 64.0f;
	checkCudaErrors(cudaMemcpyToSymbol(KernelConst4, &const4, sizeof(float)));
	float const5 = h * h;
	checkCudaErrors(cudaMemcpyToSymbol(KernelSquaredRadius, &const5, sizeof(float)));
	float const6 = h * 0.5f;
	checkCudaErrors(cudaMemcpyToSymbol(KernelHalfRadius, &const6, sizeof(float)));
	float const7 = h;
	checkCudaErrors(cudaMemcpyToSymbol(KernelRadius, &const7, sizeof(float)));
}

//#define USE_KERNEL_DISTANCE_CHECK

__device__
const float poly6Kernel(const float r2)
{
	/// TODO:: precompute these
#ifdef USE_KERNEL_DISTANCE_CHECK
	if (r2 <= KernelSquaredRadius)
#endif
	{
		const float temp = KernelSquaredRadius - r2;
		return KernelConst1 * temp * temp * temp;
	}
	return 0.f;
}

__device__
const float3 gradientSpikyKernel(const float3 v, const float r2)
{
#ifdef USE_KERNEL_DISTANCE_CHECK
	if (r2 <= KernelSquaredRadius && r2 > 0.f)
#endif
	{
		const float r = sqrtf(r2);
		const float temp = KernelRadius - r;
		return KernelConst2 * temp * temp * v / r;
	}
	return make_float3(0.f);
}

__device__
const float akinciSplineC(const float r) // akinci used 2*r instead of r
{
#ifdef USE_KERNEL_DISTANCE_CHECK
	if (r < KernelRadius && r > 0)
#endif
	{
		const float temp = (KernelRadius - r) * r;
		const float temp3 = temp * temp * temp;
		if (r >= KernelHalfRadius)
			return KernelConst3 * temp3;
		else
			return 2.0f * KernelConst3 * temp3 - KernelConst4;
	}
	return 0.f;
}

#define USE_PRECOMPUTED_FLUID

__global__ void
fluidLambda(float * __restrict__ lambdas,
			float * __restrict__ densities,
			const float3 * __restrict__ newPositionsPrev,
			const float * __restrict__ masses,
			const int * __restrict__ phases,
			const float restDensity,
			const float solidDensity,
			const float epsilon,
			const int * __restrict__ cellStart,
			const int * __restrict__ cellEnd,
			const int numParticles,
			const bool useAkinciCohesionTension)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles || phases[i] > 0) { return; }

	const float3 pi = newPositionsPrev[i];

	// compute density and gradient of constraint
	float density = 0.f;
	float3 gradientI = make_float3(0.f);
#ifndef USE_PRECOMPUTED_FLUID
	float sumGradient2 = 0.f;
#endif

	const int3 centerGridPos = calcGridPos(newPositionsPrev[i]);
	const int3 searchStart = centerGridPos - 1;
	const int3 searchEnd = centerGridPos + 1;

	for (int z = searchStart.z; z <= searchEnd.z; z++)
		for (int y = searchStart.y; y <= searchEnd.y; y++)
			for (int x = searchStart.x; x <= searchEnd.x; x++)
			{
				const int gridAddress = calcGridAddress(x, y, z);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i <= j)
					{
						const float3 pj = newPositionsPrev[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(pi - pj);
						if (dist2 <= KernelSquaredRadius)
						{
							// density
							const float massj = masses[j];
							const float liquidDensity = massj * poly6Kernel(dist2);
							density += (phases[j] < 0) ? liquidDensity : solidDensity * liquidDensity;

						#ifndef USE_PRECOMPUTED_FLUID
							// gradient for lambda
							const float3 gradient = -massj * gradientSpikyKernel(diff, dist2) / restDensity;
							sumGradient2 += dot(gradient, gradient);
							gradientI -= gradient;
						#endif
						}
					}
				}
			}

#ifndef USE_PRECOMPUTED_FLUID
	sumGradient2 += dot(gradientI, gradientI);
#endif


	// compute constraint
	float constraint = density / restDensity - 1.0f;
	if (useAkinciCohesionTension) { constraint = max(constraint, 0.0f); }
#ifdef USE_PRECOMPUTED_FLUID
	// 10000 ~ approximate of sumGradient * mass * epsilon
	const float lambda = -constraint / 15000.0f;
#else
	const float lambda = -constraint / (sumGradient2 + epsilon);
#endif
	lambdas[i] = lambda;
	densities[i] = density;
}

__global__ void
fluidPosition(float3 * __restrict__ deltaXs,
			  const float3 * __restrict__ newPositionsPrev,
			  const float * __restrict__ lambdas,
			  const float restDensity,
			  const float * __restrict__ masses,
			  const int * __restrict__ phases,
			  const float K,
			  const int N,
			  const int* __restrict__ cellStart,
			  const int * __restrict__ cellEnd,
			  const int numParticles,
			  const bool useAkinciCohesionTension)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles || phases[i] > 0) { return; }

	const float3 pi = newPositionsPrev[i];

	float3 sum = make_float3(0.f);
	const int3 centerGridPos = calcGridPos(newPositionsPrev[i]);
	const int3 searchStart = centerGridPos - 1;
	const int3 searchEnd = centerGridPos + 1;

	const float massi = masses[i];
	const float lambdai = lambdas[i];
	const float invRestDensity = 1.0f / restDensity;

	for (int z = searchStart.z; z <= searchEnd.z; z++)
		for (int y = searchStart.y; y <= searchEnd.y; y++)
			for (int x = searchStart.x; x <= searchEnd.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i < j && phases[j] < 0)
					{
						const float3 pj = newPositionsPrev[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius && dist2 > 0)
						{
							const float massj = masses[j];
							float sumLambda = lambdai + lambdas[j];
							if (!useAkinciCohesionTension) sumLambda += -K * powf(poly6Kernel(dist2) / poly6Kernel(powf(0.03f * KernelRadius, 2.f)), N);
							float3 delta = sumLambda * gradientSpikyKernel(diff, dist2);

						#if 1
							// clamp deltaposition to increase solver stability
							// good parameters can trade-off realism for performance
							const float clamping = 5.0f;
							const float lenMove = length(delta);
							delta = (lenMove < clamping) ? delta : delta * min(lenMove, clamping) / lenMove;
						#endif

							sum += massj * delta;
							atomicAdd(deltaXs, j, -massi * delta * invRestDensity);
						}
					}
				}
			}

	const float3 deltaPosition = sum * invRestDensity;
	atomicAdd(deltaXs, i, deltaPosition);
}

/// TODO:: optimize this by plug it in last loop of fluidPosition
__global__ void
fluidOmega(float3 * __restrict__ omegas,
		   const float3 * __restrict__ velocities,
		   const float3 * __restrict__ positions,
		   const int * __restrict__ phases,
		   const int* __restrict__ cellStart,
		   const int * __restrict__ cellEnd,
		   const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int phasei = phases[i];
	if (phasei >= 0) { return; }

	const float3 pi = positions[i];
	const float3 vi = velocities[i];

	float3 omegai = make_float3(0.f);
	const int3 centerGridPos = calcGridPos(positions[i]);
	const int3 searchStart = centerGridPos - 1;
	const int3 searchEnd = centerGridPos + 1;

	for (int z = searchStart.z; z <= searchEnd.z; z++)
		for (int y = searchStart.y; y <= searchEnd.y; y++)
			for (int x = searchStart.x; x <= searchEnd.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i != j && phasei == phases[j])
					{
						const float3 pj = positions[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius && dist2 > 0)
						{
							const float3 vj = velocities[j];
							omegai += cross(vj - vi, gradientSpikyKernel(diff, dist2));
						}
					}
				}
			}

	omegas[i] = omegai;
}

__global__ void
fluidVorticity(float3 * __restrict__ velocities,
			   const float3 * __restrict__ omegas,
			   const float3 * __restrict__ positions,
			   const float scalingFactor,
			   const int * __restrict__ phases,
			   const int* __restrict__ cellStart,
			   const int * __restrict__ cellEnd,
			   const int numParticles,
			   const float deltaTime)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int phasei = phases[i];
	if (phasei >= 0) { return; }

	const float3 omegai = omegas[i];
	const float3 pi = positions[i];
	const int3 centerGridPos = calcGridPos(positions[i]);
	const int3 searchStart = centerGridPos - 1;
	const int3 searchEnd = centerGridPos + 1;

	float3 eta = make_float3(0.f);

	for (int z = searchStart.z; z <= searchEnd.z; z++)
		for (int y = searchStart.y; y <= searchEnd.y; y++)
			for (int x = searchStart.x; x <= searchEnd.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i != j && phasei == phases[j])
					{
						const float3 pj = positions[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius && dist2 > 0)
						{
							eta += length(omegas[j]) * gradientSpikyKernel(diff, dist2);
						}
					}
				}
			}

	if (length2(eta) > 1e-3f)
	{
		const float3 normal = normalize(eta);

		/// TODO:: also have to be devided by mass
		velocities[i] += scalingFactor * cross(normal, omegai) * deltaTime;
	}
}

__global__ void
fluidXSph(float3 * __restrict__ newVelocities,
		  const float3 * __restrict__ velocities,
		  const float3 * __restrict__ positions,
		  const float c, // position-based fluid eq. 17
		  const int * __restrict__ phases,
		  const int* __restrict__ cellStart,
		  const int * __restrict__ cellEnd,
		  const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int phasei = phases[i];
	if (phasei >= 0) { return; }

	const float3 pi = positions[i];
	const int3 centerGridPos = calcGridPos(pi);
	const int3 searchStart = centerGridPos - 1;
	const int3 searchEnd = centerGridPos + 1;

	float3 vnew = make_float3(0.f);
	const float3 vi = velocities[i];
	for (int z = searchStart.z; z <= searchEnd.z; z++)
		for (int y = searchStart.y; y <= searchEnd.y; y++)
			for (int x = searchStart.x; x <= searchEnd.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i != j && phasei == phases[j])
					{
						float3 pj = positions[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius)
						{
							vnew += (velocities[j] - vi) * poly6Kernel(dist2);
						}
					}
				}
			}

	newVelocities[i] = vi + c * vnew;
}

__global__ void
fluidNormal(float3 * __restrict__ normals,
			const float3 * __restrict__ positions,
			const float * __restrict__ densities,
			const int * __restrict__ phases,
			const int* __restrict__ cellStart,
			const int * __restrict__ cellEnd,
			const int numParticles)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles || phases[i] > 0) { return; }

	const int phasei = phases[i];
	if (phasei >= 0) { return; }

	const float3 pi = positions[i];
	const int3 centerGridPos = calcGridPos(pi);
	const int3 start = centerGridPos - 1;
	const int3 end = centerGridPos + 1;

	float3 normal = make_float3(0.f);

	for (int z = start.z; z <= end.z; z++)
		for (int y = start.y; y <= end.y; y++)
			for (int x = start.x; x <= end.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				int neighbourStart = cellStart[gridAddress];
				int neighbourEnd = cellEnd[gridAddress];
				for (int j = neighbourStart; j < neighbourEnd; j++)
				{
					if (i != j && phasei == phases[j])
					{
						const float3 pj = positions[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius && dist2 > 0)
						{
							normal += 1.0f / densities[j] * gradientSpikyKernel(diff, dist2);
						}
					}
				}
			}

	normals[i] = KernelRadius * normal;
}

__global__ void
fluidAkinciTension(float3 * __restrict__ newVelocities,
				   const float3 * __restrict__ velocities,
				   const float3 * __restrict__ positions,
				   const float3 * __restrict__ normals,
				   const float * __restrict__ densities,
				   const float restDensity,
				   const int * __restrict__ phases,
				   const float surfaceTension,
				   const int* __restrict__ cellStart,
				   const int * __restrict__ cellEnd,
				   const int numParticles,
				   const float deltaTime)
{
	const int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles || phases[i] > 0) { return; }

	const int phasei = phases[i];
	if (phasei >= 0) { return; }

	const float3 pi = positions[i];
	const int3 centerGridPos = calcGridPos(pi);
	const int3 start = centerGridPos - 1;
	const int3 end = centerGridPos + 1;

	float3 fTension = make_float3(0.f);

	for (int z = start.z; z <= end.z; z++)
		for (int y = start.y; y <= end.y; y++)
			for (int x = start.x; x <= end.x; x++)
			{
				const int3 gridPos = make_int3(x, y, z);
				const int gridAddress = calcGridAddress(gridPos);
				for (int j = cellStart[gridAddress]; j < cellEnd[gridAddress]; j++)
				{
					if (i != j && phasei == phases[j])
					{
						const float3 pj = positions[j];
						const float3 diff = pi - pj;
						const float dist2 = length2(diff);
						if (dist2 <= KernelSquaredRadius && dist2 > 0.f)
						{
							const float dist = sqrt(dist2);
							const float3 fCohesion = -surfaceTension * akinciSplineC(dist) * (pi - pj) / dist;
							const float3 fCurvature = -surfaceTension * (normals[i] - normals[j]);
							const float kij = 2.0f * restDensity / (densities[i] + densities[j]);
							fTension += kij * (fCohesion + fCurvature);
						}
					}
				}
			}

	newVelocities[i] = velocities[i] + fTension * deltaTime;
}
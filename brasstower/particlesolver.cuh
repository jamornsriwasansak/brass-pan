#pragma once

#include <exception>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <conio.h>

#ifndef __INTELLISENSE__
#include <cub/cub.cuh>
#endif

#include "cuda/helper.cuh"
#include "cuda/cudamatrix.cuh"
#include "cuda/cudaglm.cuh"
#include "scene.h"

#define NUM_MAX_PARTICLE_PER_CELL 4
#define ENERGY_LOST_RATIO 0.1f;

void GetNumBlocksNumThreads(int * numBlocks, int * numThreads, int k)
{
	*numThreads = 512;
	*numBlocks = static_cast<int>(ceil((float)k / (float)(*numThreads)));
}

template <typename T>
void print(T * dev, int size)
{
	T * tmp = (T *)malloc(sizeof(T) * size);
	cudaMemcpy(tmp, dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << tmp[i];
		if (i != size - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
	free(tmp);
}

template <>
void print<float3>(float3 * dev, int size)
{
	float3 * tmp = (float3 *)malloc(sizeof(float3) * size);
	cudaMemcpy(tmp, dev, sizeof(float3) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << "(" << tmp[i].x << " " << tmp[i].y << " " << tmp[i].z << ")";
		if (i != size - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
	free(tmp);
}

template <typename T>
void printPair(T * dev, int size)
{
	T * tmp = (T *)malloc(sizeof(T) * size);
	cudaMemcpy(tmp, dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		if (tmp[i] != -1)
		{
			std::cout << i << ":" << tmp[i];
			if (i != size - 1)
				std::cout << ",";
		}
	}
	std::cout << std::endl;
	free(tmp);
}

// PARTICLE SYSTEM //

__global__ void increment(int * __restrict__ x)
{
	atomicAdd(x, 1);
}

__global__ void setDevArr_devIntPtr(int * __restrict__ devArr,
								    const int * __restrict__ value,
								    const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = *value;
}

__global__ void setDevArr_int(int * __restrict__ devArr,
						  const int value,
						  const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__global__ void setDevArr_float(float * __restrict__ devArr,
								const float value,
								const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__global__ void setDevArr_int2(int2 * __restrict__ devArr,
							   const int2 value,
							   const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__global__ void setDevArr_float3(float3 * __restrict__ devArr,
								 const float3 val,
								 const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = val;
}

__global__ void setDevArr_float4(float4 * __restrict__ devArr,
								 const float4 val,
								 const int numValues)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numValues) { return; }
	devArr[i] = val;
}

__global__ void initPositionBox(float3 * __restrict__ positions,
								int * __restrict__ phases,
								int * __restrict__ phaseCounter,
								const int3 dimension,
								const float3 startPosition,
								const float3 step,
								const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	int x = i % dimension.x;
	int y = (i / dimension.x) % dimension.y;
	int z = i / (dimension.x * dimension.y);
	positions[i] = make_float3(x, y, z) * step + startPosition;
	phases[i] = atomicAdd(phaseCounter, 1);
}

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

__global__ void updateGridId(int * __restrict__ gridIds,
							 int * __restrict__ particleIds,
							 const float3 * __restrict__ positions,
							 const float3 cellOrigin,
							 const float3 cellSize,
							 const int3 gridSize,
							 const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }

	int3 gridPos = calcGridPos(positions[i], cellOrigin, cellSize);
	int gridId = calcGridAddress(gridPos, gridSize);

	gridIds[i] = gridId;
	particleIds[i] = i;
}

__global__ void findStartId(int * cellStart,
							const int * sortedGridIds,
							const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
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

// SOLVER //

__global__ void applyForces(float3 * __restrict__ velocities,
							const float * __restrict__ invMass,
							const int numParticles,
							const float deltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	velocities[i] += make_float3(0.0f, -9.8f, 0.0f) * deltaTime;
}

__global__ void predictPositions(float3 * __restrict__ newPositions,
								 const float3 * __restrict__ positions,
								 const float3 * __restrict__ velocities,
								 const int numParticles,
								 const float deltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	newPositions[i] = positions[i] + velocities[i] * deltaTime;
}

__global__ void updateVelocity(float3 * __restrict__ velocities,
							   const float3 * __restrict__ newPositions,
							   const float3 * __restrict__ positions,
							   const int numParticles,
							   const float invDeltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	velocities[i] = (newPositions[i] - positions[i]) * invDeltaTime;
}

__global__ void planeStabilize(float3 * __restrict__ positions,
							   float3 * __restrict__ newPositions,
							   const int numParticles,
							   const float3 planeOrigin,
							   const float3 planeNormal,
							   const float radius)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	float3 origin2position = planeOrigin - positions[i];
	float distance = dot(origin2position, planeNormal) + radius;
	if (distance <= 0) { return; }

	positions[i] += distance * planeNormal;
	newPositions[i] += distance * planeNormal;
}

// PROJECT CONSTRAINTS //

__global__ void particlePlaneCollisionConstraint(float3 * __restrict__ newPositions,
												 float3 * __restrict__ positions,
												 const int numParticles,
												 const float3 planeOrigin,
												 const float3 planeNormal,
												 const float radius)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	float3 origin2position = planeOrigin - newPositions[i];
	float distance = dot(origin2position, planeNormal) + radius;
	if (distance <= 0) { return; }

	float diffPosition = dot(newPositions[i] - positions[i], planeNormal);
	newPositions[i] += distance * planeNormal;
	positions[i] += (2.0f * diffPosition + distance) * planeNormal * ENERGY_LOST_RATIO;
}

__global__ void particleParticleCollisionConstraint(float3 * __restrict__ newPositionsNext,
													const float3 * __restrict__ newPositionsPrev,
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
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }

	float3 positionPrev = newPositionsPrev[i];
	float3 positionNext = positionPrev;

	int3 centerGridPos = calcGridPos(newPositionsPrev[i], cellOrigin, cellSize);
	int3 start = centerGridPos - 1;
	int3 end = centerGridPos + 1;

	for (int z = start.z; z <= end.z; z++)
		for (int y = start.y; y <= end.y; y++)
			for (int x = start.x; x <= end.x; x++)
			{
				int3 gridPos = make_int3(x, y, z);
				int gridAddress = calcGridAddress(gridPos, gridSize);
				int bucketStart = cellStart[gridAddress];
				if (bucketStart == -1) { continue; }

				for (int k = 0; k < NUM_MAX_PARTICLE_PER_CELL && k + bucketStart < numParticles; k++)
				{
					int gridAddress2 = sortedCellId[bucketStart + k];
					int particleId2 = sortedParticleId[bucketStart + k];
					if (gridAddress2 != gridAddress) { break; }

					if (i != particleId2 && phases[i] != phases[particleId2])
					{
						float3 position2 = newPositionsPrev[particleId2];
						float3 diff = positionPrev - position2;
						float dist2 = length2(diff);
						if (dist2 < radius * radius * 4.0f)
						{
							float dist = sqrtf(dist2);
							positionNext -= diff * (0.5f - radius / dist);
						}
					}
				}
			}

	newPositionsNext[i] = positionNext;
}

// Meshless Deformations Based on Shape Matching
// by Muller et al.

// one block per one shape
#define NUM_MAX_PARTICLE_PER_RIGID_BODY 64
__global__ void shapeMatchingAlphaOne(quaternion * __restrict__ rotations,
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
		//float3 newPosition = qi + CM;
		positions[particleId] = newPosition;
	}
}

struct ParticleSolver
{
	ParticleSolver(const std::shared_ptr<Scene> & scene):
		scene(scene),
		cellOrigin(make_float3(-4, -1, -5)),
		cellSize(make_float3(scene->radius * 2.0f)),
		gridSize(make_int3(128))
	{
		// alloc particle vars
		checkCudaErrors(cudaMalloc(&devPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devTempNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devVelocities, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devMasses, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devInvMasses, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devPhases, scene->numMaxParticles * sizeof(int)));

		// alloc rigid body
		checkCudaErrors(cudaMalloc(&devRigidBodyParticleIdRange, scene->numMaxRigidBodies * sizeof(int2)));
		checkCudaErrors(cudaMalloc(&devRigidBodyCMs, scene->numMaxRigidBodies * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devRigidBodyInitialPositions, scene->numMaxRigidBodies * NUM_MAX_PARTICLE_PER_RIGID_BODY * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devRigidBodyRotations, scene->numMaxRigidBodies * sizeof(quaternion)));
		int numBlocksRigidBody, numThreadsRigidBody;
		GetNumBlocksNumThreads(&numBlocksRigidBody, &numThreadsRigidBody, scene->numMaxRigidBodies);
		setDevArr_float4<<<numBlocksRigidBody, numThreadsRigidBody>>>(devRigidBodyRotations, make_float4(0, 0, 0, 1), scene->numMaxRigidBodies);

		// alloc phase counter
		checkCudaErrors(cudaMalloc(&devPhaseCounter, sizeof(int)));
		checkCudaErrors(cudaMemset(devPhaseCounter, 0, sizeof(int)));

		// alloc grid accel
		checkCudaErrors(cudaMalloc(&devCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devCellStart, gridSize.x * gridSize.y * gridSize.z * sizeof(int)));

		checkCudaErrors(cudaMemset(devVelocities, 0, scene->numMaxParticles * sizeof(float3)));

		// start initing the scene
		for (std::shared_ptr<RigidBody> rigidBody : scene->rigidBodies)
		{
			addRigidBody(rigidBody->positions, rigidBody->positions_CM_Origin, 1.0f);
		}

		for (const glm::vec3 & particlePos : scene->granulars)
		{
			addParticles(glm::vec3(1, 1, 1), particlePos, glm::vec3(0), 1.0f);
		}
	}

	void updateTempStorageSize(const size_t size)
	{
		if (size > devTempStorageSize)
		{
			if (devTempStorage != nullptr) { checkCudaErrors(cudaFree(devTempStorage)); }
			checkCudaErrors(cudaMalloc(&devTempStorage, size));
			devTempStorageSize = size;
		}
	}

	glm::vec3 getParticlePosition(const int particleIndex)
	{
		if (particleIndex < 0 || particleIndex >= scene->numParticles) return glm::vec3(0.0f);
		float3 * tmp = (float3 *)malloc(sizeof(float3));
		cudaMemcpy(tmp, devPositions + particleIndex, sizeof(float3), cudaMemcpyDeviceToHost);
		glm::vec3 result(tmp->x, tmp->y, tmp->z);
		free(tmp);
		return result;
	}

	void setParticle(const int particleIndex, const glm::vec3 & position, const glm::vec3 & velocity)
	{
		if (particleIndex < 0 || particleIndex >= scene->numParticles) return;
		setDevArr_float3<<<1, 1>>>(devPositions + particleIndex, make_float3(position.x, position.y, position.z), 1);
		setDevArr_float3<<<1, 1>>>(devVelocities + particleIndex, make_float3(velocity.x, velocity.y, velocity.z), 1);
	}

	void addParticles(const glm::ivec3 & dimension, const glm::vec3 & startPosition, const glm::vec3 & step, const float mass)
	{
		int numParticles = dimension.x * dimension.y * dimension.z;

		// check if number of particles exceed num max particles or not
		if (scene->numParticles + numParticles >= scene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string(" num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		initPositionBox<<<numBlocks, numThreads>>>(devPositions + scene->numParticles,
												   devPhases + scene->numParticles,
												   devPhaseCounter,
												   make_int3(dimension),
												   make_float3(startPosition),
												   make_float3(step),
												   numParticles);
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   mass,
												   numParticles);
		// set inv masses
		setDevArr_float<<<numBlocks, numThreads>>>(devInvMasses + scene->numParticles,
												   1.0f / mass,
												   numParticles);
		scene->numParticles += numParticles;
		isNotRigidParticlesAdded = true;
	}

	void addRigidBody(const std::vector<glm::vec3> & initialPositions, const std::vector<glm::vec3> & initialPositions_CM_Origin, const float massPerParticle)
	{
		int numParticles = initialPositions.size();
		if (scene->numParticles + numParticles >= scene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		if (scene->numRigidBodies + 1 >= scene->numMaxRigidBodies)
		{
			std::string message = std::string(__FILE__) + std::string("num rigid bodies exceed num max rigid bodies");
			throw std::exception(message.c_str());
		}

		glm::vec3 cm = glm::vec3(0.0f);
		for (const glm::vec3 & position : initialPositions_CM_Origin) { cm += position; }
		cm /= (float)initialPositions_CM_Origin.size();

		if (glm::length(cm) >= 1e-5f)
		{
			std::string message = std::string(__FILE__) + std::string("expected Center of Mass at the origin");
			throw std::exception(message.c_str());
		}

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + scene->numParticles,
								   &(initialPositions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(devRigidBodyInitialPositions + scene->numParticles,
								   &(initialPositions_CM_Origin[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set inv masses
		setDevArr_float<<<numBlocks, numThreads>>>(devInvMasses + scene->numParticles,
												   1.0f / massPerParticle,
												   numParticles);
		// set phases
		setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
													   devPhaseCounter,
													   numParticles);
		// set range for particle id
		setDevArr_int2<<<1, 1>>>(devRigidBodyParticleIdRange + scene->numRigidBodies,
								 make_int2(scene->numParticles, scene->numParticles + numParticles),
								 1);
		// increment phase counter
		increment<<<1, 1>>>(devPhaseCounter);
		
		scene->numParticles += numParticles;
		scene->numRigidBodies += 1;
		devMaxRigidBodyParticleId = scene->numParticles;
	}

	void updateGrid(int numBlocks, int numThreads)
	{
		setDevArr_int<<<numBlocks, numThreads>>>(devCellStart, -1, scene->numMaxParticles);
		updateGridId<<<numBlocks, numThreads>>>(devCellId,
												devParticleId,
												devNewPositions,
												cellOrigin,
												cellSize,
												gridSize,
												scene->numParticles);
		size_t tempStorageSize = 0;
		// get temp storage size (not sorting yet)
		cub::DeviceRadixSort::SortPairs(NULL,
										tempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										scene->numParticles);
		updateTempStorageSize(tempStorageSize);
		// sort!
		cub::DeviceRadixSort::SortPairs(devTempStorage,
										devTempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										scene->numParticles);
		findStartId<<<numBlocks, numThreads>>>(devCellStart, devSortedCellId, scene->numParticles);
	}

	void update(const int numSubTimeStep,
				const float deltaTime,
				const int pickedParticleId = -1,
				const glm::vec3 & pickedParticlePosition = glm::vec3(0.0f),
				const glm::vec3 & pickedParticleVelocity = glm::vec3(0.0f))
	{
		float subDeltaTime = deltaTime / (float)numSubTimeStep;
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, scene->numParticles);

		for (int i = 0;i < numSubTimeStep;i++)
		{ 
			applyForces<<<numBlocks, numThreads>>>(devVelocities,
												   devInvMasses,
												   scene->numParticles,
												   subDeltaTime);

			// we need to make picked particle immovable
			if (pickedParticleId >= 0 && pickedParticleId < scene->numParticles)
			{
				setParticle(pickedParticleId, pickedParticlePosition, glm::vec3(0.0f));
			}

			predictPositions<<<numBlocks, numThreads>>>(devNewPositions,
														devPositions,
														devVelocities,
														scene->numParticles,
														subDeltaTime);

			// stabilize iterations
			for (int i = 0; i < 2; i++)
			{
				for (const Plane & plane : scene->planes)
				{
					planeStabilize<<<numBlocks, numThreads>>>(devPositions,
															  devNewPositions,
															  scene->numParticles,
															  make_float3(plane.origin),
															  make_float3(plane.normal),
															  scene->radius);
				}
			}

			// projecting constraints iterations
			// (update grid every n iterations)
			for (int i = 0; i < 2; i++)
			{
				// compute grid
				updateGrid(numBlocks, numThreads);

				for (int j = 0; j < 5; j++)
				{
					// solving all plane collisions
					for (const Plane & plane : scene->planes)
					{
						particlePlaneCollisionConstraint<<<numBlocks, numThreads>>>(devNewPositions,
																					devPositions,
																					scene->numParticles,
																					make_float3(plane.origin),
																					make_float3(plane.normal),
																					scene->radius);
					}

					// solving all particles collisions
					particleParticleCollisionConstraint<<<numBlocks, numThreads>>>(devTempNewPositions,
																				   devNewPositions,
																				   devPhases,
																				   devSortedCellId,
																				   devSortedParticleId,
																				   devCellStart,
																				   cellOrigin,
																				   cellSize,
																				   gridSize,
																				   scene->numParticles,
																				   scene->radius);
					std::swap(devTempNewPositions, devNewPositions);
					// solve all rigidbody constraints
					shapeMatchingAlphaOne<<<scene->numRigidBodies, NUM_MAX_PARTICLE_PER_RIGID_BODY>>>(devRigidBodyRotations,
																									  devRigidBodyCMs,
																									  devNewPositions,
																									  devRigidBodyInitialPositions,
																									  devRigidBodyParticleIdRange);
				}
			}

			updateVelocity<<<numBlocks, numThreads>>>(devVelocities,
													  devNewPositions,
													  devPositions,
													  scene->numParticles,
													   1.0f / subDeltaTime);
			std::swap(devNewPositions, devPositions); // update position
		}

		// we need to make picked particle immovable
		if (pickedParticleId >= 0 && pickedParticleId < scene->numParticles)
		{
			glm::vec3 solvedPickedParticlePosition = getParticlePosition(pickedParticleId);
			setParticle(pickedParticleId, solvedPickedParticlePosition, pickedParticleVelocity);
		}
	}

	/// TODO:: implement object's destroyer

	float3 *	devPositions;
	float3 *	devNewPositions;
	float3 *	devTempNewPositions;
	float3 *	devVelocities;
	float *		devMasses;
	float *		devInvMasses;
	int *		devPhases;
	int *		devPhaseCounter;

	int *		devSortedCellId;
	int *		devSortedParticleId;

	int2 *		devRigidBodyParticleIdRange;
	float3 *	devRigidBodyInitialPositions;
	quaternion * devRigidBodyRotations;
	float3 *	devRigidBodyCMs;// center of mass

	void *		devTempStorage = nullptr;
	size_t		devTempStorageSize = 0;

	int			devMaxRigidBodyParticleId = 0;
	bool		isNotRigidParticlesAdded = false;

	int *			devCellId;
	int *			devParticleId;
	int *			devCellStart;
	const float3	cellOrigin;
	const float3	cellSize;
	const int3		gridSize;

	std::shared_ptr<Scene> scene;
};
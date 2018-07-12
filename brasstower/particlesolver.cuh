#pragma once

#include <iostream>
#include <vector>
#include <exception>
#include <conio.h>

#include "kernel/util.cuh"
#include "kernel/grid.cuh"

#include "kernel/fluid.cuh"
#include "kernel/particleplane.cuh"
#include "kernel/particleparticle.cuh"
#include "kernel/shapematching.cuh"
#include "kernel/distance.cuh"
#include "kernel/bending.cuh"
#include "kernel/wind.cuh"
#include "kernel/immovable.cuh"

template <typename T>
void CudaAlloc(T ** devPtr, size_t num)
{
	if (num <= 0) { return; }
	checkCudaErrors(cudaMalloc(devPtr, num * sizeof(T)));
}

template <typename T>
T * CudaAlloc2(const size_t num)
{
	if (num <= 0) { return nullptr; }
	T * devPtr = nullptr;
	checkCudaErrors(cudaMalloc(&devPtr, num * sizeof(T)));
	return devPtr;
}

template <typename T>
void CudaAllocAndCopy(T ** devPtr, const std::vector<T> & vec)
{
	if (vec.size() <= 0) { return; }
	CudaAlloc(devPtr, vec.size());
	checkCudaErrors(cudaMemcpy(*devPtr,
							   &(vec[0]),
							   vec.size() * sizeof(T),
							   cudaMemcpyHostToDevice));
}

// SOLVER //

__global__ void
applyGravity(float3 * __restrict__ positions,
			const int numParticles,
			const float deltaTime)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	positions[i] += make_float3(0.0f, -9.8f, 0.0f) * deltaTime * deltaTime;
}

__global__ void
predictPositions(float3 * __restrict__ newPositions,
				 const float3 * __restrict__ positions,
				 const float3 * __restrict__ velocities,
				 const int numParticles,
				 const float deltaTime)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	newPositions[i] = positions[i] + velocities[i] * deltaTime;
}

__global__ void
updateVelocity(float3 * __restrict__ velocities,
			   const float3 * __restrict__ newPositions,
			   const float3 * __restrict__ positions,
			   const int numParticles,
			   const float invDeltaTime)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	velocities[i] = (newPositions[i] - positions[i]) * invDeltaTime;
}

// for shock propagation
__global__ void
computeInvScaledMasses(float* __restrict__ invScaledMasses,
					   const float* __restrict__ masses,
					   const float3* __restrict__ positions,
					   const float k,
					   const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const float e = 2.7182818284f;
	const float height = positions[i].y;
	const float scale = pow(e, -k * height);
	invScaledMasses[i] = 1.0f / (scale * masses[i]);
}

// for shock propagation
__global__ void
computeInvMassesAndInvScaledMasses(float* __restrict__ invScaledMasses,
								   float* __restrict__ invMasses,
								   const float* __restrict__ masses,
								   const float3* __restrict__ positions,
								   const float k,
								   const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	float mass = masses[i];
	invMasses[i] = 1.0f / (mass);
	const float e = 2.7182818284f;
	const float height = positions[i].y;
	///TODO:: switch to exp and test if it still gives identical result
	const float scale = pow(e, -k * height);
	invScaledMasses[i] = 1.0f / (scale * mass);
}

__global__ void
updatePositions(float3 * __restrict__ positions,
				const float3 * __restrict__ newPositions,
				const int * __restrict__ phases,
				const float threshold,
				const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }

	const int phase = phases[i];
	const float3 x = positions[i];
	const float3 newX = newPositions[i];

	const float dist2 = length2(newX - x);
	positions[i] = (dist2 >= threshold * threshold || phases[i] < 0) ? newX : x;
}

__device__ int queriedId;
__global__ void
queryId(int id, int * mapToIds)
{
    queriedId = mapToIds[id];
}

struct ParticleSolver
{
	ParticleSolver(const std::shared_ptr<Scene> & scene):
		cellOrigin(make_float3(-4.01, -1.01, -5.01)),
		cellSize(make_float3(scene->particleRadius * 2.3f)),
		gridSize(make_int3(512))
	{
		solidRadius = scene->particleRadius;
		SetKernelRadius(scene->fluidKernelRadius);
		numParticles = scene->numParticles();

		// alloc particle variables
		CudaAlloc(&devNewPositions, numParticles);
		CudaAlloc(&devVelocities, numParticles);
		CudaAllocAndCopy(&devPositions, scene->positions);
		CudaAllocAndCopy(&devMasses, scene->masses);
		CudaAllocAndCopy(&devPhases, scene->phases);
		CudaAllocAndCopy(&devGroupIds, scene->groupIds);
		CudaAlloc(&devMapToOriginalIds, numParticles);
		CudaAlloc(&devMapToNewIds, numParticles);

		SetDevArr(devVelocities, make_float3(0.f), numParticles);
		InitOrder_int(devMapToOriginalIds, numParticles);

		// alloc sorted particle variables
		CudaAlloc(&devSortedNewPositions, numParticles);
		CudaAlloc(&devSortedNewPositions, numParticles);
		CudaAlloc(&devSortedPositions, numParticles);
		CudaAlloc(&devSortedVelocities, numParticles);
		CudaAlloc(&devSortedMasses, numParticles);
		CudaAlloc(&devSortedPhases, numParticles);
		CudaAlloc(&devSortedMapToOriginalIds, numParticles);
		CudaAlloc(&devSortedGroupIds, numParticles);

		// alloc grid accel
		size_t numCells = gridSize.x * gridSize.y * gridSize.z;
		CudaAlloc(&devCellId, numParticles);
		CudaAlloc(&devParticleId, numParticles);
		CudaAlloc(&devSortedCellId, numParticles);
		CudaAlloc(&devSortedParticleId, numParticles);
		CudaAlloc(&devCellStart, numCells);
		CudaAlloc(&devCellEnd, numCells);
		checkCudaErrors(cudaMemcpyToSymbol(GridDim, &gridSize.x, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(GridCellSize, &cellSize, sizeof(float3)));
		SetDevArr(devCellStart, -1, numCells);
		SetDevArr(devCellEnd, -1, numCells);

		// intermediates variables
		CudaAlloc(&devDeltaX, numParticles);
		CudaAlloc(&devTempFloat3, numParticles);
		CudaAlloc(&devInvMasses, numParticles);
		CudaAlloc(&devInvScaledMasses, numParticles);

		// rigid body
		size_t numRigidbody = scene->rigidbodyParticleIdRanges.size();
		CudaAllocAndCopy(&devRigidBodyParticleIdRange, scene->rigidbodyParticleIdRanges);
		CudaAllocAndCopy(&devRigidBodyInitialPositions, scene->rigidbodyInitialPositions);
		CudaAlloc(&devRigidBodyRotations, numRigidbody);
		SetDevArr(devRigidBodyRotations, make_float4(0.f, 0.f, 0.f, 1.f), numRigidbody);

		// distance constraints
		size_t numDistanceConstraints = scene->distancePairs.size();
		CudaAllocAndCopy(&devDistancePairs, scene->distancePairs);
		CudaAllocAndCopy(&devDistanceParams, scene->distanceParams);

		// bending constraints
		size_t numBendingConstraints = scene->bendingConstraints.size();

		planes = scene->planes;
	}


	ParticleSolver(const std::shared_ptr<OldSceneFormat> & scene):
		oldScene(scene),
		cellOrigin(make_float3(-1000.01, -1000.01, -1000.01)),
		cellSize(make_float3(scene->radius * 2.3f)),
		gridSize(make_int3(512))
	{
		fluidKernelRadius = 2.3f * scene->radius;
		SetKernelRadius(fluidKernelRadius);
		fluidPhaseCounter = -1;

		// alloc particle vars
		//checkCudaErrors(cudaMalloc(&devNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devVelocities, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devMasses, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devPhases, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devMapToOriginalIds, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devMapToNewIds, scene->numMaxParticles * sizeof(int)));
		int numBlocksMax, numThreadsMax;
		GetNumBlocksNumThreads(&numBlocksMax, &numThreadsMax, scene->numMaxParticles);
		initOrder_int<<<numBlocksMax, numThreadsMax>>>(devMapToOriginalIds, scene->numMaxParticles);

		checkCudaErrors(cudaMalloc(&devSortedNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devSortedPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devSortedVelocities, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devSortedMasses, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devSortedPhases, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedMapToOriginalIds, scene->numMaxParticles * sizeof(int)));

		checkCudaErrors(cudaMalloc(&devDeltaX, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devTempFloat3, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devInvScaledMasses, scene->numMaxParticles * sizeof(float)));

		// alloc group id
		checkCudaErrors(cudaMalloc(&devGroupIds, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedGroupIds, scene->numMaxParticles * sizeof(int)));

		// set velocity
		checkCudaErrors(cudaMemset(devVelocities, 0, scene->numMaxParticles * sizeof(float3)));

		// alloc rigid body
		checkCudaErrors(cudaMalloc(&devRigidBodyParticleIdRange, scene->numMaxRigidBodies * sizeof(int2)));
		checkCudaErrors(cudaMalloc(&devRigidBodyCMs, scene->numMaxRigidBodies * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devRigidBodyInitialPositions, scene->numMaxRigidBodies * NUM_MAX_PARTICLE_PER_RIGID_BODY * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devRigidBodyRotations, scene->numMaxRigidBodies * sizeof(quaternion)));
		int numBlocksRigidBody, numThreadsRigidBody;
		GetNumBlocksNumThreads(&numBlocksRigidBody, &numThreadsRigidBody, scene->numMaxRigidBodies);
		setDevArr_float4<<<numBlocksRigidBody, numThreadsRigidBody>>>(devRigidBodyRotations, make_float4(0, 0, 0, 1), scene->numMaxRigidBodies);

		// alloc distance constraints
		checkCudaErrors(cudaMalloc(&devDistancePairs, scene->numMaxDistancePairs * sizeof(int2)));
		checkCudaErrors(cudaMalloc(&devDistanceParams, scene->numMaxDistancePairs * sizeof(float2)));

		// alloc bending constraints
		checkCudaErrors(cudaMalloc(&devBendings, scene->numMaxBendings * sizeof(int4)));

		// alloc and set phase counter
		checkCudaErrors(cudaMalloc(&devSolidPhaseCounter, sizeof(int)));
		setDevArr_int<<<1, 1>>>(devSolidPhaseCounter, 1, 1);

		// alloc grid accel
		checkCudaErrors(cudaMalloc(&devCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devCellStart, gridSize.x * gridSize.y * gridSize.z * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devCellEnd, gridSize.x * gridSize.y * gridSize.z * sizeof(int)));
		checkCudaErrors(cudaMemcpyToSymbol(GridDim, &gridSize.x, sizeof(uint)));
		checkCudaErrors(cudaMemcpyToSymbol(GridCellSize, &cellSize, sizeof(float3)));
		int numCellBlocks, numCellThreads;
		int numCells = gridSize.x * gridSize.y * gridSize.z;
		GetNumBlocksNumThreads(&numCellBlocks, &numCellThreads, numCells);
		setDevArr_int<<<numCellBlocks, numCellThreads>>>(devCellStart, -1, numCells);
		setDevArr_int<<<numCellBlocks, numCellThreads>>>(devCellEnd, -1, numCells);

		// alloc fluid vars
		checkCudaErrors(cudaMalloc(&devFluidOmegas, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devFluidLambdas, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devFluidDensities, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devFluidNormals, scene->numMaxParticles * sizeof(float3)));

		// alloc wind faces
		checkCudaErrors(cudaMalloc(&devWindFaces, scene->numMaxWindFaces * sizeof(int3)));

		// alloc immovable constraints
		checkCudaErrors(cudaMalloc(&devImmovables, scene->numMaxImmovables * sizeof(int)));

		// start initing the scene
		for (std::shared_ptr<RigidBody> rigidBody : scene->rigidBodies)
		{
			addNewGroup(scene->numParticles, rigidBody->positions.size());
			addRigidBody(rigidBody->positions, rigidBody->positions_CM_Origin, rigidBody->massPerParticle);
		}

		for (std::shared_ptr<Granulars> granulars : scene->granulars)
		{
			addNewGroup(scene->numParticles, granulars->positions.size());
			addGranulars(granulars->positions, granulars->massPerParticle);
		}

		///TODO:: find out why adding fluid before rigid leads to broken sim
		for (std::shared_ptr<Fluid> fluids : scene->fluids)
		{
			addNewGroup(scene->numParticles, fluids->positions.size());
			addFluids(fluids->positions, fluids->massPerParticle);
		}

		for (std::shared_ptr<Rope> ropes : scene->ropes)
		{
			addNewGroup(scene->numParticles, ropes->positions.size());
			addNoodles(ropes->positions, ropes->distancePairs, ropes->distanceParams, ropes->massPerParticle);
		}

		for (std::shared_ptr<Cloth> cloth : scene->clothes)
		{
			addNewGroup(scene->numParticles, cloth->positions.size());
			addCloth(cloth->positions, cloth->distancePairs, cloth->distanceParams, cloth->bendings, cloth->faces, cloth->immovables, cloth->massPerParticle);
		}

		fluidRestDensity = scene->fluidRestDensity;

		planes = scene->planes;
		numParticles = scene->numParticles;
	}

	void addNewGroup(int start, int numParticles)
	{
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);
		setDevArr_int<<<numBlocks, numThreads>>>(devGroupIds + start, groupIdCounter, numParticles);
		groupIdCounter++;
	}

    int queryOriginalParticleId(int newParticleId)
    {
        queryId<<<1, 1>>>(newParticleId, devMapToOriginalIds);
        int originalParticleId;
        checkCudaErrors(cudaMemcpyFromSymbol(&originalParticleId, queriedId, sizeof(originalParticleId), 0, cudaMemcpyDeviceToHost));
        return originalParticleId;
    }

    int queryNewParticleId(int oldParticleId)
    {
        queryId<<<1, 1>>>(oldParticleId, devMapToNewIds);
        int newParticleId;
        checkCudaErrors(cudaMemcpyFromSymbol(&newParticleId, queriedId, sizeof(newParticleId), 0, cudaMemcpyDeviceToHost));
        return newParticleId;
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
		if (particleIndex < 0 || particleIndex >= oldScene->numParticles) return glm::vec3(0.0f);
		float3 * tmp = (float3 *)malloc(sizeof(float3));
		cudaMemcpy(tmp, devPositions + particleIndex, sizeof(float3), cudaMemcpyDeviceToHost);
		glm::vec3 result(tmp->x, tmp->y, tmp->z);
		free(tmp);
		return result;
	}

	void setParticle(float3 * devPositions, const int particleIndex, const glm::vec3 & position, const glm::vec3 & velocity)
	{
		if (particleIndex < 0 || particleIndex >= oldScene->numParticles) return;
		setDevArr_float3<<<1, 1>>>(devPositions + particleIndex, make_float3(position.x, position.y, position.z), 1);
		setDevArr_float3<<<1, 1>>>(devVelocities + particleIndex, make_float3(velocity.x, velocity.y, velocity.z), 1);
	}

	void addGranulars(const std::vector<glm::vec3> & positions, const float massPerParticle)
	{
		int numParticles = positions.size();
		if (oldScene->numParticles + numParticles >= oldScene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + oldScene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + oldScene->numParticles,
												   massPerParticle,
												   numParticles);	
		// set phases
		setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
															  devSolidPhaseCounter,
															  1,
															  numParticles);
		oldScene->numParticles += numParticles;
	}

	void addRigidBody(const std::vector<glm::vec3> & initialPositions, const std::vector<glm::vec3> & initialPositions_CM_Origin, const float massPerParticle)
	{
		int numParticles = initialPositions.size();
		if (oldScene->numParticles + numParticles >= oldScene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		if (oldScene->numRigidBodies + 1 >= oldScene->numMaxRigidBodies)
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
		checkCudaErrors(cudaMemcpy(devPositions + oldScene->numParticles,
								   &(initialPositions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(devRigidBodyInitialPositions + oldScene->numParticles,
								   &(initialPositions_CM_Origin[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + oldScene->numParticles,
												   massPerParticle,
												   numParticles);
		// set phases
		setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
													   devSolidPhaseCounter,
													   numParticles);
		// set range for particle id
		setDevArr_int2<<<1, 1>>>(devRigidBodyParticleIdRange + oldScene->numRigidBodies,
								 make_int2(oldScene->numParticles, oldScene->numParticles + numParticles),
								 1);
		// increment phase counter
		increment<<<1, 1>>>(devSolidPhaseCounter);
		
		oldScene->numParticles += numParticles;
		oldScene->numRigidBodies += 1;
	}

	void addFluids(const std::vector<glm::vec3> & positions, const float massPerParticle)
	{
		int numParticles = positions.size();
		if (oldScene->numParticles + numParticles >= oldScene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + oldScene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + oldScene->numParticles,
												   massPerParticle,
												   numParticles);	
		// fluid phase is always < 0
		setDevArr_int<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
												 fluidPhaseCounter,
												 numParticles);
		fluidPhaseCounter -= 1;
		oldScene->numParticles += numParticles;
	}

	// spaghetti
    void addNoodles(const std::vector<glm::vec3> & positions, const std::vector<glm::int2> & distancePairs, const std::vector<glm::vec2> & distanceParams, const float massPerParticle, const bool doSelfCollide = true)
    {
        int numParticles = positions.size();
        if (oldScene->numParticles + numParticles >= oldScene->numMaxParticles)
        {
            std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
            throw std::exception(message.c_str());
        }

        int numDistanceConstraints = distancePairs.size();
        if (oldScene->numParticles + numDistanceConstraints >= oldScene->numMaxDistancePairs)
        {
            std::string message = std::string(__FILE__) + std::string("num distance pairs exceed num max distance pairs");
            throw std::exception(message.c_str());
        }
    
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + oldScene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + oldScene->numParticles,
												   massPerParticle,
												   numParticles);	
		if (doSelfCollide)
		{ 
			// set phases
			setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
																  devSolidPhaseCounter,
																  1,
																  numParticles);
		}
		else
		{ 
			// set phases
			setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
														   devSolidPhaseCounter,
														   numParticles);
			// increment phase counter
			increment<<<1, 1>>>(devSolidPhaseCounter);
		}

		// set distance constraints
		checkCudaErrors(cudaMemcpy(devDistancePairs + oldScene->numDistancePairs,
								   &(distancePairs[0].x),
								   numDistanceConstraints * sizeof(int) * 2,
								   cudaMemcpyHostToDevice));
		int numDistancePairsBlock, numDistancePairsThreads;
		GetNumBlocksNumThreads(&numDistancePairsBlock, &numDistancePairsThreads, numDistanceConstraints);
		accDevArr_int2<<<numDistancePairsBlock, numDistancePairsThreads>>>(devDistancePairs + oldScene->numDistancePairs,
																		   make_int2(oldScene->numParticles),
																		   numDistanceConstraints);
		checkCudaErrors(cudaMemcpy(devDistanceParams + oldScene->numDistancePairs,
								   &(distanceParams[0]),
								   numDistanceConstraints * sizeof(float) * 2,
								   cudaMemcpyHostToDevice));

		oldScene->numParticles += numParticles;
		oldScene->numDistancePairs += numDistanceConstraints;
	}

    // cloth
	void addCloth(const std::vector<glm::vec3> & positions,
				  const std::vector<glm::int2> & distancePairs,
				  const std::vector<glm::vec2> & distanceParams,
				  const std::vector<glm::int4> & bendings,
				  const std::vector<glm::int3> & faces,
				  const std::vector<int> & immovables,
				  const float massPerParticle,
				  const bool doSelfCollide = true)
    {
        int numParticles = positions.size();
        if (oldScene->numParticles + numParticles >= oldScene->numMaxParticles)
        {
            std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
            throw std::exception(message.c_str());
        }

        int numDistanceConstraints = distancePairs.size();
        if (oldScene->numDistancePairs + numDistanceConstraints >= oldScene->numMaxDistancePairs)
        {
            std::string message = std::string(__FILE__) + std::string("num distance pairs exceed num max distance pairs");
            throw std::exception(message.c_str());
        }
    
		int numBendings = bendings.size();
		if (oldScene->numBendings + numBendings >= oldScene->numMaxBendings)
		{
            std::string message = std::string(__FILE__) + std::string("num bendings exceed num max bendings");
            throw std::exception(message.c_str());
		}

		int numFaces = faces.size();
		if (oldScene->numWindFaces + numFaces >= oldScene->numMaxWindFaces)
		{
            std::string message = std::string(__FILE__) + std::string("num faces exceed num max faces");
            throw std::exception(message.c_str());
		}

		int numImmovables = immovables.size();
		if (oldScene->numImmovables + numImmovables >= oldScene->numMaxImmovables)
		{
            std::string message = std::string(__FILE__) + std::string("num immovables exceed num max immovables");
            throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + oldScene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + oldScene->numParticles,
												   massPerParticle,
												   numParticles);	
		if (doSelfCollide)
		{ 
			// set phases
			setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
																  devSolidPhaseCounter,
																  1,
																  numParticles);
		}
		else
		{ 
			// set phases
			setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + oldScene->numParticles,
														   devSolidPhaseCounter,
														   numParticles);
			// increment phase counter
			increment<<<1, 1>>>(devSolidPhaseCounter);
		}

		// set distance constraints
		checkCudaErrors(cudaMemcpy(devDistancePairs + oldScene->numDistancePairs,
								   &(distancePairs[0].x),
								   numDistanceConstraints * sizeof(int) * 2,
								   cudaMemcpyHostToDevice));
		int numDistancePairsBlock, numDistancePairsThreads;
		GetNumBlocksNumThreads(&numDistancePairsBlock, &numDistancePairsThreads, numDistanceConstraints);
		accDevArr_int2<<<numDistancePairsBlock, numDistancePairsThreads>>>(devDistancePairs + oldScene->numDistancePairs,
																		   make_int2(oldScene->numParticles),
																		   numDistanceConstraints);
		checkCudaErrors(cudaMemcpy(devDistanceParams + oldScene->numDistancePairs,
								   &(distanceParams[0].x),
								   numDistanceConstraints * sizeof(float) * 2,
								   cudaMemcpyHostToDevice));

		if (numBendings > 0)
		{ 
			int numBendingBlocks, numBendingThreads;
			GetNumBlocksNumThreads(&numBendingBlocks, &numBendingThreads, numBendings);
			// set bending constraints
			checkCudaErrors(cudaMemcpy(devBendings + oldScene->numBendings,
									   &(bendings[0].x),
									   numBendings * sizeof(int4),
									   cudaMemcpyHostToDevice));
			accDevArr_int4<<<numBendingBlocks, numBendingThreads>>>(devBendings + oldScene->numBendings,
																	make_int4(oldScene->numParticles),
																	numBendings);
		}

		// add faces
		int numWindFaceBlocks, numWindFaceThreads;
		GetNumBlocksNumThreads(&numWindFaceBlocks, &numWindFaceThreads, numFaces);
		checkCudaErrors(cudaMemcpy(devWindFaces + oldScene->numWindFaces,
								   &(faces[0].x),
								   numFaces * sizeof(int3),
								   cudaMemcpyHostToDevice));
		accDevArr_int3<<<numWindFaceBlocks, numWindFaceThreads>>>(devWindFaces + oldScene->numWindFaces,
																  make_int3(oldScene->numParticles),
																  numFaces);


		// add immovable constraints
		if (numImmovables > 0)
		{
			int numImmovableBlocks, numImmovableThreads;
			GetNumBlocksNumThreads(&numImmovableBlocks, &numImmovableThreads, numImmovables);
			checkCudaErrors(cudaMemcpy(devImmovables + oldScene->numImmovables,
									   &(immovables[0]),
									   numImmovables * sizeof(int),
									   cudaMemcpyHostToDevice));
			accDevArr_int<<<numImmovableBlocks, numImmovableThreads>>>(devImmovables + oldScene->numImmovables,
																	   oldScene->numParticles,
																	   numImmovables);
		}

		oldScene->numImmovables += numImmovables;
		oldScene->numBendings += numBendings;
		oldScene->numParticles += numParticles;
		oldScene->numDistancePairs += numDistanceConstraints;
		oldScene->numWindFaces += numFaces;
    }

	void updateGrid(int numBlocks, int numThreads)
	{
		updateGridId<<<numBlocks, numThreads>>>(devCellId,
												devParticleId,
												devNewPositions,
												cellOrigin,
												cellSize,
												gridSize,
												numParticles);
		size_t tempStorageSize = 0;
		// get temp storage size (not sorting yet)
		cub::DeviceRadixSort::SortPairs(NULL,
										tempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										numParticles);
		updateTempStorageSize(tempStorageSize);
		// sort!
		cub::DeviceRadixSort::SortPairs(devTempStorage,
										devTempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										numParticles);
		findStartEndId<<<numBlocks, numThreads>>>(devCellStart, devCellEnd, devSortedCellId, numParticles);

		reorderParticlesData<<<numBlocks, numThreads>>>(devSortedNewPositions,
														devSortedPositions,
														devSortedVelocities,
														devSortedMasses,
														devSortedPhases,
														devSortedMapToOriginalIds,
														devSortedGroupIds,
														devNewPositions,
														devPositions,
														devVelocities,
														devMasses,
														devPhases,
														devGroupIds,
														devMapToOriginalIds,
														devSortedParticleId,
														numParticles);

		std::swap(devSortedNewPositions, devNewPositions);
		std::swap(devSortedPositions, devPositions);
		std::swap(devSortedVelocities, devVelocities);
		std::swap(devSortedMasses, devMasses);
		std::swap(devSortedPhases, devPhases);
		std::swap(devSortedGroupIds, devGroupIds);
		std::swap(devSortedMapToOriginalIds, devMapToOriginalIds);
	}

	void resetGrid(int numBlocks, int numThreads)
	{
		resetStartEndId<<<numBlocks, numThreads>>>(devCellStart, devCellEnd, devSortedCellId, numParticles);
	}

	void update(const int numSubTimeStep,
				const float deltaTime,
				const int pickedOriginalParticleId = -1,
				const glm::vec3 & pickedParticlePosition = glm::vec3(0.0f),
				const glm::vec3 & pickedParticleVelocity = glm::vec3(0.0f))
	{
		if (numParticles <= 0) return;

		float subDeltaTime = deltaTime / (float)numSubTimeStep;
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		int3 fluidGridSearchOffset = make_int3(ceil(make_float3(fluidKernelRadius) / cellSize));
		bool useAkinciCohesionTension = true;

		for (int i = 0;i < numSubTimeStep;i++)
		{ 
			// we need to make picked particle immovable
			if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < numParticles)
			{
                int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
				setParticle(devPositions, pickedNewParticleId, pickedParticlePosition, glm::vec3(0.0f));
			}

			predictPositions<<<numBlocks, numThreads>>>(devNewPositions,
														devPositions,
														devVelocities,
														numParticles,
														subDeltaTime);

			applyGravity<<<numBlocks, numThreads>>>(devNewPositions,
													numParticles,
													subDeltaTime);

			// compute grid
			updateGrid(numBlocks, numThreads);

			inverseMapping<<<numBlocks, numThreads>>>(devMapToNewIds, devMapToOriginalIds, numParticles);

			// compute scaled masses
			computeInvScaledMasses<<<numBlocks, numThreads>>>(devInvScaledMasses,
															  devMasses,
															  devPositions,
															  MASS_SCALING_CONSTANT,
															  numParticles);
			#ifdef CHECK_NAN
				printIfNan(devInvScaledMasses, oldScene->numParticles, "find nan after InvScaledMasses");
			#endif

			// stabilize iterations
			for (int i = 0; i < 2; i++)
			{
				for (const Plane & plane : planes)
				{
					planeStabilize<<<numBlocks, numThreads>>>(devPositions,
															  devNewPositions,
															  numParticles,
															  make_float3(plane.origin),
															  make_float3(plane.normal),
															  solidRadius);
				}
			}
			#ifdef CHECK_NAN
				printIfNan(devNewPositions, oldScene->numParticles, "fina nan after plane stabilize collision");
			#endif

			/*
			// apply Wind Force
			if (oldScene->numWindFaces > 0)
			{
				int numWindFaceBlocks, numWindFaceThreads;
				GetNumBlocksNumThreads(&numWindFaceBlocks, &numWindFaceThreads, oldScene->numWindFaces);
				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), oldScene->numParticles);
				applyWindForce<<<numWindFaceBlocks, numWindFaceThreads>>>(devDeltaX,
																		  devNewPositions,
																		  devVelocities,
																		  devMasses,
																		  devMapToNewIds,
																		  devWindFaces,
																		  oldScene->numWindFaces,
																		  subDeltaTime);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, oldScene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, oldScene->numParticles, "find nan after wind force applied");
				#endif
			}
			*/

			// projecting constraints iterations
			// (update grid every n iterations)
			for (int j = 0; j < 2;j++)
			{
				// we need to make picked particle immovable
				if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < oldScene->numParticles)
				{
					int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
					setParticle(devNewPositions, pickedNewParticleId, pickedParticlePosition, glm::vec3(0.0f));
				}

				// solving all plane collisions
				for (const Plane & plane : planes)
				{
					particlePlaneCollisionConstraint<<<numBlocks, numThreads>>>(devNewPositions,
																				devPositions,
																				numParticles,
																				make_float3(plane.origin),
																				make_float3(plane.normal),
																				solidRadius);
				}
				/*
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, oldScene->numParticles, "find nan after particle plane collision");
				#endif

				// solving all particles collisions
				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), oldScene->numParticles);
				particleParticleCollisionConstraint<<<numBlocks, numThreads>>>(devDeltaX,
																			   devNewPositions,
																			   devPositions,
																			   devInvScaledMasses,
																			   devPhases,
																			   devCellStart,
																			   devCellEnd,
																			   numParticles,
																			   solidRadius);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, oldScene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, oldScene->numParticles, "find nan after particle particle collision");
				#endif

				// fluid
				fluidLambda<<<numBlocks, numThreads>>>(devFluidLambdas,
													   devFluidDensities,
													   devNewPositions,
													   devMasses,
													   devPhases,
													   fluidRestDensity,
													   2.0f, // solid density scaling
													   300.0f, // relaxation parameter
													   devCellStart,
													   devCellEnd,
													   oldScene->numParticles,
													   useAkinciCohesionTension);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, oldScene->numParticles, "find nan after fluid lambda");
				#endif

				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), oldScene->numParticles);
				fluidPosition<<<numBlocks, numThreads>>>(devDeltaX,
														 devNewPositions,
														 devFluidLambdas,
														 fluidRestDensity,
														 devMasses,
														 devPhases,
														 0.0001f, // k for sCorr
														 4, // N for sCorr
														 devCellStart,
														 devCellEnd,
														 oldScene->numParticles,
														 useAkinciCohesionTension);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, oldScene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, oldScene->numParticles, "find nan after fluid position");
				#endif

				// solve all distance constraints
				if (oldScene->numDistancePairs > 0)
				{ 
					setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), oldScene->numParticles);
					int numDistanceBlocks, numDistanceThreads;
					GetNumBlocksNumThreads(&numDistanceBlocks, &numDistanceThreads, oldScene->numDistancePairs);
					distanceConstraints<<<numDistanceBlocks, numDistanceThreads>>>(devDeltaX,
																				   devNewPositions,
																				   devInvScaledMasses,
																				   devDistancePairs,
																				   devDistanceParams,
																				   devMapToNewIds,
																				   oldScene->numDistancePairs);
					accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, oldScene->numParticles);
					#ifdef CHECK_NAN
						printIfNan(devNewPositions, oldScene->numParticles, "find nan after distance constraint");
					#endif
				}

				// solve all bending constraints
				if (oldScene->numBendings > 0)
				{
					setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), oldScene->numParticles);
					int numBendingBlocks, numBendingThreads;
					GetNumBlocksNumThreads(&numBendingBlocks, &numBendingThreads, oldScene->numBendings);
					bendingConstraints<<<numBendingBlocks, numBendingThreads>>>(devDeltaX,
																				devNewPositions,
																				devInvScaledMasses,
																				devBendings,
																				devMapToNewIds,
																				oldScene->numBendings);
					accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, oldScene->numParticles);
				}

				// solve all rigidbody constraints
				if (oldScene->numRigidBodies > 0)
				{ 
					shapeMatchingAlphaOne<<<oldScene->numRigidBodies, NUM_MAX_PARTICLE_PER_RIGID_BODY>>>(devRigidBodyRotations,
																									  devRigidBodyCMs,
																									  devNewPositions,
																									  devMapToNewIds,
																									  devRigidBodyInitialPositions,
																									  devRigidBodyParticleIdRange);
				}

				// solve immovalbe constraints
				if (oldScene->numImmovables > 0)
				{
					int numImmovableBlocks, numImmovableThreads;
					GetNumBlocksNumThreads(&numImmovableBlocks, &numImmovableThreads, oldScene->numImmovables);
					immovableConstraints<<<numImmovableBlocks, numImmovableThreads>>>(devNewPositions,
																					  devPositions,
																					  devImmovables,
																					  devMapToNewIds,
																					  oldScene->numImmovables);
				}
				*/
			} // end projecting constraints

			updateVelocity<<<numBlocks, numThreads>>>(devVelocities,
													  devNewPositions,
													  devPositions,
													  numParticles,
													  1.0f / subDeltaTime);

			updatePositions<<<numBlocks, numThreads>>>(devPositions, devNewPositions, devPhases, PARTICLE_SLEEPING_EPSILON, numParticles);

			// vorticity confinement part 1.
			fluidOmega<<<numBlocks, numThreads>>>(devFluidOmegas,
												  devVelocities,
												  devNewPositions,
												  devPhases,
												  devCellStart,
												  devCellEnd,
												  numParticles);

			// vorticity confinement part 2.
			fluidVorticity<<<numBlocks, numThreads>>>(devVelocities,
													  devFluidOmegas,
													  devNewPositions,
													  0.001f, // epsilon in eq. 16
													  devPhases,
													  devCellStart,
													  devCellEnd,
													  numParticles,
													  subDeltaTime);

			if (useAkinciCohesionTension)
			{ 
				// fluid normal for Akinci cohesion
				fluidNormal<<<numBlocks, numThreads>>>(devFluidNormals,
													   devNewPositions,
													   devFluidDensities,
													   devPhases,
													   devCellStart,
													   devCellEnd,
													   numParticles);

				/// TODO:: if fluid particle stuck inside solid particle then solid particle can float!
				fluidAkinciTension<<<numBlocks, numThreads>>>(devTempFloat3,
															  devVelocities,
															  devNewPositions,
															  devFluidNormals,
															  devFluidDensities,
															  fluidRestDensity,
															  devPhases,
															  0.5, // tension strength
															  devCellStart,
															  devCellEnd,
															  numParticles,
															  subDeltaTime);
				std::swap(devVelocities, devTempFloat3);
			}

			// xsph
			fluidXSph<<<numBlocks, numThreads>>>(devTempFloat3,
												 devVelocities,
												 devNewPositions,
												 0.0002f, // C in eq. 17
												 devPhases,
												 devCellStart,
												 devCellEnd,
												 numParticles);
			std::swap(devVelocities, devTempFloat3);

			resetGrid(numBlocks, numThreads);
		} // end loop over substeps

		// we need to make picked particle immovable
		if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < numParticles)
		{
            int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
			glm::vec3 solvedPickedParticlePosition = getParticlePosition(pickedNewParticleId);
			setParticle(devPositions, pickedNewParticleId, solvedPickedParticlePosition, pickedParticleVelocity);
		}
	}

	/// TODO:: implement object's destroyer
	size_t		numParticles;
	float		solidRadius;

	float3 *	devNewPositions;
	float3 *	devPositions;
	float3 *	devVelocities;
	float *		devMasses;
	int *		devPhases;
	int *		devMapToOriginalIds;
	int *		devMapToNewIds;

	float3 *	devSortedNewPositions;
	float3 *	devSortedPositions;
	float3 *	devSortedVelocities;
	float *		devSortedMasses;
	int *		devSortedPhases;
	int *		devSortedMapToOriginalIds;

	float		fluidKernelRadius;
	float		fluidRestDensity;

	float3 *	devDeltaX;
	float3 *	devTempFloat3;
	float *		devInvMasses;
	float *		devInvScaledMasses;
	int *		devSolidPhaseCounter;
	int			fluidPhaseCounter;

	float *		devFluidLambdas;
	float *		devFluidDensities;
	float3 *	devFluidNormals;
	float3 *	devFluidOmegas;

	int *		devSortedCellId;
	int *		devSortedParticleId;

	int			numRigidbody;
	int2 *		devRigidBodyParticleIdRange;
	float3 *	devRigidBodyInitialPositions;
	quaternion * devRigidBodyRotations;
	float3 *	devRigidBodyCMs;// center of mass

	int			numDistanceConstraints;
	int2 *		devDistancePairs;
	float2 *	devDistanceParams;

	int			numBendingConstraints;
	int4 *		devBendings;

	int			numWindFaces;
	int3 *		devWindFaces;

	int			numImmovables;
	int *		devImmovables;

	void *		devTempStorage = nullptr;
	size_t		devTempStorageSize = 0;

	// for rendering particles
	// can use group id to identify different group of object
	int *			devGroupIds;
	int *			devSortedGroupIds; 
	int				groupIdCounter = 0;

	int *			devCellId;
	int *			devParticleId;
	int *			devCellStart;
	int *			devCellEnd;
	const float3	cellOrigin;
	const float3	cellSize;
	const int3		gridSize;

	std::shared_ptr<OldSceneFormat> oldScene;
	std::vector<Plane> planes;
};
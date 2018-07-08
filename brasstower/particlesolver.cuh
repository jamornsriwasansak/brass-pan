#pragma once

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
		scene(scene),
		cellOrigin(make_float3(-4.01, -1.01, -5.01)),
		cellSize(make_float3(scene->radius * 2.3f)),
		gridSize(make_int3(512))
	{
		fluidKernelRadius = 2.3f * scene->radius;
		SetKernelRadius(fluidKernelRadius);
		fluidPhaseCounter = -1;

		// alloc particle vars
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
		if (particleIndex < 0 || particleIndex >= scene->numParticles) return glm::vec3(0.0f);
		float3 * tmp = (float3 *)malloc(sizeof(float3));
		cudaMemcpy(tmp, devPositions + particleIndex, sizeof(float3), cudaMemcpyDeviceToHost);
		glm::vec3 result(tmp->x, tmp->y, tmp->z);
		free(tmp);
		return result;
	}

	void setParticle(float3 * devPositions, const int particleIndex, const glm::vec3 & position, const glm::vec3 & velocity)
	{
		if (particleIndex < 0 || particleIndex >= scene->numParticles) return;
		setDevArr_float3<<<1, 1>>>(devPositions + particleIndex, make_float3(position.x, position.y, position.z), 1);
		setDevArr_float3<<<1, 1>>>(devVelocities + particleIndex, make_float3(velocity.x, velocity.y, velocity.z), 1);
	}

	void addGranulars(const std::vector<glm::vec3> & positions, const float massPerParticle)
	{
		int numParticles = positions.size();
		if (scene->numParticles + numParticles >= scene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + scene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   massPerParticle,
												   numParticles);	
		// set phases
		setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
															  devSolidPhaseCounter,
															  1,
															  numParticles);
		scene->numParticles += numParticles;
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

		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   massPerParticle,
												   numParticles);
		// set phases
		setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
													   devSolidPhaseCounter,
													   numParticles);
		// set range for particle id
		setDevArr_int2<<<1, 1>>>(devRigidBodyParticleIdRange + scene->numRigidBodies,
								 make_int2(scene->numParticles, scene->numParticles + numParticles),
								 1);
		// increment phase counter
		increment<<<1, 1>>>(devSolidPhaseCounter);
		
		scene->numParticles += numParticles;
		scene->numRigidBodies += 1;
	}

	void addFluids(const std::vector<glm::vec3> & positions, const float massPerParticle)
	{
		int numParticles = positions.size();
		if (scene->numParticles + numParticles >= scene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + scene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   massPerParticle,
												   numParticles);	
		// fluid phase is always < 0
		setDevArr_int<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
												 fluidPhaseCounter,
												 numParticles);
		fluidPhaseCounter -= 1;
		scene->numParticles += numParticles;
	}

	// spaghetti
    void addNoodles(const std::vector<glm::vec3> & positions, const std::vector<glm::int2> & distancePairs, const std::vector<glm::vec2> & distanceParams, const float massPerParticle, const bool doSelfCollide = true)
    {
        int numParticles = positions.size();
        if (scene->numParticles + numParticles >= scene->numMaxParticles)
        {
            std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
            throw std::exception(message.c_str());
        }

        int numDistanceConstraints = distancePairs.size();
        if (scene->numParticles + numDistanceConstraints >= scene->numMaxDistancePairs)
        {
            std::string message = std::string(__FILE__) + std::string("num distance pairs exceed num max distance pairs");
            throw std::exception(message.c_str());
        }
    
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + scene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   massPerParticle,
												   numParticles);	
		if (doSelfCollide)
		{ 
			// set phases
			setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
																  devSolidPhaseCounter,
																  1,
																  numParticles);
		}
		else
		{ 
			// set phases
			setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
														   devSolidPhaseCounter,
														   numParticles);
			// increment phase counter
			increment<<<1, 1>>>(devSolidPhaseCounter);
		}

		// set distance constraints
		checkCudaErrors(cudaMemcpy(devDistancePairs + scene->numDistancePairs,
								   &(distancePairs[0].x),
								   numDistanceConstraints * sizeof(int) * 2,
								   cudaMemcpyHostToDevice));
		int numDistancePairsBlock, numDistancePairsThreads;
		GetNumBlocksNumThreads(&numDistancePairsBlock, &numDistancePairsThreads, numDistanceConstraints);
		accDevArr_int2<<<numDistancePairsBlock, numDistancePairsThreads>>>(devDistancePairs + scene->numDistancePairs,
																		   make_int2(scene->numParticles),
																		   numDistanceConstraints);
		checkCudaErrors(cudaMemcpy(devDistanceParams + scene->numDistancePairs,
								   &(distanceParams[0]),
								   numDistanceConstraints * sizeof(float) * 2,
								   cudaMemcpyHostToDevice));

		scene->numParticles += numParticles;
		scene->numDistancePairs += numDistanceConstraints;
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
        if (scene->numParticles + numParticles >= scene->numMaxParticles)
        {
            std::string message = std::string(__FILE__) + std::string("num particles exceed num max particles");
            throw std::exception(message.c_str());
        }

        int numDistanceConstraints = distancePairs.size();
        if (scene->numDistancePairs + numDistanceConstraints >= scene->numMaxDistancePairs)
        {
            std::string message = std::string(__FILE__) + std::string("num distance pairs exceed num max distance pairs");
            throw std::exception(message.c_str());
        }
    
		int numBendings = bendings.size();
		if (scene->numBendings + numBendings >= scene->numMaxBendings)
		{
            std::string message = std::string(__FILE__) + std::string("num bendings exceed num max bendings");
            throw std::exception(message.c_str());
		}

		int numFaces = faces.size();
		if (scene->numWindFaces + numFaces >= scene->numMaxWindFaces)
		{
            std::string message = std::string(__FILE__) + std::string("num faces exceed num max faces");
            throw std::exception(message.c_str());
		}

		int numImmovables = immovables.size();
		if (scene->numImmovables + numImmovables >= scene->numMaxImmovables)
		{
            std::string message = std::string(__FILE__) + std::string("num immovables exceed num max immovables");
            throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		// set positions
		checkCudaErrors(cudaMemcpy(devPositions + scene->numParticles,
								   &(positions[0].x),
								   numParticles * sizeof(float) * 3,
								   cudaMemcpyHostToDevice));
		// set masses
		setDevArr_float<<<numBlocks, numThreads>>>(devMasses + scene->numParticles,
												   massPerParticle,
												   numParticles);	
		if (doSelfCollide)
		{ 
			// set phases
			setDevArr_counterIncrement<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
																  devSolidPhaseCounter,
																  1,
																  numParticles);
		}
		else
		{ 
			// set phases
			setDevArr_devIntPtr<<<numBlocks, numThreads>>>(devPhases + scene->numParticles,
														   devSolidPhaseCounter,
														   numParticles);
			// increment phase counter
			increment<<<1, 1>>>(devSolidPhaseCounter);
		}

		// set distance constraints
		checkCudaErrors(cudaMemcpy(devDistancePairs + scene->numDistancePairs,
								   &(distancePairs[0].x),
								   numDistanceConstraints * sizeof(int) * 2,
								   cudaMemcpyHostToDevice));
		int numDistancePairsBlock, numDistancePairsThreads;
		GetNumBlocksNumThreads(&numDistancePairsBlock, &numDistancePairsThreads, numDistanceConstraints);
		accDevArr_int2<<<numDistancePairsBlock, numDistancePairsThreads>>>(devDistancePairs + scene->numDistancePairs,
																		   make_int2(scene->numParticles),
																		   numDistanceConstraints);
		checkCudaErrors(cudaMemcpy(devDistanceParams + scene->numDistancePairs,
								   &(distanceParams[0].x),
								   numDistanceConstraints * sizeof(float) * 2,
								   cudaMemcpyHostToDevice));

		int numBendingBlocks, numBendingThreads;
		GetNumBlocksNumThreads(&numBendingBlocks, &numBendingThreads, numBendings);
		// set bending constraints
		checkCudaErrors(cudaMemcpy(devBendings + scene->numBendings,
								   &(bendings[0].x),
								   numBendings * sizeof(int4),
								   cudaMemcpyHostToDevice));
		accDevArr_int4<<<numBendingBlocks, numBendingThreads>>>(devBendings + scene->numBendings,
															    make_int4(scene->numParticles),
															    numBendings);

		// add faces
		int numWindFaceBlocks, numWindFaceThreads;
		GetNumBlocksNumThreads(&numWindFaceBlocks, &numWindFaceThreads, numFaces);
		checkCudaErrors(cudaMemcpy(devWindFaces + scene->numWindFaces,
								   &(faces[0].x),
								   numFaces * sizeof(int3),
								   cudaMemcpyHostToDevice));
		accDevArr_int3<<<numWindFaceBlocks, numWindFaceThreads>>>(devWindFaces + scene->numWindFaces,
																  make_int3(scene->numParticles),
																  numFaces);


		// add immovable constraints
		if (numImmovables > 0)
		{
			int numImmovableBlocks, numImmovableThreads;
			GetNumBlocksNumThreads(&numImmovableBlocks, &numImmovableThreads, numImmovables);
			checkCudaErrors(cudaMemcpy(devImmovables + scene->numImmovables,
									   &(immovables[0]),
									   numImmovables * sizeof(int),
									   cudaMemcpyHostToDevice));
			accDevArr_int<<<numImmovableBlocks, numImmovableThreads>>>(devImmovables + scene->numImmovables,
																	   scene->numParticles,
																	   numImmovables);
		}

		scene->numImmovables += numImmovables;
		scene->numBendings += numBendings;
		scene->numParticles += numParticles;
		scene->numDistancePairs += numDistanceConstraints;
		scene->numWindFaces += numFaces;
    }

	void updateGrid(int numBlocks, int numThreads)
	{
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
		findStartEndId<<<numBlocks, numThreads>>>(devCellStart, devCellEnd, devSortedCellId, scene->numParticles);

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
														scene->numParticles);

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
		resetStartEndId<<<numBlocks, numThreads>>>(devCellStart, devCellEnd, devSortedCellId, scene->numParticles);
	}

	void update(const int numSubTimeStep,
				const float deltaTime,
				const int pickedOriginalParticleId = -1,
				const glm::vec3 & pickedParticlePosition = glm::vec3(0.0f),
				const glm::vec3 & pickedParticleVelocity = glm::vec3(0.0f))
	{
		if (scene->numParticles <= 0) return;

		float subDeltaTime = deltaTime / (float)numSubTimeStep;
		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, scene->numParticles);

		int3 fluidGridSearchOffset = make_int3(ceil(make_float3(fluidKernelRadius) / cellSize));
		bool useAkinciCohesionTension = true;

		for (int i = 0;i < numSubTimeStep;i++)
		{ 
			// we need to make picked particle immovable
			if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < scene->numParticles)
			{
                int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
				setParticle(devPositions, pickedNewParticleId, pickedParticlePosition, glm::vec3(0.0f));
			}

			predictPositions<<<numBlocks, numThreads>>>(devNewPositions,
														devPositions,
														devVelocities,
														scene->numParticles,
														subDeltaTime);

			applyGravity<<<numBlocks, numThreads>>>(devNewPositions,
													scene->numParticles,
													subDeltaTime);

			// compute grid
			updateGrid(numBlocks, numThreads);

			inverseMapping<<<numBlocks, numThreads>>>(devMapToNewIds, devMapToOriginalIds, scene->numParticles);

			// compute scaled masses
			computeInvScaledMasses<<<numBlocks, numThreads>>>(devInvScaledMasses,
															  devMasses,
															  devPositions,
															  MASS_SCALING_CONSTANT,
															  scene->numParticles);
			#ifdef CHECK_NAN
				printIfNan(devInvScaledMasses, scene->numParticles, "find nan after InvScaledMasses");
			#endif

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
			#ifdef CHECK_NAN
				printIfNan(devNewPositions, scene->numParticles, "fina nan after plane stabilize collision");
			#endif

			// apply Wind Force
			if (scene->numWindFaces > 0)
			{
				int numWindFaceBlocks, numWindFaceThreads;
				GetNumBlocksNumThreads(&numWindFaceBlocks, &numWindFaceThreads, scene->numWindFaces);
				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), scene->numParticles);
				applyWindForce<<<numWindFaceBlocks, numWindFaceThreads>>>(devDeltaX,
																		  devNewPositions,
																		  devVelocities,
																		  devMasses,
																		  devMapToNewIds,
																		  devWindFaces,
																		  scene->numWindFaces,
																		  subDeltaTime);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, scene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, scene->numParticles, "find nan after wind force applied");
				#endif
			}

			// projecting constraints iterations
			// (update grid every n iterations)
			for (int j = 0; j < 2;j++)
			{
				// we need to make picked particle immovable
				if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < scene->numParticles)
				{
					int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
					setParticle(devNewPositions, pickedNewParticleId, pickedParticlePosition, glm::vec3(0.0f));
				}

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
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, scene->numParticles, "find nan after particle plane collision");
				#endif

				// solving all particles collisions
				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), scene->numParticles);
				particleParticleCollisionConstraint<<<numBlocks, numThreads>>>(devDeltaX,
																			   devNewPositions,
																			   devPositions,
																			   devInvScaledMasses,
																			   devPhases,
																			   devCellStart,
																			   devCellEnd,
																			   scene->numParticles,
																			   scene->radius);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, scene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, scene->numParticles, "find nan after particle particle collision");
				#endif

				// fluid
				fluidLambda<<<numBlocks, numThreads>>>(devFluidLambdas,
													   devFluidDensities,
													   devNewPositions,
													   devMasses,
													   devPhases,
													   fluidRestDensity,
													   1.0f, // solid density scaling
													   300.0f, // relaxation parameter
													   devCellStart,
													   devCellEnd,
													   scene->numParticles,
													   useAkinciCohesionTension);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, scene->numParticles, "find nan after fluid lambda");
				#endif

				setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), scene->numParticles);
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
														 scene->numParticles,
														 useAkinciCohesionTension);
				accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, scene->numParticles);
				#ifdef CHECK_NAN
					printIfNan(devNewPositions, scene->numParticles, "find nan after fluid position");
				#endif

				// solve all distance constraints
				if (scene->numDistancePairs > 0)
				{ 
					setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), scene->numParticles);
					int numDistanceBlocks, numDistanceThreads;
					GetNumBlocksNumThreads(&numDistanceBlocks, &numDistanceThreads, scene->numDistancePairs);
					distanceConstraints<<<numDistanceBlocks, numDistanceThreads>>>(devDeltaX,
																				   devNewPositions,
																				   devInvScaledMasses,
																				   devDistancePairs,
																				   devDistanceParams,
																				   devMapToNewIds,
																				   scene->numDistancePairs);
					accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, scene->numParticles);
					#ifdef CHECK_NAN
						printIfNan(devNewPositions, scene->numParticles, "find nan after distance constraint");
					#endif
				}

				// solve all bending constraints
				if (scene->numBendings > 0)
				{
					setDevArr_float3<<<numBlocks, numThreads>>>(devDeltaX, make_float3(0.f), scene->numParticles);
					int numBendingBlocks, numBendingThreads;
					GetNumBlocksNumThreads(&numBendingBlocks, &numBendingThreads, scene->numBendings);
					bendingConstraints<<<numBendingBlocks, numBendingThreads>>>(devDeltaX,
																				devNewPositions,
																				devInvScaledMasses,
																				devBendings,
																				devMapToNewIds,
																				scene->numBendings);
					accDevArr_float3<<<numBlocks, numThreads>>>(devNewPositions, devDeltaX, scene->numParticles);
				}

				// solve all rigidbody constraints
				if (scene->numRigidBodies > 0)
				{ 
					shapeMatchingAlphaOne<<<scene->numRigidBodies, NUM_MAX_PARTICLE_PER_RIGID_BODY>>>(devRigidBodyRotations,
																									  devRigidBodyCMs,
																									  devNewPositions,
																									  devMapToNewIds,
																									  devRigidBodyInitialPositions,
																									  devRigidBodyParticleIdRange);
				}

				// solve immovalbe constraints
				if (scene->numImmovables > 0)
				{
					int numImmovableBlocks, numImmovableThreads;
					GetNumBlocksNumThreads(&numImmovableBlocks, &numImmovableThreads, scene->numImmovables);
					immovableConstraints<<<numImmovableBlocks, numImmovableThreads>>>(devNewPositions,
																					  devPositions,
																					  devImmovables,
																					  devMapToNewIds,
																					  scene->numImmovables);
				}
			} // end projecting constraints

			updateVelocity<<<numBlocks, numThreads>>>(devVelocities,
													  devNewPositions,
													  devPositions,
													  scene->numParticles,
													  1.0f / subDeltaTime);

			updatePositions<<<numBlocks, numThreads>>>(devPositions, devNewPositions, devPhases, PARTICLE_SLEEPING_EPSILON, scene->numParticles);

			// vorticity confinement part 1.
			fluidOmega<<<numBlocks, numThreads>>>(devFluidOmegas,
												  devVelocities,
												  devNewPositions,
												  devPhases,
												  devCellStart,
												  devCellEnd,
												  scene->numParticles);

			// vorticity confinement part 2.
			fluidVorticity<<<numBlocks, numThreads>>>(devVelocities,
													  devFluidOmegas,
													  devNewPositions,
													  0.001f, // epsilon in eq. 16
													  devPhases,
													  devCellStart,
													  devCellEnd,
													  scene->numParticles,
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
													   scene->numParticles);

				fluidAkinciTension<<<numBlocks, numThreads>>>(devTempFloat3,
															  devVelocities,
															  devNewPositions,
															  devFluidNormals,
															  devFluidDensities,
															  fluidRestDensity,
															  devPhases,
															  0.1, // tension strength
															  devCellStart,
															  devCellEnd,
															  scene->numParticles,
															  deltaTime);
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
												 scene->numParticles);
			std::swap(devVelocities, devTempFloat3);

			resetGrid(numBlocks, numThreads);
		} // end loop over substeps

		// we need to make picked particle immovable
		if (pickedOriginalParticleId >= 0 && pickedOriginalParticleId < scene->numParticles)
		{
            int pickedNewParticleId = queryNewParticleId(pickedOriginalParticleId);
			glm::vec3 solvedPickedParticlePosition = getParticlePosition(pickedNewParticleId);
			setParticle(devPositions, pickedNewParticleId, solvedPickedParticlePosition, pickedParticleVelocity);
		}
	}

	/// TODO:: implement object's destroyer

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
	float *		devInvScaledMasses;
	int *		devSolidPhaseCounter;
	int			fluidPhaseCounter;

	float *		devFluidLambdas;
	float *		devFluidDensities;
	float3 *	devFluidNormals;
	float3 *	devFluidOmegas;

	int *		devSortedCellId;
	int *		devSortedParticleId;

	int2 *		devRigidBodyParticleIdRange;
	float3 *	devRigidBodyInitialPositions;
	quaternion * devRigidBodyRotations;
	float3 *	devRigidBodyCMs;// center of mass

	int2 *		devDistancePairs;
	float2 *	devDistanceParams;

	int4 *		devBendings;

	int3 *		devWindFaces;

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

	std::shared_ptr<Scene> scene;
};
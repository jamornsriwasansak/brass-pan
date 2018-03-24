#include <exception>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <conio.h>

#ifndef __INTELLISENSE__
#include <cub/cub.cuh>
#endif

#include "cuda/helper.cuh"
#include "cuda/cudaglm.cuh"
#include "scene.h"

#define NUM_MAX_PARTICLE_PER_CELL 4

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

__global__ void initializeDevFloat(float * devArr,
								   const int numParticles,
								   const float value,
								   const int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	devArr[i + offset] = value;
}

__global__ void initializeDevFloat3(float3 * devArr,
									const int numParticles,
									const float3 val,
									const int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	devArr[i + offset] = val;
}

__global__ void initializeBlockPosition(float3 * positions,
										const int numParticles,
										const int3 dimension,
										const float3 startPosition,
										const float3 step,
										const int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	int x = i % dimension.x;
	int y = (i / dimension.x) % dimension.y;
	int z = i / (dimension.x * dimension.y);
	positions[i + offset] = make_float3(x, y, z) * step + startPosition;
}

// GRID //
// from White Paper "Particles" by SIMON GREEN 

__device__ int3 calcGridPos(float3 position, float3 origin, float3 cellSize)
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

__global__ void updateGridId(int * gridIds,
							 int * particleIds,
							 float3 * positions,
							 float3 cellOrigin,
							 float3 cellSize,
							 int3 gridSize,
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
							int * sortedGridIds,
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

__global__ void applyForces(float3 * velocities,
							float * invMass,
							const int numParticles,
							const float deltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	velocities[i] += make_float3(0.0f, -9.8f, 0.0f) * deltaTime;
}

__global__ void predictPositions(float3 * newPositions,
								 float3 * positions,
								 float3 * velocities,
								 const int numParticles,
								 const float deltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	newPositions[i] = positions[i] + velocities[i] * deltaTime;
}

__global__ void updateVelocity(float3 * velocities,
							   float3 * newPositions,
							   float3 * positions,
							   const int numParticles,
							   const float invDeltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	velocities[i] = (newPositions[i] - positions[i]) * invDeltaTime;
}

__global__ void planeStabilize(float3 * positions,
							   float3 * newPositions,
							   const int numParticles,
							   float3 planeOrigin,
							   float3 planeNormal,
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

__global__ void particlePlaneCollisionConstraint(float3 * newPositions,
												 float3 * positions,
												 const int numParticles,
												 float3 planeOrigin,
												 float3 planeNormal,
												 const float radius)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	float3 origin2position = planeOrigin - newPositions[i];
	float distance = dot(origin2position, planeNormal) + radius;
	if (distance <= 0) { return; }

	float diffPosition = dot(newPositions[i] - positions[i], planeNormal);
	newPositions[i] += distance * planeNormal;
	positions[i] += (2.0f * diffPosition + distance) * planeNormal / 10.0f;
}

__global__ void checkIncorrectGridId(int* cellIds,
									 int* particleIds,
									 float3 * positions,
									 float3 cellOrigin,
									 float3 cellSize,
									 int3 gridSize,
									 const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }

	int cellId = cellIds[i];
	int particleId = particleIds[i];

	float3 position = positions[particleId];
	int3 gridPos = calcGridPos(position, cellOrigin, cellSize);
	int gridAddress = calcGridAddress(gridPos, gridSize);

	if (gridAddress != cellId)
	{
		printf("checkIncorrectGridId: error at %d (%d vs %d)\n", i, cellId, gridAddress);
	}
}

__global__ void checkEqual(int* a, int * b, const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	if (a[i] != b[i])
	{
		printf("checkEqual: error at %d (%d vs %d)\n", i, a[i], b[i]);
	}
}

__global__ void particleParticleCollisionConstraint(float3 * newPositionsNext,
													float3 * newPositionsPrev,
													int* sortedCellId,
													int* sortedParticleId,
													int* cellStart, 
													float3 cellOrigin,
													float3 cellSize,
													int3 gridSize,
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

					if (i != particleId2)
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

struct ParticleSolver
{
	ParticleSolver(const std::shared_ptr<Scene> & scene):
		scene(scene),
		cellOrigin(make_float3(-4, -1, -5)),
		cellSize(make_float3(scene->radius * 2.0f)),
		gridSize(make_int3(128))
	{
		checkCudaErrors(cudaMalloc(&devPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devTempNewPositions, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devVelocities, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devInvMasses, scene->numMaxParticles * sizeof(float)));

		checkCudaErrors(cudaMalloc(&devCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedCellId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devSortedParticleId, scene->numMaxParticles * sizeof(int)));
		checkCudaErrors(cudaMalloc(&devCellStart, gridSize.x * gridSize.y * gridSize.z * sizeof(int)));

		checkCudaErrors(cudaMemset(devVelocities, 0, scene->numMaxParticles * sizeof(float3)));
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

	void addParticles(const glm::ivec3 & dimension, const glm::vec3 & startPosition, const glm::vec3 & step, const float mass)
	{
		int numParticles = dimension.x * dimension.y * dimension.z;

		// check if number of particles exceed num max particles or not
		if (scene->numParticles + numParticles > scene->numMaxParticles)
		{
			std::string message = std::string(__FILE__) + std::string(" num particles exceed num max particles");
			throw std::exception(message.c_str());
		}

		int numBlocks, numThreads;
		GetNumBlocksNumThreads(&numBlocks, &numThreads, numParticles);

		initializeDevFloat<<<numBlocks, numThreads>>>(devInvMasses,
													  numParticles,
													  1.0f / mass,
													  scene->numParticles);
		initializeBlockPosition<<<numBlocks, numThreads>>>(devPositions,
														   numParticles,
														   make_int3(dimension),
														   make_float3(startPosition),
														   make_float3(step),
														   scene->numParticles);
		scene->numParticles += numParticles;
	}

	void updateGrid(int numBlocks, int numThreads)
	{
		checkCudaErrors(cudaMemset(devCellStart, 0, scene->numMaxParticles * sizeof(int)));

		updateGridId<<<numBlocks, numThreads>>>(devCellId,
												devParticleId,
												devNewPositions,
												cellOrigin,
												cellSize,
												gridSize,
												scene->numParticles);

	#if 0 // for debugging
		printf("before sort\n");
		checkIncorrectGridId<<<numBlocks, numThreads>>>(devCellId, devParticleId, devNewPositions, cellOrigin, cellSize, gridSize, scene->numParticles);
		checkCudaLastErrors();
		cudaDeviceSynchronize();
	#endif

		size_t tempStorageSize = 0;
		cub::DeviceRadixSort::SortPairs(devTempStorage,
										tempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										scene->numParticles);
		updateTempStorageSize(tempStorageSize);
		cub::DeviceRadixSort::SortPairs(devTempStorage,
										devTempStorageSize,
										devCellId,
										devSortedCellId,
										devParticleId,
										devSortedParticleId,
										scene->numParticles);

	#if 0 // for debugging
		printf("after sort\n");
		checkIncorrectGridId << <numBlocks, numThreads >> >(devCellId, devParticleId, devNewPositions, cellOrigin, cellSize, gridSize, scene->numParticles);
		checkCudaLastErrors();
		cudaDeviceSynchronize();
	#endif

		findStartId<<<numBlocks, numThreads>>>(devCellStart, devSortedCellId, scene->numParticles);
	}

	void update(const int numSubTimeStep, const float deltaTime)
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
																				   devSortedCellId,
																				   devSortedParticleId,
																				   devCellStart,
																				   cellOrigin,
																				   cellSize,
																				   gridSize,
																				   scene->numParticles,
																				   scene->radius);
					std::swap(devTempNewPositions, devNewPositions);
				}
			}

			updateVelocity<<<numBlocks, numThreads>>>(devVelocities,
													  devNewPositions,
													  devPositions,
													  scene->numParticles,
													   1.0f / subDeltaTime);
			std::swap(devNewPositions, devPositions); // update position

		}
	}

	/// TODO:: implement object's destroyer

	float3 * devPositions;
	float3 * devNewPositions;
	float3 * devTempNewPositions;
	float3 * devVelocities;
	float  * devInvMasses;

	int * devCellId;
	int * devParticleId;
	int * devCellStart;

	int * devSortedCellId;
	int * devSortedParticleId;

	void * devTempStorage = nullptr;
	size_t devTempStorageSize = 0;

	const float3 cellOrigin;
	const float3 cellSize;
	const int3 gridSize;
	std::shared_ptr<Scene> scene;
};
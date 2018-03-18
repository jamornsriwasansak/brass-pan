#include <exception>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "cuda/helper.cuh"
#include "cuda/cudaglm.cuh"
#include "scene.h"

void print(float3 * dev, int size)
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


void print(float * dev, int size)
{
	float * tmp = (float *)malloc(sizeof(float) * size);
	cudaMemcpy(tmp, dev, sizeof(float) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << tmp[i];
		if (i != size - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
	free(tmp);
}

// PARTICLE SYSTEM //

__global__ void initializeDevFloat(float * devArr, float value, int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	devArr[i + offset] = value;
}

__global__ void initializeDevFloat3(float3 * devArr, float3 val, int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	devArr[i + offset] = val;
}

__global__ void initializeBlockPosition(float3 * position, int3 dimension, float3 startPosition, float3 step, int offset)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int x = i % dimension.x;
	int y = (i / dimension.x) % dimension.y;
	int z = i / (dimension.x * dimension.y);
	position[i + offset] = make_float3(x, y, z) * step + startPosition;
}

// SOLVER //

__global__ void applyForces(float3 * velocity, float * invMass)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	velocity[i] += make_float3(0.0f, -9.8f, 0.0f);
}

__global__ void predictPositions(float3 * newPosition, float3 * position, float3 * velocity, float deltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	newPosition[i] = position[i] + velocity[i] * deltaTime;
}

__global__ void updateVelocity(float3 * velocity, float3 * newPosition, float3 * position, float invDeltaTime)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	velocity[i] = (newPosition[i] - position[i]) * invDeltaTime;
}

__global__ void projectStaticPlane()
{

}

struct ParticleSolver
{
	const float time = 1.0f / 1000000.0f;

	ParticleSolver(const std::shared_ptr<Scene> & scene):
		scene(scene)
	{
		checkCudaErrors(cudaMalloc(&devPosition, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devNewPosition, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devVelocity, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMalloc(&devInvMass, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMalloc(&devDelta, scene->numMaxParticles * sizeof(float3)));

		checkCudaErrors(cudaMemset(devPosition, 0, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMemset(devNewPosition, 0, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMemset(devVelocity, 0, scene->numMaxParticles * sizeof(float3)));
		checkCudaErrors(cudaMemset(devInvMass, 0, scene->numMaxParticles * sizeof(float)));
		checkCudaErrors(cudaMemset(devDelta, 0, scene->numMaxParticles * sizeof(float3)));
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

		initializeDevFloat<<<1, numParticles>>>(devInvMass, 1.0f / mass, scene->numParticles);
		initializeBlockPosition<<<1, numParticles>>>(devPosition, make_int3(dimension), make_float3(startPosition), make_float3(step), scene->numParticles);
		scene->numParticles += numParticles;
	}

	void update()
	{
		applyForces<<<1, scene->numParticles>>>(devVelocity, devInvMass);
		predictPositions<<<1, scene->numParticles>>>(devNewPosition, devPosition, devVelocity, time);

		// project

		updateVelocity<<<1, scene->numParticles>>>(devVelocity, devNewPosition, devPosition, 1.0f / time);
		std::swap(devNewPosition, devPosition); // update position
	}

	~ParticleSolver()
	{
		checkCudaErrors(cudaFree(&devPosition));
		checkCudaErrors(cudaFree(&devNewPosition));
		checkCudaErrors(cudaFree(&devVelocity));
		checkCudaErrors(cudaFree(&devInvMass));
		checkCudaErrors(cudaFree(&devDelta));
	}

	// particle system data
	float3 *devPosition;
	float3 *devVelocity;
	float3 *devNewPosition;
	float *devInvMass;

	// 2 buffers
	float3 *devDelta;

	std::shared_ptr<Scene> scene;
};
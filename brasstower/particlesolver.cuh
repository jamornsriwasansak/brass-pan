#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "cuda/helper.cuh"
#include "scene.h"

__global__ void updateGravity()
{
}

struct ParticleSolver
{
	ParticleSolver(const std::shared_ptr<Scene> & scene):
		scene(scene)
	{}

	void update()
	{
	}

	std::shared_ptr<Scene> scene;
};
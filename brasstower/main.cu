//#include "kernel.cuh"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "cuda/helper.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "ext/helper_math.h"
#include "opengl/shader.h"
#include "mesh.h"
#include "scene.h"

#include "particlerenderer.cuh"
#include "particlesolver.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

GLFWwindow * window;
ParticleRenderer * renderer;
ParticleSolver * solver;

void updateControl()
{
	// mouse control
	{
		static bool isHoldingLeftMouse = false;
		static bool isHoldingRightMouse = false;
		static glm::dvec2 prevMousePos = [&]()
		{
			glm::dvec2 mousePos;
			glfwGetCursorPos(window, &mousePos.y, &mousePos.x);
			return mousePos;
		}();

		glm::vec2 rotation(0.0f);
		glm::dvec2 mousePos;
		glfwGetCursorPos(window, &mousePos.y, &mousePos.x);

		//if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)

		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
			rotation += (mousePos - prevMousePos);

		glm::dvec2 diff = (mousePos - prevMousePos); 
		prevMousePos = mousePos;
		renderer->camera.rotate(rotation * glm::vec2(0.005f, 0.005f));
	}

	// key control
	{
		glm::vec3 control(0.0f);
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			control += glm::vec3(0.f, 0.f, 1.f);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			control -= glm::vec3(0.f, 0.f, 1.f);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			control += glm::vec3(1.f, 0.f, 0.f);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			control -= glm::vec3(1.f, 0.f, 0.f);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			control += glm::vec3(0.f, 1.f, 0.f);
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
			control -= glm::vec3(0.f, 1.f, 0.f);
		float length = glm::length(control);
		control = length > 0 ? control / length : control;
		renderer->camera.shift(control * 0.01f);
	}
}

std::shared_ptr<Scene> initSimpleScene()
{
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	scene->planes.push_back(Plane(glm::vec3(0), glm::vec3(0, 1, 0)));
	scene->planes.push_back(Plane(glm::vec3(0, 0, -0.19), glm::normalize(glm::vec3(0, 0, 1))));
	scene->planes.push_back(Plane(glm::vec3(0, 0, 0.19), glm::normalize(glm::vec3(0, 0, -1))));
	scene->planes.push_back(Plane(glm::vec3(-0.28, 0, 0), glm::normalize(glm::vec3(1, 0, 0))));
	scene->planes.push_back(Plane(glm::vec3(0.28, 0, 0), glm::normalize(glm::vec3(-1, 0, 0))));
	//scene->numParticles = 10;
	scene->numMaxParticles = 1000;
	scene->radius = 0.01f;
	return scene;
}

__global__ void mapPositions(float4 * ssboDptr, float3 * position)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	ssboDptr[i] = make_float4(position[i].x, position[i].y, position[i].z, 0.0f);
}

int main()
{
	cudaGLSetGLDevice(0);
	int numParticles = 10;

	float p = 0.0f;

	std::shared_ptr<Scene> scene = initSimpleScene();

	window = InitGL(1280, 720);
	renderer = new ParticleRenderer(glm::uvec2(1280, 720), scene);
	solver = new ParticleSolver(scene);

	solver->addParticles(glm::ivec3(4), glm::vec3(0, 1, 0), glm::vec3(0.01f * 2.0f), 1.0f);

	do
	{
		updateControl();

		// solver update
		solver->update();

		// renderer update
		float4 *dptr = renderer->mapSsbo();
		mapPositions<<<1, scene->numParticles>>>(dptr, solver->devPosition);
		renderer->unmapSsbo();
		renderer->update();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	delete renderer;
	delete solver;
	return 0;
}
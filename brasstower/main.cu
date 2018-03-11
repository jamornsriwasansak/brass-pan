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

const float radius = 0.01;

__global__ void mapPositions(float4 * positions, float p)
{
	int i = threadIdx.x;
	positions[i] = make_float4(0.f, p + i * 10.0f + 0.f, 0.f, 0.0f);
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
		renderer->camera.shift(control * 0.01f);
	}
}

int main()
{
	cudaGLSetGLDevice(0);
	int numParticles = 10;

	float p = 0.0f;

	window = InitGL(1280, 720);
	renderer = new ParticleRenderer(glm::uvec2(1280, 720), radius);
	solver = new ParticleSolver();

	do
	{
		updateControl();

		float4 *dptr = renderer->mapSsbo();
		mapPositions<<<1, numParticles>>>(dptr, p);
		renderer->unmapSsbo();

		//p += 0.1f;

		renderer->update(numParticles);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
}
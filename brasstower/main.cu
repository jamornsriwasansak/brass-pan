#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "cuda/helper.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <ctime>
#include <chrono>

#include "ext/helper_math.h"
#include "opengl/shader.h"
#include "mesh.h"
#include "scene.h"

#include "particlerenderer.cuh"
#include "particlesolver.cuh"

GLFWwindow * window;
ParticleRenderer * renderer;
ParticleSolver * solver;
int currentPickingObject = -1;
glm::vec2 currentPickingObjectScreenCoord;

void updateControl()
{
	// mouse control
	{
		static glm::dvec2 prevMousePos = [&]()
		{
			glm::dvec2 mousePos;
			glfwGetCursorPos(window, &mousePos.x, &mousePos.y);
			return mousePos;
		}();

		glm::vec2 rotation(0.0f);
		glm::dvec2 mousePos;
		glfwGetCursorPos(window, &mousePos.x, &mousePos.y);

		static bool isHoldingLeftMouse = false;
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		{
			int width, height;
			glfwGetWindowSize(window, &width, &height);
			currentPickingObjectScreenCoord = glm::vec2((float)mousePos.x / (float)width, (float)mousePos.y / (float)height);

			if (!isHoldingLeftMouse)
			{
				glm::uvec2 uMousePos(mousePos);
				currentPickingObject = renderer->queryParticleColorCode(uMousePos);
				isHoldingLeftMouse = true;
			}
		}
		else
		{
			currentPickingObject = -1;
			isHoldingLeftMouse = false;
		}

		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
			rotation += (mousePos - prevMousePos);

		prevMousePos = mousePos;
		renderer->camera.rotate(glm::vec2(rotation.y, rotation.x) * glm::vec2(0.005f, 0.005f));
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
		renderer->camera.shift(control * 0.1f);
	}
}

void computePickedParticlePosition(glm::vec3 * particlePosition, glm::vec3 * particleVelocity)
{
	glm::vec3 pickedParticlePosition = solver->getParticlePosition(currentPickingObject);

	// compute space that allows the object to be moved
	float d = glm::dot(pickedParticlePosition - renderer->camera.pos, renderer->camera.dir);
	glm::vec3 spanOrigin = renderer->camera.pos + d * renderer->camera.dir;
	glm::vec3 spanBasisZ = glm::normalize(-renderer->camera.dir);
	glm::vec3 spanBasisX = glm::normalize(glm::cross(glm::vec3(0, 1, 0), spanBasisZ));
	glm::vec3 spanBasisY = glm::normalize(glm::cross(spanBasisZ, spanBasisX));
	float spanYSize = d * std::tan(renderer->camera.fovY * 0.5f);
	float spanXSize = spanYSize * renderer->camera.aspectRatio;
	glm::vec3 newPickedParticlePosition = spanOrigin
		+ (currentPickingObjectScreenCoord.x - 0.5f) * 2.0f * spanBasisX * spanXSize
		+ (0.5f - currentPickingObjectScreenCoord.y) * 2.0f * spanBasisY * spanYSize;

	*particlePosition = newPickedParticlePosition;
	*particleVelocity = (newPickedParticlePosition - pickedParticlePosition) * 60.0f;
}

std::vector<glm::vec3> CreateBoxParticles(const glm::ivec3 & dimension, const glm::vec3 & startPosition, const glm::vec3 & stepSize)
{
	std::vector<glm::vec3> positions;
	for (int i = 0; i < dimension.x; i++)
		for (int j = 0; j < dimension.y; j++)
			for (int k = 0; k < dimension.z; k++)
			{
				positions.push_back(startPosition + stepSize * glm::vec3(i, j, k));
			}
	return positions;
}

std::shared_ptr<Scene> initSimpleScene()
{
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	scene->planes.push_back(Plane(glm::vec3(0), glm::vec3(0, 1, 0)));
	scene->planes.push_back(Plane(glm::vec3(0, 0, -1.9), glm::normalize(glm::vec3(0, 0, 1))));
	scene->planes.push_back(Plane(glm::vec3(0, 0, 1.9), glm::normalize(glm::vec3(0, 0, -1))));
	scene->planes.push_back(Plane(glm::vec3(-2.8, 0, 0), glm::normalize(glm::vec3(1, 0, 0))));
	scene->planes.push_back(Plane(glm::vec3(2.8, 0, 0), glm::normalize(glm::vec3(-1, 0, 0))));
	scene->numParticles = 0;
	scene->numMaxParticles = 50000;
	scene->numRigidBodies = 0;
	scene->numMaxRigidBodies = 128;
	scene->radius = 0.05f;

	scene->pointLight.direction = glm::normalize(glm::vec3(-1, -1, -1));
	scene->pointLight.exponent = 1.0f;
	scene->pointLight.intensity = glm::vec3(50.0f);
	scene->pointLight.position = glm::vec3(2, 2, 2);

	//scene->granulars.push_back(glm::vec3(1, 1, 1));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(glm::vec3(0.8, 0.4, 0.2), glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 2, 0 - scene->radius), glm::vec3(scene->radius * 2.0f)));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(glm::vec3(0.2, 0.7, 0.1), glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 1, 0 - scene->radius), glm::vec3(scene->radius * 2.0f)));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(glm::vec3(0.8, 0.2, 0.5), glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 3, 0 - scene->radius), glm::vec3(scene->radius * 2.0f)));

	return scene;
}

__global__ void mapPositions(float4 * ssboDptr, float3 * position, const int numParticles)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numParticles) { return; }
	ssboDptr[i] = make_float4(position[i].x, position[i].y, position[i].z, 0.0f);
}

__global__ void mapMatrices(matrix4 * matrices, quaternion * quaternions, float3 * CMs, const int numRigidBodies)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numRigidBodies) { return; }
	float3 CM = CMs[i];
	matrices[i] = extract_rotation_matrix4(quaternions[i]);
	matrices[i].col[3] = make_float4(CM.x, CM.y, CM.z, 1.0f);
}

int main()
{
	cudaGLSetGLDevice(0);
	window = InitGL(1280, 720);

	std::shared_ptr<Scene> scene = initSimpleScene();
	renderer = new ParticleRenderer(glm::uvec2(1280, 720), scene);
	solver = new ParticleSolver(scene);

	// fps counter
	std::chrono::high_resolution_clock::time_point lastUpdateTime = std::chrono::high_resolution_clock::now();
	int numFrame = 0;

	do
	{
		updateControl();

		// pick particle 1
		if (currentPickingObject >= 0 && currentPickingObject < scene->numParticles)
		{
			glm::vec3 position, velocity;
			computePickedParticlePosition(&position, &velocity);
			solver->update(2, 1.0f / 60.0f, currentPickingObject, position, velocity);
		}
		else
		{
			// solver update
			solver->update(2, 1.0f / 60.0f);
		}

		// renderer update
		{
			// for particles
			float4 *dptr = renderer->mapParticlePositionsSsbo();
			int numBlocks, numThreads;
			GetNumBlocksNumThreads(&numBlocks, &numThreads, scene->numParticles);
			mapPositions<<<numBlocks, numThreads>>>(dptr,
													solver->devPositions,
													scene->numParticles);
			renderer->unmapParticlePositionsSsbo();
		}
		{
			matrix4 *dptr = renderer->mapMatricesSsbo();
			int numBlocks, numThreads;
			GetNumBlocksNumThreads(&numBlocks, &numThreads, scene->numRigidBodies);
			mapMatrices<<<numBlocks, numThreads>>>(dptr,
												   solver->devRigidBodyRotations,
												   solver->devRigidBodyCMs,
												   scene->numRigidBodies);
			renderer->unmapMatricesSsbo();
		}
		renderer->update();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		// update fps counter
		std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
		long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastUpdateTime).count();
		if (elapsedMs >= 100)
		{
			glfwSetWindowTitle(window, std::to_string((float)elapsedMs / (float)numFrame).c_str());
			numFrame = 0;
			lastUpdateTime = currentTime;
		}

		numFrame++;
	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	delete renderer;
	delete solver;
	return 0;
}
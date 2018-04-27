#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

#include "cuda/helper.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <ctime>
#include <chrono>

#include "global.h"

#include "ext/helper_math.h"
#include "opengl/shader.h"
#include "mesh.h"
#include "scene.h"

#include "particlerenderer.cuh"
#include "particlesolver.cuh"

const float windowWidth = 1280;
const float windowHeight = 720;

std::shared_ptr<Scene> scene;
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
		scene->camera.rotate(glm::vec2(rotation.y, rotation.x) * glm::vec2(0.005f, 0.005f));
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
		scene->camera.shift(control * 0.1f);
	}
}

void computePickedParticlePosition(glm::vec3 * particlePosition, glm::vec3 * particleVelocity)
{
	glm::vec3 pickedParticlePosition = solver->getParticlePosition(currentPickingObject);

	// compute space that allows the object to be moved
	Camera & camera = scene->camera;
	float d = glm::dot(pickedParticlePosition - camera.pos, camera.dir);
	glm::vec3 spanOrigin = camera.pos + d * camera.dir;
	glm::vec3 spanBasisZ = glm::normalize(-camera.dir);
	glm::vec3 spanBasisX = glm::normalize(glm::cross(glm::vec3(0, 1, 0), spanBasisZ));
	glm::vec3 spanBasisY = glm::normalize(glm::cross(spanBasisZ, spanBasisX));
	float spanYSize = d * std::tan(camera.fovY * 0.5f);
	float spanXSize = spanYSize * camera.aspectRatio;
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
	/*scene->planes.push_back(Plane(glm::vec3(0, 0, -1.9), glm::normalize(glm::vec3(0, 0, 1))));
	scene->planes.push_back(Plane(glm::vec3(0, 0, 1.9), glm::normalize(glm::vec3(0, 0, -1))));
	scene->planes.push_back(Plane(glm::vec3(-2.8, 0, 0), glm::normalize(glm::vec3(1, 0, 0))));
	scene->planes.push_back(Plane(glm::vec3(2.8, 0, 0), glm::normalize(glm::vec3(-1, 0, 0))));*/
	scene->numParticles = 0;
	scene->numMaxParticles = 50000;
	scene->numRigidBodies = 0;
	scene->numMaxRigidBodies = 128;
	scene->radius = 0.05f;

	scene->pointLight.intensity = glm::vec3(5.0f);
	scene->pointLight.position = glm::vec3(1, 5, 1);
	scene->pointLight.direction = glm::normalize(-scene->pointLight.position);

	scene->camera = Camera(glm::vec3(0, 5, 7), glm::vec3(0, 2, 0), glm::radians(55.0f), (float)windowWidth / (float)windowHeight),

	//scene->granulars.push_back(glm::vec3(1, 1, 1));
	scene->granulars.push_back(Granulars::CreateGranularsBlock(glm::ivec3(30, 50, 30), glm::vec3(-6, scene->radius, -6), glm::vec3(scene->radius * 2.0f), 1.0f));
	/*scene->rigidBodies.push_back(RigidBody::CreateRigidBox(OxbloodColor, glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 2, 0 - scene->radius), glm::vec3(scene->radius * 2.0f), 2.0f));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(BlackBoardColor, glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 1, 0 - scene->radius), glm::vec3(scene->radius * 2.0f), 1.5f));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(GrainColor, glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 3, 0 - scene->radius), glm::vec3(scene->radius * 2.0f), 1.0f));
	scene->rigidBodies.push_back(RigidBody::CreateRigidBox(TanColor, glm::ivec3(3, 4, 2), glm::vec3(0 - scene->radius, scene->radius + 4, 0 - scene->radius), glm::vec3(scene->radius * 2.0f), 0.5f));*/

	return scene;
}

std::shared_ptr<Scene> initFluidScene()
{
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	scene->radius = 0.05f;

	const unsigned int width = 15;
	const unsigned int depth = 15;
	const unsigned int height = 40;
	const float containerWidth = (width + 1) * scene->radius * 2.0 * 5.0;
	const float containerDepth = (depth + 1) * scene->radius * 2.0;
	const float containerHeight = 4.0;
	const float diam = 2.0 * scene->radius;
	const float startX = -0.5*containerWidth + diam;
	const float startY = diam;
	const float startZ = -0.5*containerDepth + diam;

	scene->planes.push_back(Plane(glm::vec3(0), glm::vec3(0, 1, 0)));
	scene->planes.push_back(Plane(glm::vec3(0, 0, -containerDepth / 2.0f), glm::normalize(glm::vec3(0, 0, 1))));
	scene->planes.push_back(Plane(glm::vec3(0, 0, containerDepth / 2.0f), glm::normalize(glm::vec3(0, 0, -1))));
	scene->planes.push_back(Plane(glm::vec3(-containerWidth / 2.0f, 0, 0), glm::normalize(glm::vec3(1, 0, 0))));
	scene->planes.push_back(Plane(glm::vec3(containerWidth / 2.0f, 0, 0), glm::normalize(glm::vec3(-1, 0, 0))));
	scene->numParticles = 0;
	scene->numMaxParticles = 60000;
	scene->numRigidBodies = 0;
	scene->numMaxRigidBodies = 128;

	scene->pointLight.intensity = glm::vec3(5.0f);
	scene->pointLight.position = glm::vec3(1, 5, 1);
	scene->pointLight.direction = glm::normalize(-scene->pointLight.position);

	scene->camera = Camera(glm::vec3(0, 5, 7), glm::vec3(0, 2, 0), glm::radians(55.0f), (float) windowWidth / (float) windowHeight),

	// mass per particle unimplemented
	scene->fluids.push_back(Fluid::CreateFluidBlock(glm::ivec3(width, height, depth), glm::vec3(startX, startY, startZ), glm::vec3(diam), 1.0f, 800.0));
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
	window = InitGL(windowWidth, windowHeight);

	scene = initFluidScene();
	renderer = new ParticleRenderer(glm::uvec2(windowWidth, windowHeight), scene);
	solver = new ParticleSolver(scene);

	// fps counter
	std::chrono::high_resolution_clock::time_point lastUpdateTime = std::chrono::high_resolution_clock::now();
	int numFrame = 0;
	
	// imgui
	// Setup ImGui binding
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
	ImGui_ImplGlfwGL3_Init(window, true);

	// Setup style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();
	
	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

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
			if (scene->numRigidBodies > 0)
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
		}
		renderer->update();
		glfwPollEvents();

		/*ImGui_ImplGlfwGL3_NewFrame();

		// 1. Show a simple window.
		// Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets automatically appears in a window called "Debug".
		{
			static float f = 0.0f;
			static int counter = 0;
			ImGui::Text("Hello, world!");                           // Display some text (you can use a format string too)
			ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f    
			ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

			ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our windows open/close state
			ImGui::Checkbox("Another Window", &show_another_window);

			if (ImGui::Button("Button"))                            // Buttons return true when clicked (NB: most widgets return true when edited/activated)
				counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		}

		// 2. Show another simple window. In most cases you will use an explicit Begin/End pair to name your windows.
		if (show_another_window)
		{
			ImGui::Begin("Another Window", &show_another_window);
			ImGui::Text("Hello from another window!");
			if (ImGui::Button("Close Me"))
				show_another_window = false;
			ImGui::End();
		}

		// 3. Show the ImGui demo window. Most of the sample code is in ImGui::ShowDemoWindow(). Read its code to learn more about Dear ImGui!
		if (show_demo_window)
		{
			ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver); // Normally user code doesn't need/want to call this because positions are saved in .ini file anyway. Here we just want to make the demo initial state a bit more friendly!
			ImGui::ShowDemoWindow(&show_demo_window);
		}

		// Rendering
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);*/
		/*glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);*/
		/*ImGui::Render();
		ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());		// Swap buffers*/

		glfwSwapBuffers(window);

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
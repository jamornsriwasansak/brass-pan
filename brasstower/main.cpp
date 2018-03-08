#include "kernel.cuh"
#include "ext/helper_math.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>

#define cudaCheckErrors(msg) \
    { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    }

int main()
{
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 1); // 1x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 4.5
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

	// Open a window and create its OpenGL context
	GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
	window = glfwCreateWindow(1280, 720, "Work Please", NULL, NULL);
	if (window == NULL)
	{
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental = true; // Needed in core profile
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glClearColor(1.0f, 0.0f, 0.0f, 0.0f);

	int numParticles = 100;
	unsigned long long dataSize = 0;
	dataSize += sizeof(float) * 4 * numParticles; // positions (4 float * numParticles)
	//dataSize += sizeof(float) * 4 * numParticles; // velocity (4 float * numParticles)
	//dataSize += sizeof(float) * numParticles; // invMass (float)
	//dataSize += sizeof(int) * numParticles;

	// Create SSBO
	GLuint ssbo = 0;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, dataSize, NULL, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	cudaGraphicsResource **cmapSsbo;
	cudaCheckErrors(cudaGraphicsGLRegisterBuffer(cmapSsbo, ssbo, cudaGraphicsRegisterFlagsWriteDiscard));

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaCheckErrors(cudaGraphicsMapResources(1, cmapSsbo, stream));
	cudaCheckErrors(cudaGraphicsUnmapResources(1, cmapSsbo, stream));

	do
	{
		// Draw nothing, see you in tutorial 2 !
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	/*
	float4 k = make_float4(10, 10, 1, 0);
	float4 p = k * 2.0f;
	tryReduce();
	main2();
	_getch();
	return 0;
	*/
}
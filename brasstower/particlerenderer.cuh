#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "cuda/helper.cuh"
#include "mesh.h"

struct Camera
{
	static inline glm::vec3 SphericalToWorld(const glm::vec2 & thetaPhi)
	{
		const float & phi = thetaPhi.x;
		const float & theta = thetaPhi.y;
		const float sinphi = std::sin(phi);
		const float cosphi = std::cos(phi);
		const float sintheta = std::sin(theta);
		const float costheta = std::cos(theta);
		return glm::vec3(costheta * sinphi, cosphi, sintheta * sinphi);
	}

	static inline glm::vec2 WorldToSpherical(const glm::vec3 & pos)
	{
		const float phi = std::atan2(pos.z, pos.x);
		const float numerator = std::sqrt(pos.x * pos.x + pos.z * pos.z);
		const float theta = std::atan2(numerator, pos.y);
		return glm::vec2(theta, phi);
	}

	Camera(const glm::vec3 & pos, const glm::vec3 & lookAt, const float fovy, const float aspectRatio):
		pos(pos),
		dir(glm::normalize(lookAt - pos)),
		thetaPhi(WorldToSpherical(glm::normalize(lookAt - pos))),
		up(glm::vec3(0.0f, 1.0f, 0.0f)),
		fovY(fovy),
		aspectRatio(aspectRatio)
	{}

	void shift(const glm::vec3 & move)
	{
		const glm::vec3 & mBasisZ = dir;
		const glm::vec3 & mBasisX = glm::normalize(glm::cross(up, mBasisZ));
		pos += mBasisZ * move.z + mBasisX * move.x;
	}

	void rotate(const glm::vec2 & rotation)
	{
		thetaPhi += rotation;
		dir = SphericalToWorld(thetaPhi);
	}

	glm::mat4 vpMatrix()
	{
		glm::mat4 viewMatrix = glm::lookAt(pos, dir + pos, glm::vec3(0, 1, 0));
		glm::mat4 projMatrix = glm::perspective(fovY, aspectRatio, 0.01f, 100.0f);
		return projMatrix * viewMatrix;
	}

	float fovY;
	float aspectRatio;
	glm::vec3 pos;
	glm::vec2 thetaPhi;
	glm::vec3 dir;
	glm::vec3 up;
};

static GLFWwindow* InitGL(const size_t width, const size_t height)
{
	if (!glfwInit())
	{
		throw new std::exception("Failed to initialize GLFW\n");
	}

	glfwWindowHint(GLFW_SAMPLES, 1); // 1x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 

	GLFWwindow* window; 
	window = glfwCreateWindow(width, height, "Work Please", NULL, NULL);
	if (window == NULL)
	{
		throw new std::exception("Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
	}
	glfwMakeContextCurrent(window); 
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		throw new std::exception("Failed to initialize GLEW\n");
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	return window;
}

// should be singleton
struct ParticleRenderer
{
	const size_t MaxNumParticles = 1000;

	ParticleRenderer(const glm::uvec2 & resolution, const float radius):
		camera(glm::vec3(0, 1, -1), glm::vec3(0), glm::radians(70.0f), (float)resolution.x / (float)resolution.y),
		radius(radius)
	{
		glGenVertexArrays(1, &globalVaoHandle);
		glBindVertexArray(globalVaoHandle);

		// init ssbobuffer and register in cuda
		glGenBuffers(1, &ssboBuffer);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(float) * MaxNumParticles, 0, GL_DYNAMIC_COPY);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ssboGraphicsRes, ssboBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		// load particle mesh
		particleMesh = Mesh::Load("icosphere.obj")[0];

		// init simple program
		initSimpleProgram();
	}

	std::shared_ptr<OpenglProgram> simpleProgram;
	std::shared_ptr<OpenglUniform> simpleProgram_uMVPMatrix;
	std::shared_ptr<OpenglUniform> simpleProgram_uRadius;
	GLuint simpleProgram_ssboBinding;
	void initSimpleProgram()
	{
		simpleProgram = std::make_shared<OpenglProgram>();
		simpleProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/simple.vert"));
		simpleProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/simple.frag"));
		simpleProgram->compile();

		simpleProgram_uMVPMatrix = simpleProgram->registerUniform("uMVP");
		simpleProgram_uRadius = simpleProgram->registerUniform("uRadius");
		GLuint index = glGetProgramResourceIndex(simpleProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "ParticlesInfo");
		simpleProgram_ssboBinding = 0;
		glShaderStorageBlockBinding(simpleProgram_ssboBinding, index, simpleProgram_ssboBinding);
	}

	void update(const size_t numParticles)
	{
		// render
		glUseProgram(simpleProgram->mHandle);

		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, particleMesh->mGl.mVerticesBuffer->mHandle);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		simpleProgram_uMVPMatrix->setMat4(camera.vpMatrix());
		simpleProgram_uRadius->setFloat(radius);

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, simpleProgram_ssboBinding, ssboBuffer);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particleMesh->mGl.mIndicesBuffer->mHandle);
		glDrawElementsInstanced(GL_TRIANGLES, particleMesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0, numParticles);
	}

	float4* mapSsbo()
	{
		float4 *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &ssboGraphicsRes, 0));
		size_t numBytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &numBytes, ssboGraphicsRes));
		return dptr;
	}

	void unmapSsbo()
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &ssboGraphicsRes, 0));
	}

	float radius;
	std::shared_ptr<Mesh> particleMesh;
	Camera camera;
	GLuint ssboBuffer;
	cudaGraphicsResource_t ssboGraphicsRes;

private:
	GLuint globalVaoHandle;
};
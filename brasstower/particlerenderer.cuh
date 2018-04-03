#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "cuda/helper.cuh"
#include "cuda/cudamatrix.cuh"
#include "mesh.h"
#include "scene.h"

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
		pos += mBasisZ * move.z + mBasisX * move.x + up * move.y;
	}

	void rotate(const glm::vec2 & rotation)
	{
		thetaPhi += rotation;
		dir = SphericalToWorld(thetaPhi);
	}

	glm::mat4 vpMatrix()
	{
		glm::mat4 viewMatrix = glm::lookAt(pos, dir + pos, glm::vec3(0, 1, 0));
		glm::mat4 projMatrix = glm::perspective(fovY, aspectRatio, 0.05f, 100.0f);
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

	glfwWindowHint(GLFW_SAMPLES, 16); // antialiasing
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
	//glfwSwapInterval(1); // vsync

	return window;
}

// should be singleton
struct ParticleRenderer
{
	ParticleRenderer(const glm::uvec2 & resolution, const std::shared_ptr<Scene> & scene):
		resolution(resolution),
		camera(glm::vec3(0, 5, 7), glm::vec3(0, 2, 0), glm::radians(55.0f), (float)resolution.x / (float)resolution.y),
		scene(scene)
	{
		glGenVertexArrays(1, &globalVaoHandle);
		glBindVertexArray(globalVaoHandle);

		// init ssbobuffer for particle positions
		glGenBuffers(1, &particlePositionsSsboBuffer);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, particlePositionsSsboBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(float) * scene->numMaxParticles, 0, GL_DYNAMIC_COPY);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&particlePositionsSsboGraphicsRes, particlePositionsSsboBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

		// init ssbobuffer for transformation matrices
		glGenBuffers(1, &rigidBodyMatricesSsboBuffer);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, rigidBodyMatricesSsboBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 4 * sizeof(float) * scene->numMaxParticles, 0, GL_DYNAMIC_COPY);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&rigidBodyMatricesSsboGraphicsRes, rigidBodyMatricesSsboBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable(GL_MULTISAMPLE);
		glDepthFunc(GL_LEQUAL);

		// load particle mesh
		particleMesh = MeshGenerator::Cube();
		particleMesh->createOpenglBuffer();
		planeMesh = MeshGenerator::Plane();
		planeMesh->createOpenglBuffer();

		initParticleDrawingProgram();
		initMeshDrawingProgram();
		initMeshShadowProgram();
		initInfinitePlaneDrawingProgram();
		initParticleColorCodeFramebuffer();
		initParticleColorCodeProgram();
	}

	std::shared_ptr<OpenglProgram> particlesDrawingProgram;
	std::shared_ptr<OpenglUniform> particlesDrawingProgram_uMVPMatrix;
	std::shared_ptr<OpenglUniform> particlesDrawingProgram_uRadius;
	std::shared_ptr<OpenglUniform> particlesDrawingProgram_uCameraPosition;
	GLuint particlesDrawingProgram_ssboBinding;
	void initParticleDrawingProgram()
	{
		particlesDrawingProgram = std::make_shared<OpenglProgram>();
		particlesDrawingProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/particle.vert"));
		particlesDrawingProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/particle.frag"));
		particlesDrawingProgram->compile();

		particlesDrawingProgram_uMVPMatrix = particlesDrawingProgram->registerUniform("uMVP");
		particlesDrawingProgram_uRadius = particlesDrawingProgram->registerUniform("uRadius");
		particlesDrawingProgram_uCameraPosition = particlesDrawingProgram->registerUniform("uCameraPosition");
		GLuint index = glGetProgramResourceIndex(particlesDrawingProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "ParticlePositions");
		particlesDrawingProgram_ssboBinding = 0;
		glShaderStorageBlockBinding(particlesDrawingProgram->mHandle, index, particlesDrawingProgram_ssboBinding);
	}

	std::shared_ptr<OpenglProgram> meshDrawingProgram;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uVPMatrix;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uColor;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uRigidBodyId;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uLightPosition;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uLightDir;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uLightIntensity;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uLightExponent;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uShadowMatrix;
	std::shared_ptr<OpenglUniform> meshDrawingProgram_uShadowMap;
	GLuint meshDrawingProgram_ssboBinding;
	void initMeshDrawingProgram()
	{
		meshDrawingProgram = std::make_shared<OpenglProgram>();
		meshDrawingProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/mesh.vert"));
		meshDrawingProgram->attachGeometryShader(OpenglGeometryShader::CreateFromFile("glshaders/mesh.geom"));
		meshDrawingProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/mesh.frag"));
		meshDrawingProgram->compile();

		meshDrawingProgram_uVPMatrix = meshDrawingProgram->registerUniform("uVPMatrix");
		meshDrawingProgram_uColor = meshDrawingProgram->registerUniform("uColor");
		meshDrawingProgram_uRigidBodyId = meshDrawingProgram->registerUniform("uRigidBodyId");
		meshDrawingProgram_uLightPosition = meshDrawingProgram->registerUniform("uLightPosition");
		meshDrawingProgram_uLightDir = meshDrawingProgram->registerUniform("uLightDir");
		meshDrawingProgram_uLightIntensity = meshDrawingProgram->registerUniform("uLightIntensity");
		meshDrawingProgram_uLightExponent = meshDrawingProgram->registerUniform("uLightExponent");
		meshDrawingProgram_uShadowMatrix = meshDrawingProgram->registerUniform("uShadowMatrix");
		meshDrawingProgram_uShadowMap = meshDrawingProgram->registerUniform("uShadowMap");
		GLuint index = glGetProgramResourceIndex(meshDrawingProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "ModelMatrices");
		meshDrawingProgram_ssboBinding = 0;
		glShaderStorageBlockBinding(meshDrawingProgram->mHandle, index, meshDrawingProgram_ssboBinding);
	}

	std::shared_ptr<OpenglProgram> planeDrawingProgram;
	std::shared_ptr<OpenglUniform> planeDrawingProgram_uVPMatrix;
	std::shared_ptr<OpenglUniform> planeDrawingProgram_uModelMatrix;
	std::shared_ptr<OpenglUniform> planeDrawingProgram_uCameraPosition;
	std::shared_ptr<OpenglUniform> planeDrawingProgram_uShadowMatrix;
	std::shared_ptr<OpenglUniform> planeDrawingProgram_uShadowMap;
	void initInfinitePlaneDrawingProgram()
	{
		planeDrawingProgram = std::make_shared<OpenglProgram>();
		planeDrawingProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/plane.vert"));
		planeDrawingProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/plane.frag"));
		planeDrawingProgram->compile();

		planeDrawingProgram_uVPMatrix = planeDrawingProgram->registerUniform("uVPMatrix");
		planeDrawingProgram_uModelMatrix = planeDrawingProgram->registerUniform("uModelMatrix");
		planeDrawingProgram_uCameraPosition = planeDrawingProgram->registerUniform("uCameraPos");
		planeDrawingProgram_uShadowMatrix = planeDrawingProgram->registerUniform("uShadowMatrix");
		planeDrawingProgram_uShadowMap = planeDrawingProgram->registerUniform("uShadowMap");
	}

	GLuint meshShadowFramebufferHandle;
	GLuint meshShadowDepthTextureHandle;
	std::shared_ptr<OpenglProgram> meshShadowProgram;
	std::shared_ptr<OpenglUniform> meshShadowProgram_uShadowMatrix;
	std::shared_ptr<OpenglUniform> meshShadowProgram_uRigidBodyId;
	GLuint meshShadowProgram_ssboBinding;
	void initMeshShadowProgram()
	{
		glGenFramebuffers(1, &meshShadowFramebufferHandle);
		glBindFramebuffer(GL_FRAMEBUFFER, meshShadowFramebufferHandle);

		GLuint depthRenderbuffer;
		glGenRenderbuffers(1, &depthRenderbuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);

		glGenTextures(1, &meshShadowDepthTextureHandle);
		glBindTexture(GL_TEXTURE_2D, meshShadowDepthTextureHandle);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1024, 1024, 0, GL_RGB, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, meshShadowDepthTextureHandle, 0);
		GLenum drawBuffers[1] = {GL_COLOR_ATTACHMENT0};
		glDrawBuffers(1, drawBuffers);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::exception("framebuffer error");

		glBindFramebuffer(GL_FRAMEBUFFER, NULL);

		meshShadowProgram = std::make_shared<OpenglProgram>();
		meshShadowProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/meshshadow.vert"));
		meshShadowProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/meshshadow.frag"));
		meshShadowProgram->compile();

		meshShadowProgram_uRigidBodyId = meshShadowProgram->registerUniform("uRigidBodyId");
		meshShadowProgram_uShadowMatrix = meshShadowProgram->registerUniform("uShadowMatrix");
		GLuint index = glGetProgramResourceIndex(meshShadowProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "ModelMatrices");
		meshShadowProgram_ssboBinding = 0;
		glShaderStorageBlockBinding(meshShadowProgram->mHandle, index, meshShadowProgram_ssboBinding);
	}

	GLuint particlesColorCodeFramebufferHandle;
	GLuint particlesColorCodeTextureHandle;
	void initParticleColorCodeFramebuffer()
	{
		glGenFramebuffers(1, &particlesColorCodeFramebufferHandle);
		glBindFramebuffer(GL_FRAMEBUFFER, particlesColorCodeFramebufferHandle);

		glGenTextures(1, &particlesColorCodeTextureHandle);
		glBindTexture(GL_TEXTURE_2D, particlesColorCodeTextureHandle);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, particlesColorCodeTextureHandle, 0);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			throw std::exception("framebuffer error");

		glBindFramebuffer(GL_FRAMEBUFFER, NULL);
	}

	// for picking a particle
	std::shared_ptr<OpenglProgram> particlesColorCodeProgram;
	std::shared_ptr<OpenglUniform> particlesColorCodeProgram_uMVPMatrix;
	std::shared_ptr<OpenglUniform> particlesColorCodeProgram_uRadius;
	std::shared_ptr<OpenglUniform> particlesColorCodeProgram_uCameraPosition;
	GLuint particlesColorCodeProgram_ssboBinding;
	void initParticleColorCodeProgram()
	{
		particlesColorCodeProgram = std::make_shared<OpenglProgram>();
		particlesColorCodeProgram->attachVertexShader(OpenglVertexShader::CreateFromFile("glshaders/particlecolorcode.vert"));
		particlesColorCodeProgram->attachFragmentShader(OpenglFragmentShader::CreateFromFile("glshaders/particlecolorcode.frag"));
		particlesColorCodeProgram->compile();

		particlesColorCodeProgram_uMVPMatrix = particlesColorCodeProgram->registerUniform("uMVP");
		particlesColorCodeProgram_uRadius = particlesColorCodeProgram->registerUniform("uRadius");
		particlesColorCodeProgram_uCameraPosition = particlesColorCodeProgram->registerUniform("uCameraPosition");
		GLuint index = glGetProgramResourceIndex(particlesDrawingProgram->mHandle, GL_SHADER_STORAGE_BLOCK, "ParticlePositions");
		particlesDrawingProgram_ssboBinding = 0;
		glShaderStorageBlockBinding(particlesDrawingProgram->mHandle, index, particlesDrawingProgram_ssboBinding);
	}

	int queryParticleColorCode(const glm::uvec2 & pos)
	{
		glm::mat4 cameraVpMatrix = camera.vpMatrix();
		glBindFramebuffer(GL_FRAMEBUFFER, particlesColorCodeFramebufferHandle);
		// set to max
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		// draw all particles
		{
			glUseProgram(particlesColorCodeProgram->mHandle);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, particleMesh->mGl.mVerticesBuffer->mHandle);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
			particlesColorCodeProgram_uMVPMatrix->setMat4(cameraVpMatrix);
			particlesColorCodeProgram_uRadius->setFloat(scene->radius);
			particlesColorCodeProgram_uCameraPosition->setVec3(camera.pos);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particlesDrawingProgram_ssboBinding, particlePositionsSsboBuffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particleMesh->mGl.mIndicesBuffer->mHandle);
			glDrawElementsInstanced(GL_TRIANGLES, particleMesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0, scene->numParticles);
			glDisableVertexAttribArray(0);
		}

		GLubyte pixels[3];
		glReadPixels(pos.x, resolution.y - pos.y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		int particleId = (pixels[0] * 256 * 256) + (pixels[1] * 256) + pixels[2];
		glBindFramebuffer(GL_FRAMEBUFFER, NULL);
		// reset to old color
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		return particleId;
	}

	void update()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, meshShadowFramebufferHandle);
		glViewport(0, 0, 1024, 1024);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glm::mat4 shadowMatrix = scene->pointLight.shadowMatrix();
		// render shadow map
		{
			// for mesh
			{
				glUseProgram(meshShadowProgram->mHandle);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, meshShadowProgram_ssboBinding, rigidBodyMatricesSsboBuffer);
				glEnableVertexAttribArray(0);
				meshShadowProgram_uShadowMatrix->setMat4(shadowMatrix);
				for (int i = 0; i < scene->numRigidBodies; i++)
				{
					glBindBuffer(GL_ARRAY_BUFFER, scene->rigidBodies[i]->mesh->mGl.mVerticesBuffer->mHandle);
					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
					glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->rigidBodies[i]->mesh->mGl.mIndicesBuffer->mHandle);
					meshShadowProgram_uRigidBodyId->setInt(i);
					glDrawElements(GL_TRIANGLES, scene->rigidBodies[i]->mesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0);
				}
				glDisableVertexAttribArray(0);
			}
		}

		glBindFramebuffer(GL_FRAMEBUFFER, NULL);
		glViewport(0, 0, 1280, 720);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glm::mat4 cameraVpMatrix = camera.vpMatrix();

		/*
		// render particles
		{
			glUseProgram(particlesDrawingProgram->mHandle);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, particleMesh->mGl.mVerticesBuffer->mHandle);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
			particlesDrawingProgram_uMVPMatrix->setMat4(cameraVpMatrix);
			particlesDrawingProgram_uRadius->setFloat(scene->radius);
			particlesDrawingProgram_uCameraPosition->setVec3(camera.pos);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particlesDrawingProgram_ssboBinding, particlePositionsSsboBuffer);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particleMesh->mGl.mIndicesBuffer->mHandle);
			glDrawElementsInstanced(GL_TRIANGLES, particleMesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0, scene->numParticles);
			glDisableVertexAttribArray(0);
		}
		*/

		// render rigidbody meshes
		{
			glUseProgram(meshDrawingProgram->mHandle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, meshDrawingProgram_ssboBinding, rigidBodyMatricesSsboBuffer);
			glEnableVertexAttribArray(0);
			meshDrawingProgram_uLightPosition->setVec3(scene->pointLight.position);
			meshDrawingProgram_uLightDir->setVec3(scene->pointLight.direction);
			meshDrawingProgram_uLightExponent->setFloat(scene->pointLight.exponent);
			meshDrawingProgram_uLightIntensity->setVec3(scene->pointLight.intensity);
			meshDrawingProgram_uVPMatrix->setMat4(cameraVpMatrix);
			meshDrawingProgram_uShadowMatrix->setMat4(shadowMatrix);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, meshShadowDepthTextureHandle);
			meshDrawingProgram_uShadowMap->setInt(0);

			for (int i = 0; i < scene->numRigidBodies; i++)
			{
				glBindBuffer(GL_ARRAY_BUFFER, scene->rigidBodies[i]->mesh->mGl.mVerticesBuffer->mHandle);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, scene->rigidBodies[i]->mesh->mGl.mIndicesBuffer->mHandle);
				meshDrawingProgram_uRigidBodyId->setInt(i);
				meshDrawingProgram_uColor->setVec3(scene->rigidBodies[i]->color);
				glDrawElements(GL_TRIANGLES, scene->rigidBodies[i]->mesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0);
			}
			glDisableVertexAttribArray(0);
		}

		// render plane
		{
			glUseProgram(planeDrawingProgram->mHandle);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, planeMesh->mGl.mVerticesBuffer->mHandle);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, planeMesh->mGl.mIndicesBuffer->mHandle);
			planeDrawingProgram_uVPMatrix->setMat4(cameraVpMatrix);
			planeDrawingProgram_uCameraPosition->setVec3(camera.pos);
			planeDrawingProgram_uShadowMatrix->setMat4(shadowMatrix);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, meshShadowDepthTextureHandle);
			planeDrawingProgram_uShadowMap->setInt(0);

			for (const Plane & plane : scene->planes)
			{
				planeDrawingProgram_uModelMatrix->setMat4(plane.modelMatrix);
				glDrawElements(GL_TRIANGLES, planeMesh->mNumTriangles * 3, GL_UNSIGNED_INT, (void*)0);
			}
			glDisableVertexAttribArray(0);
		}
	}

	float4* mapParticlePositionsSsbo()
	{
		float4 *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &particlePositionsSsboGraphicsRes, 0));
		size_t numBytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &numBytes, particlePositionsSsboGraphicsRes));
		return dptr;
	}

	void unmapParticlePositionsSsbo()
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &particlePositionsSsboGraphicsRes, 0));
	}

	matrix4* mapMatricesSsbo()
	{
		matrix4 *dptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &rigidBodyMatricesSsboGraphicsRes, 0));
		size_t numBytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &numBytes, rigidBodyMatricesSsboGraphicsRes));
		return dptr;
	}

	void unmapMatricesSsbo()
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &rigidBodyMatricesSsboGraphicsRes, 0));
	}

	std::shared_ptr<Mesh>			particleMesh;
	std::shared_ptr<Mesh>			planeMesh;

	const std::shared_ptr<Scene>	scene;
	const glm::uvec2				resolution;
	Camera							camera;
	GLuint							particlePositionsSsboBuffer;
	cudaGraphicsResource_t			particlePositionsSsboGraphicsRes;
	GLuint							rigidBodyMatricesSsboBuffer;
	cudaGraphicsResource_t			rigidBodyMatricesSsboGraphicsRes;
private:
	GLuint globalVaoHandle;
};
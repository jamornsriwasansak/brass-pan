#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

struct Plane
{
	glm::vec3 origin;
	glm::vec3 normal;
	glm::mat4 modelMatrix;

	Plane(const glm::vec3 & origin, const glm::vec3 & normal) :
		origin(origin), normal(normal)
	{
		// compute orthonormal bases for constructing rotation part of the matrix
		glm::vec3 zBasis = normal;
		glm::vec3 xBasis, yBasis;

		float sign = std::copysign((float)(1.0), zBasis.z);
		const float a = -1.0f / (sign + zBasis.z);
		const float b = zBasis.x * zBasis.y * a;
		xBasis = glm::vec3(1.0f + sign * zBasis.x * zBasis.x * a, sign * b, -sign * zBasis.x);
		yBasis = glm::vec3(b, sign + zBasis.y * zBasis.y * a, -zBasis.y);

		modelMatrix = glm::mat4(yBasis.x, yBasis.y, yBasis.z, 0.0f,
								zBasis.x, zBasis.y, zBasis.z, 0.0f,
								xBasis.x, xBasis.y, xBasis.z, 0.0f,
								origin.x, origin.y, origin.z, 1.f);
	}
};

struct RigidBody
{
	std::shared_ptr<Mesh> mesh;
	size_t numParticles;

	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> positions_CM_Origin; // recomputed positions where all CM are placed at origin
	glm::vec3 CM; // center of mass
	glm::vec3 color;

	float massPerParticle;

	static std::shared_ptr<RigidBody> CreateRigidBox(const glm::vec3 & color,
													 const glm::ivec3 & dimension,
													 const glm::vec3 & startPosition,
													 const glm::vec3 & stepSize,
													 const float massPerParticle)
	{
		std::shared_ptr<RigidBody> result = std::make_shared<RigidBody>();
		result->color = color;
		result->massPerParticle = massPerParticle;

		std::vector<glm::vec3> & positions = result->positions;
		glm::vec3 CM = glm::vec3(0.0f);
		for (int i = 0; i < dimension.x; i++)
			for (int j = 0; j < dimension.y; j++)
				for (int k = 0; k < dimension.z; k++)
				{
					glm::vec3 position = startPosition + stepSize * glm::vec3(i, j, k);
					CM += position;
					positions.push_back(position);
				}

		CM /= static_cast<float>(positions.size());

		std::vector<glm::vec3> & positions_CM_Origin = result->positions_CM_Origin;
		for (int i = 0; i < positions.size(); i++)
			positions_CM_Origin.push_back(positions[i] - CM);

		result->CM = CM;

		result->mesh = MeshGenerator::Cube();
		glm::vec3 size = stepSize * glm::vec3(dimension) * 0.5f;
		result->mesh->applyTransform(glm::scale(size));
		result->mesh->createOpenglBuffer();
		return result;
	}

	RigidBody()
	{
	}
};

struct Granulars
{
	std::vector<glm::vec3> positions;
	float massPerParticle;

	static std::shared_ptr<Granulars> CreateGranularsBlock(const glm::ivec3 & dimension,
														   const glm::vec3 & startPosition,
														   const glm::vec3 & stepSize,
														   const float massPerParticle)
	{
		std::shared_ptr<Granulars> result = std::make_shared<Granulars>();
		result->massPerParticle = massPerParticle;

		std::vector<glm::vec3> & positions = result->positions;
		for (int i = 0; i < dimension.x; i++)
			for (int j = 0; j < dimension.y; j++)
				for (int k = 0; k < dimension.z; k++)
				{
					glm::vec3 position = startPosition + stepSize * glm::vec3(i, j, k);
					positions.push_back(position);
				}

		return result;
	}

	Granulars()
	{
	}
};

struct Fluid
{
	std::vector<glm::vec3> positions;
	float restDensity;

	static std::shared_ptr<Fluid> CreateWaterBlock(const glm::ivec3 & dimension,
													const glm::vec3 & startPosition,
													const glm::vec3 & stepSize,
													const float density)
	{
		std::shared_ptr<Fluid> result = std::make_shared<Fluid>();
		result->restDensity = density;

		std::vector<glm::vec3> & positions = result->positions;
		for (int i = 0; i < dimension.x; i++)
			for (int j = 0; j < dimension.y; j++)
				for (int k = 0; k < dimension.z; k++)
				{
					glm::vec3 position = startPosition + stepSize * glm::vec3(i, j, k);
					positions.push_back(position);
				}

		return result;
	}
};

struct PointLight
{
	glm::mat4 shadowMatrix()
	{
		glm::mat4 projMatrix = glm::perspective(thetaMinMax.y * 2.0f, 1.0f, 0.5f, 100.0f);
		glm::mat4 viewMatrix = glm::lookAt(position, position + direction, glm::vec3(0, 1, 0));
		return projMatrix * viewMatrix;
	}

	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 intensity;
	glm::vec2 thetaMinMax = glm::radians(glm::vec2(45.f, 50.0f));
};

struct Scene
{
	std::vector<Plane> planes;
	std::vector<std::shared_ptr<RigidBody>> rigidBodies;
	std::vector<std::shared_ptr<Granulars>> granulars; // position of solid particles (without any constraints)
	std::vector<std::shared_ptr<Fluid>> fluids;

	PointLight pointLight;

	/// THESE ARE PARTICLE SYSTEM PARAMETERS! don't touch!
	/// TODO:: move these to particle system
	size_t numParticles = 0;
	size_t numMaxParticles = 0;
	size_t numRigidBodies = 0;
	size_t numMaxRigidBodies = 0;
	float radius;
};
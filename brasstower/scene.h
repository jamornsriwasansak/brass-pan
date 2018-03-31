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
	size_t startIndex;
	size_t numParticles;

	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> positions_CM_Origin; // recomputed positions where all CM are placed at origin
	glm::vec3 CM; // center of mass

	static std::shared_ptr<RigidBody> CreateRigidBox(const glm::ivec3 & dimension,
													 const glm::vec3 & startPosition,
													 const glm::vec3 & stepSize)
	{
		std::shared_ptr<RigidBody> result = std::make_shared<RigidBody>();

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

struct Scene
{
	std::vector<Plane> planes;
	std::vector<std::shared_ptr<RigidBody>> rigidBodies;
	std::vector<std::shared_ptr<glm::vec3>> granulars; // position of solid particles (without any constraints)

	/// THESE ARE PARTICLE SYSTEM PARAMETERS! don't touch!
	/// TODO:: move these to particle system
	size_t numParticles = 0;
	size_t numMaxParticles = 0;
	size_t numRigidBodies = 0;
	size_t numMaxRigidBodies = 0;
	float radius;
};
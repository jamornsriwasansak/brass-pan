#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>

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
	size_t startIndex;
	size_t numParticles;

	RigidBody()
	{
	}
};

struct Scene
{
	std::vector<Plane> planes;
	size_t numParticles = 0;
	size_t numMaxParticles = 0;

	size_t numRigidBodies = 0;
	size_t numMaxRigidBodies = 0;
	float radius;
};
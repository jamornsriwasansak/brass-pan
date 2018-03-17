#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>

struct StaticPlane
{
	glm::vec3 origin;
	glm::vec3 dir;
	glm::mat4 modelMatrix;

	StaticPlane(const glm::vec3 & origin, const glm::vec3 & dir) :
		origin(origin), dir(dir)
	{
		// compute orthonormal bases for constructing rotation part of the matrix
		glm::vec3 zBasis = dir;
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

struct Scene
{
	std::vector<StaticPlane> planes;
	size_t numParticles;
	float radius;
};
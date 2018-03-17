#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>

struct StaticPlane
{
	glm::vec4 origin;
	glm::vec4 dir;
};

struct Scene
{
	std::vector<StaticPlane> planes;
};
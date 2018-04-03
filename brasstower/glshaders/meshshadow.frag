#version 450 core

layout(location = 0) out vec3 depth;

in vec3 vPosition;

void main()
{
	depth = 10.0f - vec3(vPosition.z);
}
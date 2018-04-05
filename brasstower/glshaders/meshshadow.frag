#version 450 core

layout(location = 0) out float depth;

in vec3 vPosition;

void main()
{
	depth = 10.0f - vPosition.z;
}
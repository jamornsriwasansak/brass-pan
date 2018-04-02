#version 450 core

layout(location = 0) out vec3 depth;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;

in vec3 vPosition;

void main()
{
	depth = vPosition;
}
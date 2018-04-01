#version 450 core

layout(location = 0) out float depth;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;

void main()
{
	depth = gl_FragCoord.z;
}
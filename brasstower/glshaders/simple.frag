#version 450 core

layout(location = 0) out vec4 fPosition;

in vec3 vPosition;
in vec3 vNormal;

void main()
{
	vec3 shadingNormal = normalize(vNormal);
	fPosition = vec4(shadingNormal, 1.0f);
}
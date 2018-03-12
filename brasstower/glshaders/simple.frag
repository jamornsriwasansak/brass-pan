#version 450 core

layout(location = 0) out vec4 fPosition;

in vec3 vPosition;
in vec3 vNormal;

const vec3 pointLightPosition = vec3(1, 0, 0);

void main()
{
	vec3 diff = pointLightPosition - vPosition;
	float dist2 = dot(diff, diff);

	vec3 shadingNormal = normalize(vNormal);
	fPosition = vec4(shadingNormal, 1.0f);
}
#version 450 core

layout(location = 0) out vec4 color;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform float uLightExponent;

uniform vec3 uColor;

in vec3 gNormal;
in vec3 gPosition;

vec3 light()
{
	vec3 diff = uLightPosition - gPosition;
	float dist2 = dot(diff, diff);
	float unnormCos1 = max(dot(diff, gNormal), 0.f);
	float unnormCos2 = max(-dot(diff, uLightDir), 0.f);
	return uLightIntensity * unnormCos1 * unnormCos2 / (dist2 * dist2);
}

void main()
{
	vec3 ambient = uColor * 0.5f;
	color = vec4(light() * uColor + ambient, 1.0f);
}
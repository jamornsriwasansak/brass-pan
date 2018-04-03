#version 450 core

layout(location = 0) out vec4 color;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;

uniform vec3 uColor;

uniform sampler2D uShadowMap;

in vec3 gNormal;
in vec3 gPosition;
in vec4 gShadowCoord;

vec3 light()
{
	vec3 diff = uLightPosition - gPosition;
	float dist2 = dot(diff, diff);
	float unnormCos1 = max(dot(diff, gNormal), 0.f);
	float unnormCos2 = max(-dot(diff, uLightDir), 0.f);
	return uLightIntensity * unnormCos1 * unnormCos2 / (dist2 * dist2);
}

float visibility()
{
	float bias = 0.003f * tan(acos(dot(normalize(uLightPosition - gPosition), gNormal)));
	return (10.f - texture(uShadowMap, gShadowCoord.xy).z) > (gShadowCoord.z - bias) ? 0.f : 1.f;
}

void main()
{
	vec3 ambient = uColor * 0.5f;
	color = vec4((1.0f - visibility()) * light() * uColor + ambient, 1.0f);
}
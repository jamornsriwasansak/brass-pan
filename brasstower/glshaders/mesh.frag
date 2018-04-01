#version 450 core

layout(location = 0) out vec4 color;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform float uLightExponent;

uniform vec3 uColor;

uniform sampler2D uShadowMap;
//uniform sampler2DShadow uShadowMap;

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
	//float v = texture(uShadowMap, vec3(gShadowCoord.xy, gShadowCoord.z / gShadowCoord.w));
	vec3 projShadowCoord = gShadowCoord.xyz / gShadowCoord.w;
	projShadowCoord = projShadowCoord * 0.5f + 0.5f;
	if (texture(uShadowMap, projShadowCoord.xy).r < projShadowCoord.z)
	{
		return 0.0f;
	}
	return 1.0f;
	//return v;
}

void main()
{
	vec3 ambient = uColor * 0.5f;
	//color = vec4(visibility() * light() * uColor + ambient, 1.0f);
	//color = vec4(vec3(visibility()), 1.0f);
	vec3 projShadowCoord = gShadowCoord.xyz / gShadowCoord.w;
	projShadowCoord = projShadowCoord * 0.5f + 0.5f;
	color = vec4(texture(uShadowMap, projShadowCoord.xy).r);
}
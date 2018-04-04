#version 450 core

layout(location = 0) out vec4 color;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform vec2 uLightThetaMinMax;

uniform vec3 uColor;

uniform sampler2D uShadowMap;

in vec3 gNormal;
in vec3 gPosition;
in vec4 gShadowCoord;

vec3 shadeSpotlight(vec3 position, vec3 normal)
{
	// compute contrib
	vec3 diff = uLightPosition - position;
	float dist2 = dot(diff, diff);
	float dist = sqrt(dist2);
	vec3 nDiff = diff / dist;

	float cos1 = max(dot(nDiff, normal), 0.f);
	float cos2 = max(-dot(nDiff, uLightDir), 0.f);
	float spotlightScale = 1.0f - smoothstep(uLightThetaMinMax.x, uLightThetaMinMax.y, acos(cos2));
	return min(cos1 * spotlightScale / dist2, 0.01f) * 10.f * uLightIntensity;
}

float visibility()
{
	float bias = 0.005f * tan(acos(dot(normalize(uLightPosition - gPosition), gNormal)));
	return (10.f - texture(uShadowMap, gShadowCoord.xy).z) > (gShadowCoord.z - bias) ? 0.f : 1.f;
}

void main()
{
	vec3 ambient = uColor * 0.5f;
	color = vec4((1.0f - visibility()) * shadeSpotlight(gPosition, gNormal) * uColor + ambient, 1.0f);
}
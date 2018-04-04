#version 450 core

layout(location = 0) out vec4 color;

in vec4 vPosition;
in vec4 vShadowCoord;

uniform vec3 uCameraPos;
uniform sampler2D uShadowMap;

uniform vec3 uPlaneNormal;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform vec2 uLightThetaMinMax;

vec3 shadeSpotlight(vec3 position)
{
	vec3 diff = uLightPosition - position;
	float dist2 = dot(diff, diff);
	float dist = sqrt(dist2);
	vec3 nDiff = diff / dist;

	float cos1 = max(dot(nDiff, uPlaneNormal), 0.f);
	float cos2 = max(-dot(nDiff, uLightDir), 0.f);
	float spotlightScale = 1.0f - smoothstep(uLightThetaMinMax.x, uLightThetaMinMax.y, acos(cos2));
	return min(cos1 * spotlightScale / dist2, 0.01f) * 10.f * uLightIntensity;
}

void main()
{
	// discretize position
	ivec4 discretized = ivec4(ceil(vPosition));
	// grid code if code = (1, 1, 1) means that it's odd block else it's even block
	vec3 code = vec3((discretized.x + discretized.y + discretized.z) % 2);
	// map code to diff reflectance
	vec3 diffuseReflectance = (code + 5.5f) / 10.0;

	vec3 projShadowCoord = vShadowCoord.xyz / vShadowCoord.w * 0.5f + 0.5f;
	vec3 shadowColor = texture(uShadowMap, projShadowCoord.xy).xyz;

	color = vec4(diffuseReflectance * shadeSpotlight(vec3(vPosition)) * vec3(1.0f - shadowColor.z), 1.0f);
}
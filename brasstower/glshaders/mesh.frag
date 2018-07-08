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

vec2 poissonDisk[4] = vec2[](
	vec2( -0.94201624, -0.39906216 ),
 	vec2( 0.94558609, -0.76890725 ),
 	vec2( -0.094184101, -0.92938870 ),
 	vec2( 0.34495938, 0.29387760 )
);

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
	float bias = 0.009f;
	float shadowColor = 0.0f;
	for (int i = 0;i < 4;i++)
	{
		vec2 offset = poissonDisk[i] / 500.0f;
		shadowColor += (10000.f - texture(uShadowMap, gShadowCoord.xy + offset).r) > gShadowCoord.z - bias ? 0.25f : 0.0f;
	}
	return shadowColor;
}

void main()
{
	vec3 ambient = uColor * 0.5f;
	color = vec4(visibility() * shadeSpotlight(gPosition, gNormal) * uColor + ambient, 1.0f);
}
#version 450 core

layout(location = 0) out vec4 color;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vColor;
in vec3 vParticleCentroid;
in vec4 vShadowCoord;

uniform vec3 uCameraPosition;
uniform float uRadius;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform vec2 uLightThetaMinMax;

uniform sampler2D uShadowMap;

vec2 poissonDisk[4] = vec2[](
	vec2( -0.94201624, -0.39906216 ),
 	vec2( 0.94558609, -0.76890725 ),
 	vec2( -0.094184101, -0.92938870 ),
 	vec2( 0.34495938, 0.29387760 )
);

vec3 shadeSpotlight(vec3 position, vec3 normal)
{
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
	float bias = 0.001f;
	float shadowColor = 0.0f;
	for (int i = 0;i < 4;i++)
	{
		vec2 offset = poissonDisk[i] / 500.0f;
		shadowColor += (10.f - texture(uShadowMap, vShadowCoord.xy + offset).r) > vShadowCoord.z - bias ? 0.25f : 0.0f;
	}
	return shadowColor;
}

void main()
{
	vec3 diff = vec3(0, 20, 0) - vPosition;
	float dist2 = dot(diff, diff);

	// ray - sphere intersection test
	vec3 cameraToCentroid = vParticleCentroid - uCameraPosition;
	float h2 = dot(cameraToCentroid, cameraToCentroid);
	vec3 cameraToPosition = vPosition - uCameraPosition;
	float d2 = dot(cameraToPosition, cameraToPosition);
	float dotResult = dot(cameraToCentroid, cameraToPosition);
	float cosTheta2 = dotResult * dotResult / h2 / d2;
	if (h2 * (1.0 - cosTheta2) >= uRadius * uRadius) discard;

	vec3 diffuse = visibility() * shadeSpotlight(vPosition, normalize(vNormal)) * vColor;
	vec3 ambient = vColor * 0.01f;
	color = vec4(diffuse + ambient, 1.0f);
	color = pow(color, vec4(1.0/2.2));
}
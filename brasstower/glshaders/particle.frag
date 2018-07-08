#version 450 core

layout(location = 0) out vec4 color;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vColor;
in vec3 vParticleCentroid;

uniform vec3 uCameraPosition;
uniform float uRadius;

uniform vec3 uLightPosition;
uniform vec3 uLightDir;
uniform vec3 uLightIntensity;
uniform vec2 uLightThetaMinMax;

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

	vec3 diffuse = shadeSpotlight(vPosition, normalize(vNormal)) * vColor;
	vec3 ambient = vColor * 0.01f;
	color = vec4(diffuse + ambient, 1.0f);
	color = pow(color, vec4(1.0/2.2));
}
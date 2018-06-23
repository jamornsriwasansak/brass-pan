#version 450 core

layout(location = 0) out vec4 fPosition;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vColor;
in vec3 vParticleCentroid;

uniform vec3 uCameraPosition;
uniform float uRadius;

const vec3 pointLightPosition = vec3(1, 0, 0);

void main()
{
	vec3 diff = pointLightPosition - vPosition;
	float dist2 = dot(diff, diff);

	// ray - sphere intersection test
	vec3 cameraToCentroid = vParticleCentroid - uCameraPosition;
	float h2 = dot(cameraToCentroid, cameraToCentroid);
	vec3 cameraToPosition = vPosition - uCameraPosition;
	float d2 = dot(cameraToPosition, cameraToPosition);
	float dotResult = dot(cameraToCentroid, cameraToPosition);
	float cosTheta2 = dotResult * dotResult / h2 / d2;
	if (h2 * (1.0 - cosTheta2) >= uRadius * uRadius) discard;

	vec3 shadingNormal = normalize(vNormal) * vColor;
	fPosition = vec4(shadingNormal, 1.0f);
}
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

	vec3 ambient = vColor * 0.4f;
	vec3 diffuse = max(-dot(normalize(vNormal), normalize(diff)), 0.f) * vColor / dist2;
	vec3 color = ambient + diffuse;
	fPosition = vec4(color, 1.0f);
}
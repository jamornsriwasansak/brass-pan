#version 450 core

layout(location = 0) out vec3 code;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vParticleCentroid;
in flat int vInstanceId;

uniform vec3 uCameraPosition;
uniform float uRadius;

void main()
{
	// ray - sphere intersection test
	vec3 cameraToCentroid = vParticleCentroid - uCameraPosition;
	float h2 = dot(cameraToCentroid, cameraToCentroid);
	vec3 cameraToPosition = vPosition - uCameraPosition;
	float d2 = dot(cameraToPosition, cameraToPosition);
	float dotResult = dot(cameraToCentroid, cameraToPosition);
	float cosTheta2 = dotResult * dotResult / h2 / d2;
	if (h2 * (1.0 - cosTheta2) >= uRadius * uRadius) discard;

	int rInt = vInstanceId / (256 * 256);
	float rFloat = rInt / 255.0f;

	int gInt = (vInstanceId % 65536) / (256);
	float gFloat = gInt / 255.0f;

	int bInt = vInstanceId % 256;
	float bFloat = bInt / 255.0f;

	code = vec3(rFloat, gFloat, bFloat);
}
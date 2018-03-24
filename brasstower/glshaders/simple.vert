#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vParticleCentroid;

uniform mat4 uMVP;
uniform float uRadius;

layout (std430, binding=0) buffer ParticlePositions
{
	vec4 positions[];
};

void main()
{
	vParticleCentroid = vec3(positions[gl_InstanceID]);
	vPosition = vertexPos * uRadius + vParticleCentroid;
	vNormal = vertexPos;
	gl_Position = uMVP * vec4(vPosition, 1.0);
}

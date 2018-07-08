#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vColor;
out vec3 vParticleCentroid;

uniform mat4 uMVP;
uniform float uRadius;

struct ParticleRenderInfo
{
	vec3 position;
	int phase;
};

layout (std430, binding=0) buffer ParticlePositions
{
	ParticleRenderInfo particles[];
};

const vec3 ParticleDefaultColors[4] = vec3[](
	vec3(68.0, 132.0, 206.0) / 256.0,
	vec3(217.0, 217.0, 217.0) / 256.0,
	vec3(249.0, 207.0, 0.0) / 256.0,
	vec3(241.0, 159.0, 77.0) / 256.0
);

void main()
{
	vParticleCentroid = vec3(particles[gl_InstanceID].position);
	vPosition = vertexPos * uRadius + vParticleCentroid;
	vNormal = vertexPos;
	vColor = ParticleDefaultColors[(particles[gl_InstanceID].phase + 4) % 4];
	gl_Position = uMVP * vec4(vPosition, 1.0);
}

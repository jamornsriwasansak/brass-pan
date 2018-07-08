#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vColor;
out vec3 vParticleCentroid;
out vec4 vShadowCoord;

uniform mat4 uMVP;
uniform mat4 uShadowMatrix;
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
	pow(vec3(00.0, 206.0, 206.0) / 256.0, vec3(2.2f)),
	pow(vec3(112.0, 217.0, 153.0) / 256.0, vec3(2.2f)),
	pow(vec3(249.0, 207.0, 0.0) / 256.0, vec3(2.2f)),
	pow(vec3(241.0, 159.0, 77.0) / 256.0, vec3(2.2f))
);

void main()
{
	vParticleCentroid = vec3(particles[gl_InstanceID].position);
	vPosition = vertexPos * uRadius + vParticleCentroid;
	vNormal = vertexPos;
	vColor = ParticleDefaultColors[(particles[gl_InstanceID].phase + 4) % 4];
	vShadowCoord = uShadowMatrix * vec4(vPosition, 1.0);
	vShadowCoord = vShadowCoord / vShadowCoord.w * 0.5f + 0.5f;
	gl_Position = uMVP * vec4(vPosition, 1.0);
}

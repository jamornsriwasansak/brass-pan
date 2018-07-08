#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec3 vPosition;

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

void main()
{
	vec3 center = vec3(particles[gl_InstanceID].position);
	vec3 position = vertexPos * uRadius + center;
	gl_Position = uShadowMatrix * vec4(position, 1.0);
	vPosition = gl_Position.xyz / gl_Position.w * 0.5f + 0.5f;
}
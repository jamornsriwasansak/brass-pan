#version 450 core

layout(location = 0) in vec3 vertexPos;

uniform mat4 uShadowMatrix;
uniform int uRigidBodyId;

layout (std430, binding=0) buffer ParticlePositions
{
	mat4 modelMatrices[];
};

void main()
{
	gl_Position = uShadowMatrix * modelMatrices[uRigidBodyId] * vec4(vertexPos, 1.0);
}
#version 450 core

layout(location = 0) in vec3 vertexPos;
uniform mat4 uVPMatrix;
uniform int uRigidBodyId;
out vec4 vPosition;

layout (std430, binding=0) buffer ParticlePositions
{
	mat4 modelMatrices[];
};

void main()
{
	//vPosition = uModelMatrix * vec4(vertexPos, 1.0);
	//gl_Position = uVPMatrix * vPosition;
	//gl_Position = uVPMatrix * vec4(vertexPos, 1.0);
	gl_Position = uVPMatrix * modelMatrices[uRigidBodyId] * vec4(vertexPos, 1.0);
}
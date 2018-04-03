#version 450 core

layout(location = 0) in vec3 vertexPos;
uniform mat4 uVPMatrix;
uniform mat4 uShadowMatrix;
uniform int uRigidBodyId;

out vec3 vPosition;
out vec4 vShadowCoord;

layout (std430, binding=0) buffer ParticlePositions
{
	mat4 modelMatrices[];
};

void main()
{
	mat4 modelMatrix = modelMatrices[uRigidBodyId];
	vec4 pos = modelMatrix * vec4(vertexPos, 1.0);
	vShadowCoord = uShadowMatrix * pos;
	vShadowCoord = vShadowCoord / vShadowCoord.w * 0.5f + 0.5f;
	gl_Position = uVPMatrix * pos;
	vPosition = pos.xyz;
}
#version 450 core

layout(location = 0) in vec3 vertexPos;

out vec3 vPosition;
out vec3 vNormal;

uniform mat4 uMVP;
uniform float uRadius;

layout (std430, binding=0) buffer ParticlePositions
{
	vec4 positions[];
};

void main()
{
	vPosition = vertexPos * uRadius + vec3(positions[gl_InstanceID]);
	vNormal = vertexPos;
	gl_Position = uMVP * vec4(vPosition, 1.0);
}

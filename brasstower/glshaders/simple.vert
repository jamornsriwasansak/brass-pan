#version 450 core

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec2 uv;

out vec2 vUv;
out vec3 vNormal;

uniform mat4 uMVP;
uniform float uRadius;

layout (std430, binding=0) buffer ParticlesInfo
{
	vec3 positions[];
};

void main()
{
    vUv = uv;
	//gl_Position = uMVP * vec4(vertexPos * uRadius + vec3(gl_InstanceID), 1.0);
	gl_Position = uMVP * vec4(vertexPos * uRadius + vec3(positions[gl_InstanceID]) / 100.0, 1.0);
	vNormal = vertexPos;
	//gl_Position = vec4(vertexPos, 1.0);
}

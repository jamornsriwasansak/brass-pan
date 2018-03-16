#version 450 core

layout(location = 0) in vec3 vertexPos;
uniform mat4 uMVP;

out vec3 vPosition;

void main()
{
	vPosition = vertexPos * 1000;
	gl_Position = uMVP * vec4(vPosition, 1.0);
}
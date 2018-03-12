#version 450 core

layout(location = 0) in vec3 vertexPos;
uniform mat4 uMVP;

void main()
{
	gl_Position = uMVP * vec4(vertexPos * 1000, 1.0);
}
#version 450 core

layout(location = 0) in vec3 vertexPos;
uniform mat4 uVPMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uShadowMatrix;

out vec4 vPosition;
out vec4 vShadowCoord;

void main()
{
	vPosition = uModelMatrix * vec4(vertexPos * 100.0, 1.0);
	gl_Position = uVPMatrix * vPosition;
	vShadowCoord = uShadowMatrix * vPosition;
}
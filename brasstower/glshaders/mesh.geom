#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vPosition[];
in vec4 vShadowCoord[];
out vec3 gNormal;
out vec3 gPosition;
out vec4 gShadowCoord;

void main()
{
	gNormal = normalize(cross(vPosition[1] - vPosition[0], vPosition[2] - vPosition[0]));
	for (int i = 0;i < gl_in.length();i++)
	{
		gPosition = vPosition[i];
		gShadowCoord = vShadowCoord[i];
		gl_Position = gl_in[i].gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}
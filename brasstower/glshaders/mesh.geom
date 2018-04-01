#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vPosition[];
out vec3 gNormal;
out vec3 gPosition;

void main()
{
	gNormal = normalize(cross(vPosition[1] - vPosition[0], vPosition[2] - vPosition[0]));
	for (int i = 0;i < gl_in.length();i++)
	{
		gPosition = vPosition[i];
		gl_Position = gl_in[i].gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}
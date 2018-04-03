#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vPosition[];
in vec4 vShadowCoord[];
out vec3 gNormal;
out vec3 gPosition;
out vec4 gShadowCoord;

vec2 poissonDisk[4] = vec2[](
	vec2( -0.94201624, -0.39906216 ),
 	vec2( 0.94558609, -0.76890725 ),
 	vec2( -0.094184101, -0.92938870 ),
 	vec2( 0.34495938, 0.29387760 )
);

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
#version 450 core

layout(location = 0) out vec4 color;

in vec3 vPosition;

void main()
{
	// discretize position
	ivec3 discretized = ivec3(ceil(vPosition * 10));
	// grid code if code = (1, 1, 1) means that it's odd block else it's even block
	vec3 code = vec3((discretized.x + discretized.y + discretized.z) % 2);
	// map code to diff reflectance
	vec3 diffuseReflectance = (code + 1.5f) / 4.0;
	color = vec4(diffuseReflectance, 1.0f);
}
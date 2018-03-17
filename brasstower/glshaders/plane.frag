#version 450 core

layout(location = 0) out vec4 color;

in vec4 vPosition;
uniform vec3 uCameraPos;

void main()
{
	// discretize position
	ivec4 discretized = ivec4(ceil((vPosition - vec4(0.002)) * 10));
	// grid code if code = (1, 1, 1) means that it's odd block else it's even block
	vec3 code = vec3((discretized.x + discretized.y + discretized.z) % 2);
	// map code to diff reflectance
	vec3 diffuseReflectance = (code + 1.5f) / 4.0;

	// color falloff (to avoid aliasing issues)
	vec3 diffPos = vec3(vPosition) - uCameraPos;
	float dist2 = dot(diffPos, diffPos);
	float attenuation = min(400.0 / dist2, 1.0f);

	color = vec4(diffuseReflectance, 1.0f) * attenuation;
}
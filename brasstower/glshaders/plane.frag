#version 450 core

layout(location = 0) out vec4 color;

in vec4 vPosition;
in vec4 vShadowCoord;

uniform vec3 uCameraPos;
uniform sampler2D uShadowMap;

void main()
{
	// discretize position
	ivec4 discretized = ivec4(ceil(vPosition));
	// grid code if code = (1, 1, 1) means that it's odd block else it's even block
	vec3 code = vec3((discretized.x + discretized.y + discretized.z) % 2);
	// map code to diff reflectance
	vec3 diffuseReflectance = (code + 5.5f) / 10.0;

	// color falloff (to avoid aliasing issues)
	vec3 diffPos = vec3(vPosition) - uCameraPos;
	float dist2 = dot(diffPos, diffPos);
	float attenuation = min(400.0 / dist2, 1.0f);

	vec3 projShadowCoord = vShadowCoord.xyz / vShadowCoord.w * 0.5f + 0.5f;
	vec3 shadowColor = texture(uShadowMap, projShadowCoord.xy).xyz;

	color = vec4(diffuseReflectance * attenuation * vec3(1.0f - shadowColor.z), 1.0f);
}
#pragma once

#include <iostream>
#include "GL/glew.h"

void CheckFramebuffer()
{
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		//std::cout << glewGetErrorString(status) << std::endl;
		switch (status)
		{
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			std::cout << "incomplete layer targets" << std::endl;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			std::cout << "unsupported" << std::endl;
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			std::cout << "incomplete attachment" << std::endl;
			break;
		default:
			std::cout << "super error" << std::endl;
			break;
		}
		throw std::exception("errorororororor");
	}
}
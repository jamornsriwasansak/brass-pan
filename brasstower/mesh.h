#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <map>
#include <memory>

#include <vector_types.h>

#include "glm/glm.hpp"
#include "glm/ext.hpp"
#include "opengl/buffer.h"

struct Mesh
{
	static std::vector<std::shared_ptr<Mesh>> Load(const std::string & filepath) {
		const unsigned int aiProcesses = aiProcess_Triangulate
			| aiProcess_GenSmoothNormals
			| aiProcess_JoinIdenticalVertices
			| aiProcessPreset_TargetRealtime_Fast;

		std::vector<std::shared_ptr<Mesh>> result;

		// load aiScene
		Assimp::Importer importer;
		const aiScene * scene = importer.ReadFile(filepath.c_str(), aiProcesses);
		if (scene == nullptr) {
			std::cerr << "Impossible to load the scene: " << filepath << "\n";
			assert(false);
		}

		// populate rtmesh result vector
		for (size_t iMesh = 0;iMesh < scene->mNumMeshes;iMesh++)
		{
			std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();

			const auto aiSceneMesh = scene->mMeshes[iMesh];
			const size_t numVertices = aiSceneMesh->mNumVertices;

			assert(aiSceneMesh->HasNormals());

			for (size_t iVert = 0;iVert < numVertices;iVert++)
			{
				// position
				{
					const float x = aiSceneMesh->mVertices[iVert][0];
					const float y = aiSceneMesh->mVertices[iVert][1];
					const float z = aiSceneMesh->mVertices[iVert][2];

					mesh->mVertices.push_back(x);
					mesh->mVertices.push_back(y);
					mesh->mVertices.push_back(z);
				}

				// normal
				{
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][0]);
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][1]);
					mesh->mNormals.push_back(aiSceneMesh->mNormals[iVert][2]);
				}

				// tex coords
				if (aiSceneMesh->HasTextureCoords(0))
				{
					mesh->mTexCoords.push_back(aiSceneMesh->mTextureCoords[0][iVert][0]);
					mesh->mTexCoords.push_back(aiSceneMesh->mTextureCoords[0][iVert][1]);
				}
				else
				{
					mesh->mTexCoords.push_back(0.0f);
					mesh->mTexCoords.push_back(0.0f);
				}
			}

			const size_t numTriangles = aiSceneMesh->mNumFaces;
			for (size_t iIdx = 0;iIdx < numTriangles;iIdx++)
			{
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[0]);
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[1]);
				mesh->mTriIndices.push_back(aiSceneMesh->mFaces[iIdx].mIndices[2]);
			}

			mesh->mNumVertices		= numVertices;
			mesh->mNumTriangles		= numTriangles;
			//mesh->createOpenglBuffer();

			result.push_back(mesh);
		}

		return result;
	}

	Mesh(): mNumVertices(0)
	{}

	void applyTransform(const glm::mat4 & transformMatrix)
	{
		glm::vec3 * vertices = reinterpret_cast<glm::vec3*>(mVertices.data());
		glm::vec3 * normals = reinterpret_cast<glm::vec3*>(mNormals.data());

		glm::mat4 transformMatrix_Normal = glm::inverseTranspose(transformMatrix);

		for (size_t i = 0;i < mNumVertices;i++)
		{
			vertices[i] = transformMatrix * glm::vec4(vertices[i], 1.0f);
			normals[i] = transformMatrix_Normal * glm::vec4(normals[i], 0.0f);
			normals[i] = glm::normalize(normals[i]);
		}
	}

	void createOpenglBuffer()
	{
		assert(mTexCoords.size() / 2 == mVertices.size() / 3);

		mGl.mVerticesBuffer = std::make_shared<OpenglBuffer>();
		mGl.mNormalsBuffer = std::make_shared<OpenglBuffer>();
		mGl.mIndicesBuffer = std::make_shared<OpenglBuffer>();

		glNamedBufferData(mGl.mVerticesBuffer->mHandle, sizeof(float) * mNumVertices * 3, &(mVertices[0]), GL_STATIC_DRAW);
		glNamedBufferData(mGl.mNormalsBuffer->mHandle, sizeof(float) * mNumVertices * 3, &(mNormals[0]), GL_STATIC_DRAW);
		glNamedBufferData(mGl.mIndicesBuffer->mHandle, sizeof(uint32_t) * mNumTriangles * 3, &(mTriIndices[0]), GL_STATIC_DRAW);
	}

	int32_t						mNumVertices;
	int32_t						mNumTriangles;
	int32_t						mMatIndex;

	std::vector<float>			mVertices;
	std::vector<float>			mNormals;
	std::vector<float>			mTexCoords;
	std::vector<int32_t>		mTriIndices;

	struct OpenglMeshBuffers
	{
		std::shared_ptr<OpenglBuffer> mVerticesBuffer;
		std::shared_ptr<OpenglBuffer> mNormalsBuffer;
		std::shared_ptr<OpenglBuffer> mIndicesBuffer;
	} mGl;
};

struct MeshGenerator
{
	static std::shared_ptr<Mesh> Box()
	{
		std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
		const float vertices[] = { 1.0f,-1.0f,-1.0f,1.0f,-1.0f,1.0f,-1.0f,-1.0f,1.0f,-1.0f,-1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,1.0f,-1.0f,1.0f,1.0f,-1.0f,1.0f,-1.0f };
		const float texcoords[] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
		const float normals[] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
		const int32_t indices[] = { 0,3,0,7,5,4,4,1,0,5,2,1,2,7,3,0,7,4,1,2,3,7,6,5,4,5,1,5,6,2,2,6,7,0,3,7 };
		mesh->mNumVertices = 8;
		mesh->mNumTriangles = 12;
		mesh->mVertices = std::vector<float>(vertices, vertices + sizeof(vertices) / sizeof(vertices[0]));
		mesh->mNormals = std::vector<float>(normals, normals + sizeof(normals) / sizeof(normals[0]));
		mesh->mTexCoords = std::vector<float>(texcoords, texcoords + sizeof(texcoords) / sizeof(texcoords[0]));
		mesh->mTriIndices = std::vector<int32_t>(indices, indices + sizeof(indices) / sizeof(indices[0]));
		return nullptr;
	}

	static std::shared_ptr<Mesh> Plane()
	{
		std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
		const float vertices[] = { -1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,-1.0f,0.0f,-1.0f,1.0f,0.0f,-1.0f };
		const float texcoords[] = { 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
		const float normals[] = { 0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f };
		const int32_t indices[] = { 1,2,0,1,3,2 };
		mesh->mNumVertices = 12;
		mesh->mNumTriangles = 2;
		mesh->mVertices = std::vector<float>(vertices, vertices + sizeof(vertices) / sizeof(vertices[0]));
		mesh->mNormals = std::vector<float>(normals, normals + sizeof(normals) / sizeof(normals[0]));
		mesh->mTexCoords = std::vector<float>(texcoords, texcoords + sizeof(texcoords) / sizeof(texcoords[0]));
		mesh->mTriIndices = std::vector<int32_t>(indices, indices + sizeof(indices) / sizeof(indices[0]));
		return mesh;
	}
};
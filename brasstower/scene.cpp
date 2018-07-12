#include "scene.h"
#include "cuda/cudamatrix.cuh"

int2 Scene::addCloth(const float3 & startPosition,
					 const float3 & stepX,
					 const float3 & stepY,
					 const int numJointX,
					 const int numJointY,
					 const float massPerParticle,
					 const float kStiffness,
					 const float kBending,
					 const bool isSelfPenetratable)
{
	float lengthX = length(stepX);
	float lengthY = length(stepY);
	float lengthDiag = length(stepX + stepY);

	for (int x = 0; x < numJointX; x++)
	{
		for (int y = 0; y < numJointY; y++)
		{
			// positions
			positions.push_back(startPosition + stepY * float(y) + stepX * float(x));
			if (isSelfPenetratable)
			{
				phases.push_back(solidPhaseCounter++);
			}
			else
			{
				phases.push_back(solidPhaseCounter);
			}

			int p1 = y * numJointX + x;
			int p2 = y * numJointX + x + 1;
			int p3 = (y + 1) * numJointX + x;
			int p4 = (y + 1) * numJointX + x + 1;

			// face
			{
				if (x < numJointX - 1 && y < numJointY - 1)
				{
					faces.push_back(make_int3(p1, p2, p3));
					faces.push_back(make_int3(p2, p3, p4));
				}
			}

			// distance pairs
			{
				// horizontal
				if (x < numJointX - 1)
				{
					//distanceConstraints.push_back(DistanceConstraint(p1, p2, lengthX, kStiffness));
					distancePairs.push_back(make_int2(p1, p2));
					distanceParams.push_back(make_float2(lengthX, kStiffness));
				}

				// vertical
				if (y < numJointY - 1)
				{
					//distanceConstraints.push_back(DistanceConstraint(p1, p3, lengthY, kStiffness));
					distancePairs.push_back(make_int2(p1, p3));
					distanceParams.push_back(make_float2(lengthY, kStiffness));
				}

				// diagonal1
				if (x < numJointX - 1 && y < numJointY - 1)
				{
					//distanceConstraints.push_back(DistanceConstraint(p1, p4, lengthDiag, kStiffness));
					//distanceConstraints.push_back(DistanceConstraint(p2, p3, lengthDiag, kStiffness));
					distancePairs.push_back(make_int2(p1, p4));
					distanceParams.push_back(make_float2(lengthDiag, kStiffness));
					distancePairs.push_back(make_int2(p2, p3));
					distanceParams.push_back(make_float2(lengthDiag, kStiffness));
				}
			}


			// hack bending
			{
				int q[3][3];
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						q[i + 1][j + 1] = (y + i) * numJointX + x + j;
					}
				}

				if (x > 0 && y > 0 && x < numJointX - 1 && y < numJointY - 1)
				{
					for (int i = 0; i < 3; i++)
					{
						//distanceConstraints.push_back(DistanceConstraint(q[i][0], q[i][2], lengthX * 2, kBending));
						distancePairs.push_back(make_int2(q[i][0], q[i][2]));
						distanceParams.push_back(make_float2(lengthX * 2, kBending));
					}

					for (int i = 0; i < 3; i++)
					{
						//distanceConstraints.push_back(DistanceConstraint(q[0][i], q[2][i], lengthY * 2, kBending));
						distancePairs.push_back(make_int2(q[0][i], q[2][i]));
						distanceParams.push_back(make_float2(lengthY * 2, kBending));
					}

				}
			}

			// bending constraints
			/*
			{
				if (x < numJointX - 1 && y < numJointY - 1)
				{
					if ((x + y) % 2)
					{
						bendings.push_back(glm::int4(p3, p2, p1, p4));
					}
					else
					{
						bendings.push_back(glm::int4(p1, p4, p3, p2));
					}
				}
			}
			*/
		}
	}

	return int2();
}

int2 Scene::addGranularsBlock(const uint3 & dimension, const float3 & startPosition, const float3 & step, const float massPerParticle)
{
	int startIndex = positions.size();
	int groupId = groupIdCounter++;
	for (int i = 0; i < dimension.x; i++)
		for (int j = 0; j < dimension.y; j++)
			for (int k = 0; k < dimension.z; k++)
			{
				float3 position = startPosition + step * make_float3(i, j, k);
				positions.push_back(position);
				masses.push_back(massPerParticle);
				phases.push_back(solidPhaseCounter++);
				groupIds.push_back(groupId);
			}
	int endIndex = positions.size();
	return make_int2(startIndex, endIndex);
}

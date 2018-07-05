#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "cuda/helper.cuh"
#include "cuda/cudamatrix.cuh"
#include "cuda/cudaglm.cuh"

#ifndef __INTELLISENSE__
#include <cub/cub.cuh>
#endif

#include "scene.h"

#define NUM_MAX_PARTICLE_PER_CELL 15
#define FRICTION_STATIC 0.0f
#define FRICTION_DYNAMICS 0.0f
#define MASS_SCALING_CONSTANT 2 // refers to k in equation (21)
#define PARTICLE_SLEEPING_EPSILON 0.00
#define NUM_MAX_PARTICLE_PER_RIGID_BODY 64

#define MATH_PI 3.141592
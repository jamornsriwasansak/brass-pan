#include "kernel/cubrasstower.cuh"

__inline__
void GetNumBlocksNumThreads(int * numBlocks, int * numThreads, int k)
{
	*numThreads = 128;
	*numBlocks = static_cast<int>(ceil((float)k / (float)(*numThreads)));
}

template <typename T>
__inline__
void print(T * dev, int size)
{
	T * tmp = (T *)malloc(sizeof(T) * size);
	cudaMemcpy(tmp, dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << tmp[i];
		if (i != size - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
	free(tmp);
}

template <>
__inline__
void print<float3>(float3 * dev, int size)
{
	float3 * tmp = (float3 *)malloc(sizeof(float3) * size);
	cudaMemcpy(tmp, dev, sizeof(float3) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << "(" << tmp[i].x << " " << tmp[i].y << " " << tmp[i].z << ")";
		if (i != size - 1)
			std::cout << ",";
	}
	std::cout << std::endl;
	free(tmp);
}

template <typename T>
__inline__
void printPair(T * dev, int size)
{
	T * tmp = (T *)malloc(sizeof(T) * size);
	cudaMemcpy(tmp, dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		if (tmp[i] != -1)
		{
			std::cout << i << ":" << tmp[i];
			if (i != size - 1)
				std::cout << ",";
		}
	}
	std::cout << std::endl;
	free(tmp);
}

// PARTICLE SYSTEM //

__inline__ __device__ void
atomicAdd(float3 * arr, int index, const float3 val)
{
	atomicAdd(&arr[index].x, val.x);
	atomicAdd(&arr[index].y, val.y);
	atomicAdd(&arr[index].z, val.z);
}

__inline__ __global__ void
increment(int * __restrict__ x)
{
	atomicAdd(x, 1);
}

__inline__ __global__ void
setDevArr_devIntPtr(int * __restrict__ devArr,
					const int * __restrict__ value,
					const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = *value;
}

__inline__ __global__ void
setDevArr_int(int * __restrict__ devArr,
			  const int value,
			  const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__inline__ __global__ void
setDevArr_float(float * __restrict__ devArr,
				const float value,
				const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__inline__ __global__ void
setDevArr_int2(int2 * __restrict__ devArr,
			   const int2 value,
			   const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = value;
}

__inline__ __global__ void
setDevArr_float3(float3 * __restrict__ devArr,
				 const float3 val,
				 const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = val;
}

__inline__ __global__ void
setDevArr_float4(float4 * __restrict__ devArr,
				 const float4 val,
				 const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = val;
}

__inline__ __global__ void
setDevArr_counterIncrement(int * __restrict__ devArr,
						   int * counter,
						   const int incrementValue,
						   const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] = atomicAdd(counter, incrementValue);
}

__inline__ __global__ void
accDevArr_float3(float3 * __restrict__ devArr,
				 const float3 * __restrict__ delta,
				 const int numValues)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numValues) { return; }
	devArr[i] += delta[i];
}

__inline__ __global__ void
initPositionBox(float3 * __restrict__ positions,
				int * __restrict__ phases,
				int * phaseCounter,
				const int3 dimension,
				const float3 startPosition,
				const float3 step,
				const int numParticles)
{
	int i = threadIdx.x + __mul24(blockIdx.x, blockDim.x);
	if (i >= numParticles) { return; }
	int x = i % dimension.x;
	int y = (i / dimension.x) % dimension.y;
	int z = i / (dimension.x * dimension.y);
	positions[i] = make_float3(x, y, z) * step + startPosition;
	phases[i] = atomicAdd(phaseCounter, 1);
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define POP_PER_ISLAND 1024
#define NUM_ISLANDS 7

__global__ void evolve(Individual* pop, Individual* boat)
{
	__shared__ Individual oldPop[POP_PER_ISLAND];
	__shared__ Individual newPop[POP_PER_ISLAND];
	int g;
	
	oldPop[threadIdx.x] = pop[threadIdx.x + blockIdx.x * blockDim.x];
	boat[blockIdx.x] = oldPop[0];
	
	curand_init(
	
	for(g = 0; g < 100; g++)
	{
		evaluate(oldPop[threadIdx.x]);
		
		int a = pharaohRand() * RAND_MAX % POP_PER_ISLAND;
		int b = pharaohRand() * RAND_MAX % POP_PER_ISLAND;
		newPop[threadIdx.x] = arena(oldPop[a], oldPop[b]);
		
		mutate(newPop[threadIdx.x]);
		//crossover(newPop[threadIdx.x]);

		boat[(bockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
		__syncthreads()
		newPop[0] = boat[blockIdx.x];
		
		__syncthreads();
		
		oldPop[threadIdx.x] = newPop[threadIdx.x];
	}
}

__device__ Individual arena(Individual a, Individual b)
{
	if(a.fitness > b.fitness)
	{
		return a;
	}
	else
	{
		return b;
	}
}

__device__ float pharaohRand()
{
	int seed = clock();
	curandState_t myState;
	curand_init(seed, blockIdx.x * blockDim.x, threadIdx.x, &myState);
	
	return curand_uniform(curandState_t);
}

int main void()
{
	Individual* pop;
	Individual* d_pop;
	
	
	<<<NUM_ISLANDS, POP_PER_ISLAND>>> evolve(d_pop, d_boat);
}
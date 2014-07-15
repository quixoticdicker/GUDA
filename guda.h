#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "pharaohrand.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
    }
}

#ifndef NUM_ISLANDS
#define NUM_ISLANDS 21
#endif

#ifndef POP_PER_ISLAND
#define POP_PER_ISLAND 256
#endif

struct Individual {
    MEMBERS

    float fitness;

    __device__ void master_init();
    __device__ void init();
    __device__ void master_mutate();
    __device__ void mutate();
    __device__ void master_evaluate();
    __device__ float evaluate();
    __host__ __device__ float getFitness();
    __device__ void destroy() {};
};

__device__ void Individual::master_init()
{
    init();
    master_evaluate();
}

__device__ void Individual::master_mutate()
{
#ifdef SELECT_MUTATION
    Individual old = *this;
#endif
    mutate();
    master_evaluate();
#ifdef SELECT_MUTATION
    if (old.getFitness() > getFitness())
    {
	*this = old;
    }
#endif
}

__device__ void Individual::master_evaluate() {
    fitness = evaluate();
}

__host__ __device__ float Individual::getFitness()
{
    return fitness;
}

__device__ Individual arena(Individual a, Individual b)
{
    if (a.fitness > b.fitness)
    {
	return a;
    }
    else
    {
	return b;
    }
}

__global__ void evolve(Individual* pop, Individual* boat, int generations, bool first)
{
    __shared__ Individual oldPop[POP_PER_ISLAND];
    __shared__ Individual newPop[POP_PER_ISLAND];
	
	if (first)
		oldPop[threadIdx.x].master_init();
	else
		oldPop[threadIdx.x] = pop[threadIdx.x + blockIdx.x * blockDim.x];
    __syncthreads();

    boat[blockIdx.x] = oldPop[0];
	
    for (int g = 0; g < generations; g++)
    {
		int a = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
		int b = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
		newPop[threadIdx.x] = arena(oldPop[a], oldPop[b]);
	
		newPop[threadIdx.x].master_mutate();
		//crossover(newPop[threadIdx.x]);

		__syncthreads();
	
		boat[(blockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
		__syncthreads();
		newPop[0] = boat[blockIdx.x];
		
		__syncthreads();
		
		oldPop[threadIdx.x] = newPop[threadIdx.x];
		__syncthreads();
    }

    pop[threadIdx.x + blockIdx.x * blockDim.x] = newPop[threadIdx.x];
	
    oldPop[threadIdx.x].destroy();
    newPop[threadIdx.x].destroy();
}
/*
float avg_value(Individual* pop, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
	sum += pop[i].value;
    return sum / n;
}*/

float avg_fitness(Individual* pop, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
	sum += pop[i].fitness;
    return sum / n;
}

Individual run(int generations, int gens_each = 500)
{
    Individual pop[NUM_ISLANDS * POP_PER_ISLAND];
    Individual *d_pop;
    gpuErrchk(cudaMalloc((void**) &d_pop,
			 sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND)));
    gpuErrchk(cudaMemcpy(d_pop, pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
			 cudaMemcpyHostToDevice));

    Individual boat[NUM_ISLANDS];
    Individual *d_boat;
    gpuErrchk(cudaMalloc((void**) &d_boat, sizeof(Individual) * NUM_ISLANDS));
    gpuErrchk(cudaMemcpy(d_boat, boat, sizeof(Individual) * NUM_ISLANDS,
			 cudaMemcpyHostToDevice));

	int i = 0;
    for (i = 0; i < generations / gens_each; i++)
    {
		evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat, gens_each, i == 0);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
    }
    evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat, generations % gens_each, i == 0);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(pop, d_pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
			 cudaMemcpyDeviceToHost));

    float max_fitness = pop[0].fitness;
    int best_index = 0;
    for (int i = 1; i < NUM_ISLANDS * POP_PER_ISLAND; i++)
    {
	if (pop[i].fitness > max_fitness)
	{
	    best_index = i;
	    max_fitness = pop[i].fitness;
	}
    }

    return pop[best_index];
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "pharaohrand.h"

#define POP_PER_ISLAND 256
#define NUM_ISLANDS 7
#define RATE 0.03

struct Individual {
    long value;
    float fitness;
	bool fitFlag;

    Individual();
    void mutate();
    void evaluate();
	float getFitness();
};

__device__ Individual::Individual() {
    // replace with something random
    value = 0;
	int i;
	for(i = 0; i < sizeof(long) * 8; i++)
	{
		if(pharaohRand() > 0.5)
		{
			value++;
		}
		value <<= 1;
	}
	evaluate();
}

__device__ void Individual::mutate()
{
    // replace with something random
	long mutagen = 0;
	int i;
	for(i = 0; i < sizeof(long) * 8; i++)
	{
		if(pharaohRand() < RATE)
		{
			mutagen++;
		}
		mutagen <<= 1;
	}
	value ^= mutagen;
	fitFlag = false;
}

__device__ void Individual::evaluate()
{
    float tempFitness = 0;
	int i;
	long tempVal = value;
	for(i = 0; i < sizeof(long) * 8; i++)
	{
		if(tempVal & 1 == 1)
		{
			tempFitness++;
		}
		tempVal >>= 1;
	}
	fitness = tempFitness;
	fitFlag = true;
}

__device__ float Individual::getFitness()
{
	if(!fitFlag) evaluate();
	
	return fitness;	
}

__device__ Individual arena(Individual a, Individual b)
{
    if (a.getFitness() > b.getFitness())
    {
		return a;
    }
    else
    {
		return b;
    }
}

__global__ void evolve(Individual* pop, Individual* boat)
{
    __shared__ Individual oldPop[POP_PER_ISLAND];
    __shared__ Individual newPop[POP_PER_ISLAND];
    int g;
	
    oldPop[threadIdx.x] = pop[threadIdx.x + blockIdx.x * blockDim.x];
    boat[blockIdx.x] = oldPop[0];
	
    for (g = 0; g < 100; g++)
    {
	oldPop[threadIdx.x].evaluate();
		
	int a = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
	int b = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
	newPop[threadIdx.x] = arena(oldPop[a], oldPop[b]);
		
	newPop[threadIdx.x].mutate();
	//crossover(newPop[threadIdx.x]);

	boat[(blockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
	__syncthreads();
	newPop[0] = boat[blockIdx.x];
		
	__syncthreads();
		
	oldPop[threadIdx.x] = newPop[threadIdx.x];
    }

    pop[threadIdx.x + blockIdx.x * blockDim.x] = newPop[threadIdx.x];
}

float avg_value(Individual* pop, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
	sum += pop[i].value;
    return sum / n;
}

float avg_fitness(Individual* pop, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
	sum += pop[i].fitness;
    return sum / n;
}

int main()
{
    Individual pop[NUM_ISLANDS * POP_PER_ISLAND];
    Individual *d_pop;
    cudaMalloc((void**) &d_pop,
	       sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND));
    cudaMemcpy(d_pop, pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
	       cudaMemcpyHostToDevice);

    Individual boat[NUM_ISLANDS];
    Individual *d_boat;
    cudaMalloc((void**) &d_boat, sizeof(Individual) * NUM_ISLANDS);
    cudaMemcpy(d_boat, boat, sizeof(Individual) * NUM_ISLANDS,
	       cudaMemcpyHostToDevice);

    printf("Average fitness BEFORE: %f\n", avg_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
    printf("Average value BEFORE: %f\n", avg_value(pop, NUM_ISLANDS * POP_PER_ISLAND));

    evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat);

    cudaMemcpy(pop, d_pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
	       cudaMemcpyDeviceToHost);

    printf("Average fitness AFTER: %f\n", avg_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
    printf("Average value AFTER: %f\n", avg_value(pop, NUM_ISLANDS * POP_PER_ISLAND));
}

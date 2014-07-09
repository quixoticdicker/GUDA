#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "pharaohrand.h"

#define POP_PER_ISLAND 32
#define NUM_ISLANDS 7
#define RATE 0.01
#define GENERATIONS 100
#define SELECT_MUTATION 1
#define STRING_LEN 100

struct Individual {
    bool value[STRING_LEN];
    int fitness;

    __device__ void init();
    __device__ void mutate();
    __device__ void evaluate();
    __host__ __device__ int getFitness();
    __device__ void destroy();
};

__device__ void Individual::init() {
    //value = (bool*) malloc(STRING_LEN * sizeof(bool));
    for(int i = 0; i < STRING_LEN; i++)
    {
	value[i] = pharaohRand() > 0.5;
    }
    evaluate();
}

__device__ void Individual::destroy()
{
    //free(value);
}

__device__ void Individual::mutate()
{
#ifdef SELECT_MUTATION
	Individual oldI = *this;
#endif	
    for(int i = 0; i < STRING_LEN; i++)
    {
	if(pharaohRand() < RATE)
	{
	    value[i] = !value[i];
	}
    }
    evaluate();
#ifdef SELECT_MUTATION
	if(oldI.getFitness() > getFitness())
	{
		*this = oldI;
	}
#endif
}

__device__ void Individual::evaluate()
{
    fitness = 0;
    for (int i = 0; i < STRING_LEN; i++)
    {
	if (value[i])
	    fitness++;

    }
}

__host__ __device__ int Individual::getFitness()
{
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

    oldPop[threadIdx.x].init();

    __syncthreads();

    boat[blockIdx.x] = oldPop[0];
	
    for (int g = 0; g < GENERATIONS; g++)
    {
	int a = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
	int b = (int)(pharaohRand() * RAND_MAX) % POP_PER_ISLAND;
	newPop[threadIdx.x] = arena(oldPop[a], oldPop[b]);
		
	newPop[threadIdx.x].mutate();
	//crossover(newPop[threadIdx.x]);

	__syncthreads();

	boat[(blockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
	__syncthreads();
	newPop[0] = boat[blockIdx.x];
		
	__syncthreads();
		
	oldPop[threadIdx.x] = newPop[threadIdx.x];
	__syncthreads();
    }

    pop[threadIdx.x + blockIdx.x * blockDim.x] = oldPop[threadIdx.x];

    oldPop[threadIdx.x].destroy();
    newPop[threadIdx.x].destroy();
}

/*
char* best_value(Individual* pop, int n)
{
    int best_i = 0;
    float best_fit = 0.0f;

    for (int i = 0; i < n; i++)
    {
	if (pop[i].getFitness() > best_fit)
	{
	    best_fit = pop[i].getFitness();
	    best_i = i;
	}
    }

    return pop[best_i].value;
}
*/

int best_fitness(Individual* pop, int n)
{
    int best = 0;
    for (int i = 0; i < n; i++)
    {
	printf("%d ", pop[i].getFitness());
	if (pop[i].getFitness() > best)
	    best = pop[i].getFitness();
    }
    printf("\n");
    return best;
}

float avg_fitness(Individual* pop, int n)
{
    long int sum = 0;
    for (int i = 0; i < n; i++)
	sum += pop[i].fitness;
    return sum / n;
}

int main()
{
    Individual *d_pop;
    cudaMalloc((void**)&d_pop, sizeof(Individual)*NUM_ISLANDS*POP_PER_ISLAND);
    Individual *d_boat;
    cudaMalloc((void**)&d_boat, sizeof(Individual)*NUM_ISLANDS);

    evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat);

    Individual *pop = (Individual *)malloc(sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND));
    cudaMemcpy(pop, d_pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
	       cudaMemcpyDeviceToHost);

    printf("Average fitness AFTER: %f\n", avg_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
    printf("Best fitness AFTER: %d\n", best_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
    //printf("Best value AFTER: %lu\n", best_value(pop, NUM_ISLANDS * POP_PER_ISLAND));
}

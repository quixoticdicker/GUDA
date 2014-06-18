#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define POP_PER_ISLAND 256
#define NUM_ISLANDS 7

__device__ float pharaohRand();

struct Individual {
    double value;
    double fitness;

    Individual();
	void mutate();
	void evaluate();
};

__device__ Individual::Individual() {
	// replace with something random
	value = pharaohRand() * 2;
	fitness = -RAND_MAX;
}

__device__ void Individual::mutate()
{
    // replace with something random
	value *= 1 + (pharaohRand() - 0.5) / 5.0;
}

__device__ void Individual::evaluate()
{
    fitness = -fabs(2.0 - value * value);
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

__global__ void evolve(Individual* pop, Individual* boat)
{
	__shared__ Individual oldPop[POP_PER_ISLAND];
	__shared__ Individual newPop[POP_PER_ISLAND];
	int g;
	
	oldPop[threadIdx.x] = pop[threadIdx.x + blockIdx.x * blockDim.x];
	boat[blockIdx.x] = oldPop[0];
	
	for(g = 0; g < 100; g++)
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

__device__ float pharaohRand()
{
	int seed = clock();
	curandState_t myState;
	curand_init(seed, blockIdx.x * blockDim.x, threadIdx.x, &myState);
	
	return curand_uniform(&myState);
}

double avg_fitness(Individual* pop, int n)
{
    double sum = 0.0;
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

    evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat);

    cudaMemcpy(pop, d_pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
	cudaMemcpyDeviceToHost);

    printf("Average fitness AFTER: %f\n", avg_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "pharaohrand.h"

#define POP_PER_ISLAND 256
#define NUM_ISLANDS 7
#define RATE 0.03
#define GENERATIONS 100

#define NUM_ROIDS 3

__const__ float roids_x[] = {0.00f, 3.00f, 3.00f};
__const__ float roids_y[] = {0.00f, 0.00f, 3.00f};
__const__ float roids_r[] = {1.00f, 1.00f, 1.00f};

struct Individual {
    float theta;
	float x, y;
    float fitness;

	__device__ void init();
    __device__ void mutate();
    __device__ void evaluate();
	__host__ __device__ int getFitness();
	__device__ void destroy();
};

__device__ Individual::init() {	
	x = pharaohRand() * 2000000.0f - 1000000.0f;
	y = pharaohRand() * 2000000.0f - 1000000.0f;
	theta = pharaohRand() * 2 * M_PI;
	
	evaluate();
}

__device__ void Individual::mutate()
{
	// xMutagen, yMutagen, mMutagen
	short xMutagen = 0;
	short yMutagen = 0;
	short thetaMutagen = 0;
	short i;
	for(i = 0; i < sizeof(long) * 8; i++)
	{
		if(pharaohRand() < RATE)
		{
			xMutagen++;
		}
		if(pharaohRand() < RATE)
		{
			yMutagen++;
		}
		if(pharaohRand() < RATE)
		{
			thetaMutagen++;
		}
		xMutagen <<= 1;
		yMutagen <<= 1;
		thetaMutagen <<= 1;
	}
	i = * (short *) &x; // evil floating point bit level hacking
	i ^= xMutagen;
	x = * (float *) &i;
	
	i = * (short *) &y;
	i ^= yMutagen;
	y = * (float *) &i;
	
	i = * (short *) &alpha;
	i ^= thetaMutagen;
	alpha = * (float *) &i;
	
	evaluate();
}

__device__ void Individual::evaluate()
{
	float m, b, mp, bp, xa, ya, xs, ys, d;
	int i;
	fitness = 0;
	m = sinf(theta) / cosf(theta);
	b = y - m * x;
	mp = -1.0f / m;
	
	for(i = 0; i < NUM_ROIDS; i++)
	{
		xa = roid_x[i];
		ya = roid_y[i];
		bp = ya - mp * xa;
		
		xs = (bp - b)/(m - mp);
		ys = m * xs + b;
		
		xd = xp - xs;
		yd = yp - ys;
		d = sqrtf(xd * xd + yd * dy);
		
		if(d < roid_r[i]) fitness++;
	}
}

__host__ __device__ int Individual::getFitness()
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

__global__ void evolve(Individual* pop, Individual* boat)
{
    __shared__ Individual oldPop[POP_PER_ISLAND];
    __shared__ Individual newPop[POP_PER_ISLAND];
	
    oldPop[threadIdx.x].init();
	
	__syncthreads();
	
    boat[blockIdx.x] = oldPop[0];
	
    for (int g = 0; g < GENERATIONS; g++)
    {
	//oldPop[threadIdx.x].evaluate();
		
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

    pop[threadIdx.x + blockIdx.x * blockDim.x] = newPop[threadIdx.x];
	
	oldPop[threadIdx.x].destroy();
	newPop[threadIdx.x].destroy();
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

float best_fitness(Individual* pop, int n)
{
	float max = pop[0].fitness;
	for (int i = 1; i < n; i++)
	{
		if(pop[i].fitness > max)
		{
			max = pop[i].fitness;
		}
	}
	return max;
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
	printf("Most asteroids destroyed: %f\n", best_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND)); 
}

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
#define SELECT_MUTATION 1

#define IMG_HEIGHT 256
#define IMG_WIDTH 256
#define NUM_TRIANGLES 35

struct Triangle {
	float x1, y1, x2, y2, x3, y3;
	float alpha;
	float r, g, b;
	
	__device__ void init();
	__device__ void init(float x1, float y1, float x2, float y2, float x3, float y3, float a, float r, float g, float b);
	__device__ void init(Triangle that);
	__mutate__ void mutate();
}

__device__ void Triangle::init()
{
	x1 = pharaohRand() * 256.0f;
	x2 = pharaohRand() * 256.0f;
	x3 = pharaohRand() * 256.0f;
	y1 = pharaohRand() * 256.0f;
	y2 = pharaohRand() * 256.0f;
	y3 = pharaohRand() * 256.0f;
	a = pharaohRand() * 256.0f;
	r = pharaohRand() * 256.0f;
	g = pharaohRand() * 256.0f;
	b = pharaohRand() * 256.0f;
}

__device__ void init(float x1, float y1, float x2, float y2, float x3, float y3, float a, float r, float g, float b)
{
	this.x1 = x1;
	this.y1 = y1;
	this.x2 = x2;
	this.y2 = y2;
	this.x3 = x3;
	this.y3 = y3;
	this.a = a;
	this.r = r;
	this.g = g;
	this.b = b;
}

__device__ void init(Triangle that)
{
	this.init(that.x1, that.y1, that.x2, that.y2, that.x3, that.y3, that.a, that.r, that.g, that.b)
}

__device__ void mutate()
{
	int fieldNumber = (int) (pharaohRand() * 10);
	float c = (float) (pharaohRand() * 100.0f - 50.0f);
	switch(fieldNumber)
	{
		case 0:	x1 = (float) min(max(0, x1 + c), 255); break;
 		case 1: y1 = (float) min(max(0, y1 + c), 255); break;
		case 2: x2 = (float) min(max(0, x2 + c), 255); break;
		case 3: y2 = (float) min(max(0, y2 + c), 255); break;
		case 4: x3 = (float) min(max(0, x3 + c), 255); break;
		case 5: y3 = (float) min(max(0, y3 + c), 255); break;
		case 6: a = (float) min(max(0, a + c), 255); break;
		case 7: r = (float) min(max(0, r + c), 255); break;
		case 8: g = (float) min(max(0, g + c), 255); break;
		case 9: b = (float) min(max(0, b + c), 255); break;
	}
}

struct Individual {
	Triangle triangles[NUM_TRIANGLES];
    float fitness;

	__device__ void init();
    __device__ void mutate();
    __device__ void evaluate();
	__host__ __device__ float getFitness();
	__device__ void destroy() {};
};

__device__ void Individual::init() {
	for(int i = 0; i < NUM_TRIANGLES; i++)
	{
		triangles[i] = Triangle();
		triangles[i].init();
	}

	evaluate();
}

__device__ void Individual::mutate()
{
#ifdef SELECT_MUTATION
	Individual oldI = *this;
#endif	
	// xMutagen, yMutagen, mMutagen
	int xMutagen = 0;
	int yMutagen = 0;
	int thetaMutagen = 0;
	int i;
	for(i = 0; i < sizeof(float) * 8; i++)
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
	i = * (int *) &x; // evil floating point bit level hacking
	i ^= xMutagen;
	x = * (float *) &i;
	
	i = * (int *) &y;
	i ^= yMutagen;
	y = * (float *) &i;
	
	i = * (int *) &theta;
	i ^= thetaMutagen;
	theta = * (float *) &i;
	
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
	float m, b, mp, bp, xa, ya, xs, ys, d, xd, yd;
	int i;
	fitness = 0.0f;
	m = sinf(theta) / cosf(theta);
	b = y - m * x;
	mp = -1.0f / m;
	
	for(i = 0; i < NUM_ROIDS; i++)
	{
		xa = roids_x[i];
		ya = roids_y[i];
		bp = ya - mp * xa;
		
		xs = (bp - b)/(m - mp);
		ys = m * xs + b;
		
		xd = xa - xs;
		yd = ya - ys;
		d = sqrtf(xd * xd + yd * yd);
		
		if(true) fitness++; //d < roids_r[i]) fitness++;
	}
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
    //printf("Average value BEFORE: %f\n", avg_value(pop, NUM_ISLANDS * POP_PER_ISLAND));

    evolve<<<NUM_ISLANDS, POP_PER_ISLAND>>>(d_pop, d_boat);

    cudaMemcpy(pop, d_pop, sizeof(Individual) * (NUM_ISLANDS * POP_PER_ISLAND),
	       cudaMemcpyDeviceToHost);

    printf("Average fitness AFTER: %f\n", avg_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND));
    //printf("Average value AFTER: %f\n", avg_value(pop, NUM_ISLANDS * POP_PER_ISLAND));
	printf("Most asteroids destroyed: %f\n", best_fitness(pop, NUM_ISLANDS * POP_PER_ISLAND)); 
}
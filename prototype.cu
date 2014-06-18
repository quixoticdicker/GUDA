
struct Individual {
  double fitness;
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
		fitness(oldPop[threadIdx.x]);
		
		int a = rand() % POP_PER_ISLAND;
		int b = rand() % POP_PER_ISLAND;
		newPop[threadIdx.x] = evaluation(oldPop[a], oldPop[b]);
		
		mutate(newPop[threadIdx.x]);
		crossover(newPop[threadIdx.x]);

		boat[(bockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
		__syncthreads()
		newPop[0] = boat[blockIdx.x];
		
		__syncthreads();
		
		oldPop[threadIdx.x] = newPop[threadIdx.x];
	}
}

__device__ Individual evaluation(Individual a, Individual b)
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

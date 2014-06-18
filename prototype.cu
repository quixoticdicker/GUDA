__global__ void evolve()
{
	__shared__ Individual oldPop[POP_PER_ISLAND];
	__shared__ Individual newPop[POP_PER_ISLAND];
	int g;
	
	oldPop[threadIdx.x] = pop[threadIdx.x + blockIdx.x * blockDim.x];
	
	for(g = 0; g < 100; g++)
	{
		fitness(oldPop[threadIdx.x]);
		
		int a = rand() % POP_PER_ISLAND;
		int b = rand() % POP_PER_ISLAND;
		newPop[threadIdx.x] = evaluation(oldPop[a], oldPop[b]);
		
		mutate(newPop[threadIdx.x]);
		crossover(newPop[threadIdx.x]);
		
		if(idx == 0)
		{
			boat[(bockIdx.x + 1) % NUM_ISLANDS] = newPop[0];
			newPop[0] = boat[blockIdx.x];
		}
		__threadsynch();
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